# ═══════════════════════════════════════════════════════════════════════════════
#  PANTHEON ENGINE v3.0 — Orquestrador Supremo
#  Substitui o run_ensemble() do Leviathan original.
#
#  INOVAÇÕES vs Leviathan v2.0:
#  1. 12 experts (vs 9) com SYBIL, CHAOS, HERMES, ATLAS, TITAN
#  2. ORACLE Q-Learning: pesos adaptativos por micro-regime (vs gradiente fixo)
#  3. Dempster-Shafer Fusion: combina evidências sem inflar confiança falsa
#  4. 18 micro-regimes (vs 5 regimes genéricos)
#  5. Correlation Guard: bloqueia quando experts usam mesma fonte
#  6. Regime-Shift Freeze: pausa após mudança detectada pelo HERMES
# ═══════════════════════════════════════════════════════════════════════════════

import json
import math
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Experts originais do Leviathan ───────────────────────────────────────────
from analisador import (
    expert_miner, expert_catalog, expert_markov, expert_streak,
    expert_white_cycle, expert_momentum, expert_alternation,
    expert_volatility, expert_antidrift,
    detect_regime, poisson_white_hazard, streak_info,
    alternation_ratio, kelly_fraction, calibrate_confidence,
    get_neural_weights, update_neural_weights,
    get_sys_config, set_sys_config, _conn,
    bayes_prob, wilson_lower, cs, cn,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_START,
    THRESHOLD_STEP_UP, THRESHOLD_STEP_DOWN,
    BANCA_NORMAL_LOSSES, BANCA_ALERT_LOSSES,
    BANCA_LOCKDOWN_ROUNDS, BANCA_ALERT_ROUNDS, BANCA_RECOVERY_WINS,
    _threshold_state, _threshold_lock,
    GLOBAL_MINED_STRATS, GLOBAL_CATALOG_STRATS,
)

# ── Novos agentes ─────────────────────────────────────────────────────────────
from agents.agent_sybil  import expert_sybil
from agents.agent_chaos  import expert_chaos
from agents.agent_hermes import expert_hermes, _hermes_state
from agents.agent_oracle import (
    oracle_get_weights,
    oracle_learn,
    oracle_save_state,
    oracle_load_state,
    ORACLE_EXPERTS,
    _oracle_state,
)
from agents.agent_atlas  import expert_atlas, atlas_rebuild_tree
from agents.agent_titan  import expert_titan, titan_rebuild

log = logging.getLogger("leviathan")

# ─────────────────────────── CONSTANTES ──────────────────────────────────────
NUM_EXPERTS_PANTHEON  = 12
MIN_VOTES_PANTHEON    = 3
CONFIDENCE_FLOOR_P    = 0.66
THRESHOLD_START_P     = 0.74

# Correlation Guard: experts agrupados por fonte de informação
# Se 2+ experts do mesmo grupo votam igual, penaliza redundância
CORRELATION_GROUPS = {
    "pattern":     ["miner", "titan", "atlas"],       # mineração de padrões
    "statistical": ["markov", "streak", "alternation"], # estatística clássica
    "spectral":    ["sybil", "chaos"],                  # análise espectral/entropia
    "contextual":  ["momentum", "volatility"],          # desvio de distribuição
    "protective":  ["antidrift", "hermes"],             # detectores de mudança
    "catalog":     ["catalog", "white"],                # catálogo e branco
}

# Freeze após mudança de regime
REGIME_SHIFT_FREEZE_ROUNDS = 8
_pantheon_state = {
    "regime_freeze_until": 0,
    "total_rounds":        0,
    "last_regime":         None,
    "consecutive_losses":  0,
    "consecutive_wins":    0,
    "banca_level":         "NORMAL",
    "lockdown_until":      0,
    "recovery_wins":       0,
    "history":             deque(maxlen=300),
}
_pantheon_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
#  MICRO-REGIME ENCODER (18 micro-regimes vs 5 originais)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_micro_regime(colors):
    """
    Classifica em 18 micro-regimes combinando 3 dimensões:
    - Regime base (6): white_zone, streak_red, streak_black,
                       alternating, chaotic, balanced
    - Volatilidade (3): high, medium, low
    → 6 × 3 = 18 micro-regimes possíveis
    """
    base   = detect_regime(colors)
    nw     = [c for c in colors if c != 0]
    strk   = streak_info(nw)

    # Subdivide streak por cor
    base_name = base["name"]
    if base_name == "streak_hot":
        if strk["color"] == 1:
            base_name = "streak_red"
        else:
            base_name = "streak_black"

    # Volatilidade
    alt10 = alternation_ratio(colors, 10)
    if alt10 >= 0.75:
        vol = "high_vol"
    elif alt10 <= 0.35:
        vol = "low_vol"
    else:
        vol = "med_vol"

    micro = f"{base_name}_{vol}"
    base["micro_regime"] = micro
    return base


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMPSTER-SHAFER EVIDENCE FUSION
# ═══════════════════════════════════════════════════════════════════════════════

def dempster_shafer_fusion(expert_results, neural_weights):
    """
    Combina evidências dos experts via Teoria de Dempster-Shafer.

    Diferença vs soma ponderada simples:
    - Trata incerteza explicitamente (massa atribuída a {V, P, ∅})
    - Conflito entre experts é penalizado matematicamente
    - Não infla confiança quando experts se contradizem

    Frame de discernimento: Θ = {VERMELHO, PRETO}
    Massas básicas de probabilidade (BPA):
      m({V}) = confiança no vermelho
      m({P}) = confiança no preto
      m({V,P}) = incerteza (quando expert abstém ou conf baixa)
    """
    # Inicializa massas combinadas
    m_red   = 0.0   # massa em VERMELHO
    m_black = 0.0   # massa em PRETO
    m_unc   = 1.0   # massa em {V,P} (incerteza total)

    conflict_total = 0.0
    n_voters = 0

    for exp in expert_results:
        if exp["vote"] is None:
            continue

        src  = exp.get("source", "unknown")
        w    = neural_weights.get(src, 1.0)
        conf = exp["confidence"] * w

        # Normaliza conf para [0, 0.95]
        conf = min(conf, 0.95)

        # BPA deste expert
        if exp["vote"] == 1:
            bpa_red   = conf
            bpa_black = 0.0
        else:
            bpa_red   = 0.0
            bpa_black = conf

        bpa_unc = 1.0 - conf   # incerteza residual

        # Regra de combinação de Dempster (versão iterativa)
        # Combina (m_red, m_black, m_unc) com (bpa_red, bpa_black, bpa_unc)
        new_red   = m_red * bpa_red + m_red * bpa_unc + m_unc * bpa_red
        new_black = m_black * bpa_black + m_black * bpa_unc + m_unc * bpa_black
        new_unc   = m_unc * bpa_unc

        # Conflito: evidências contraditórias
        conflict  = m_red * bpa_black + m_black * bpa_red
        conflict_total += conflict

        # Normaliza (1 - K) onde K é o conflito acumulado
        k_factor = 1.0 - conflict
        if k_factor < 0.01:
            k_factor = 0.01   # evita divisão por zero

        m_red   = new_red   / k_factor
        m_black = new_black / k_factor
        m_unc   = new_unc   / k_factor

        n_voters += 1

    if n_voters == 0:
        return None, 0.0, 0.0, 0.0, 0.0, 0.0

    # Decisão: cor com maior massa
    if m_red > m_black:
        winner   = 1
        ds_conf  = m_red
    elif m_black > m_red:
        winner   = 2
        ds_conf  = m_black
    else:
        return None, 0.0, conflict_total, 0.0, 0.0, 0.0

    # Penalidade por conflito alto
    conflict_penalty = min(conflict_total * 0.15, 0.20)
    ds_conf = max(0.0, ds_conf - conflict_penalty)

    return winner, round(ds_conf, 4), round(conflict_total, 4), round(m_red, 4), round(m_black, 4), round(m_unc, 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORRELATION GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _correlation_penalty(expert_results, winner):
    """
    Detecta quando experts do mesmo grupo de correlação votam todos igual.
    Retorna penalidade de confiança [0.0, 0.25].

    Motivação: 3 experts baseados em padrões concordando não é
    evidência independente — é correlação espúria.
    """
    penalty = 0.0

    for group_name, group_members in CORRELATION_GROUPS.items():
        group_votes = [
            e for e in expert_results
            if e.get("source") in group_members and e["vote"] == winner
        ]
        group_against = [
            e for e in expert_results
            if e.get("source") in group_members and e["vote"] is not None
            and e["vote"] != winner
        ]

        # Se todos de um grupo votaram igual e nenhum discordou
        if len(group_votes) >= 2 and len(group_against) == 0:
            # Penalidade proporcional ao tamanho do grupo concordante
            penalty += 0.04 * (len(group_votes) - 1)
            log.debug("🛡️ CorrelationGuard: grupo '%s' %d concordantes → penalidade %.2f",
                      group_name, len(group_votes), penalty)

    return min(penalty, 0.25)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORQUESTRADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_pantheon(colors, regime, last_round=None):
    """
    PANTHEON ENGINE v3.0 — Orquestrador Supremo.
    Substitui run_ensemble() do Leviathan v2.0.
    """

    with _pantheon_lock:
        total_rounds  = _pantheon_state["total_rounds"]
        freeze_until  = _pantheon_state["regime_freeze_until"]
        banca_level   = _pantheon_state["banca_level"]
        lockdown_until = _pantheon_state["lockdown_until"]
        last_regime   = _pantheon_state["last_regime"]

    # ── FASE 0: Regime-Shift Freeze ──────────────────────────────────────────
    current_regime_name = regime.get("micro_regime", regime["name"])

    if last_regime and last_regime != current_regime_name:
        freeze_end = total_rounds + REGIME_SHIFT_FREEZE_ROUNDS
        with _pantheon_lock:
            _pantheon_state["regime_freeze_until"] = freeze_end
            _pantheon_state["last_regime"]         = current_regime_name
        log.info("❄️ REGIME SHIFT: %s → %s | Freeze %d rounds",
                 last_regime, current_regime_name, REGIME_SHIFT_FREEZE_ROUNDS)

    with _pantheon_lock:
        _pantheon_state["last_regime"] = current_regime_name
        freeze_until = _pantheon_state["regime_freeze_until"]

    if total_rounds < freeze_until:
        remaining = freeze_until - total_rounds
        return {
            "action": "wait", "color": None, "confidence": 0.0,
            "votes": [], "reason": f"❄️ FREEZE pós-mudança de regime — {remaining} rounds restantes.",
            "kelly": 0.0, "vote_count": 0,
        }

    # ── FASE 1: Coleta todos os 12 experts ───────────────────────────────────
    expert_results = [
        expert_miner(colors, regime),          #  1 — original
        expert_catalog(colors, regime),        #  2 — original
        expert_markov(colors, regime),         #  3 — original
        expert_streak(colors, regime),         #  4 — original
        expert_white_cycle(colors, regime),    #  5 — original
        expert_momentum(colors, regime),       #  6 — original
        expert_alternation(colors, regime),    #  7 — original
        expert_volatility(colors, regime),     #  8 — original
        expert_antidrift(colors, regime),      #  9 — original
        expert_sybil(colors, regime),          # 10 — NOVO
        expert_chaos(colors, regime),          # 11 — NOVO
        expert_hermes(colors, regime),         # 12 — NOVO
        expert_atlas(colors, regime),          # 13 — NOVO
        expert_titan(colors, regime),          # 14 — NOVO
    ]

    # ── FASE 2: VETOs ─────────────────────────────────────────────────────────
    for exp in expert_results:
        if exp.get("veto"):
            return {
                "action": "block", "color": None, "confidence": 0.0,
                "votes": expert_results, "reason": f"🛡️ VETO: {exp['label']}",
                "kelly": 0.0, "vote_count": 0,
            }

    # ── FASE 3: Pesos ORACLE × Neural (híbrido) ───────────────────────────────
    neural_weights_base = get_neural_weights()
    oracle_weights      = oracle_get_weights(colors, regime)

    # Fusão multiplicativa: ambos precisam concordar para peso alto
    hybrid_weights = {}
    all_sources = set(list(neural_weights_base.keys()) + list(oracle_weights.keys()))
    for src in all_sources:
        nw_val  = neural_weights_base.get(src, 1.0)
        orc_val = oracle_weights.get(src, 1.0)
        hybrid_weights[src] = round(
            math.sqrt(nw_val * orc_val),   # média geométrica
            3
        )

    # ── FASE 4: Dempster-Shafer Fusion ────────────────────────────────────────
    # inicializa valores padrão para persistência
    ds_winner = None
    ds_conf = 0.0
    ds_conflict = 0.0
    ds_mass_red = 0.0
    ds_mass_black = 0.0
    ds_res = dempster_shafer_fusion(expert_results, hybrid_weights)
    if ds_res:
        # desempacota: winner, conf, conflict, mass_red, mass_black, mass_unc
        ds_winner, ds_conf, ds_conflict, ds_mass_red, ds_mass_black, ds_mass_unc = ds_res

    if ds_winner is None:
        return {
            "action": "wait", "color": None, "confidence": 0.0,
            "votes": expert_results,
            "reason": "⏳ D-S: sem evidência suficiente.",
            "kelly": 0.0, "vote_count": 0,
            # Pantheon fields
            "micro_regime": current_regime_name,
            "ds_conflict": ds_conflict,
            "ds_mass_red": ds_mass_red,
            "ds_mass_black": ds_mass_black,
            "ds_mass_unc": ds_mass_unc,
            "oracle_weights": hybrid_weights,
            "oracle_q_states": len(_oracle_state.get("q_table", {})),
            "banca_level": banca_level,
        }

    # ── FASE 5: Correlation Guard ─────────────────────────────────────────────
    corr_penalty = _correlation_penalty(expert_results, ds_winner)
    ds_conf      = max(0.0, ds_conf - corr_penalty)

    if corr_penalty > 0.05:
        log.info("🛡️ CorrelationGuard: penalidade=%.3f → conf ajustada=%.3f",
                 corr_penalty, ds_conf)

    # ── FASE 6: Hot bonus do AntiDrift ────────────────────────────────────────
    hot_bonus = sum(e.get("hot_bonus", 0.0) for e in expert_results)
    ds_conf   = min(ds_conf + hot_bonus, 0.97)

    # ── FASE 7: Calibração final ──────────────────────────────────────────────
    votes_for_winner = sum(1 for e in expert_results
                           if e["vote"] == ds_winner)
    cal_conf = calibrate_confidence(ds_conf, votes_for_winner, NUM_EXPERTS_PANTHEON)
    cal_conf = round(max(0.0, min(cal_conf, 0.97)), 4)

    # ── FASE 8: Lockdown / Banca Protection ──────────────────────────────────
    with _pantheon_lock:
        threshold      = _threshold_state.get("value", THRESHOLD_START_P)
        lockdown_until = _threshold_state.get("lockdown_until", 0)
        total_r        = _threshold_state.get("total_rounds_seen", 0)
        banca_level    = _threshold_state.get("banca_level", "NORMAL")

    if total_r < lockdown_until:
        remaining = lockdown_until - total_r
        return {
            "action": "wait", "color": None, "confidence": cal_conf,
            "votes": expert_results,
            "reason": f"🔒 LOCKDOWN [{banca_level}] — {remaining} rounds restantes.",
            "kelly": 0.0, "vote_count": votes_for_winner,
            # Pantheon fields
            "micro_regime": current_regime_name,
            "ds_conflict": ds_conflict,
            "ds_mass_red": ds_mass_red,
            "ds_mass_black": ds_mass_black,
            "ds_mass_unc": ds_mass_unc,
            "oracle_weights": hybrid_weights,
            "oracle_q_states": len(_oracle_state.get("q_table", {})),
            "banca_level": banca_level,
        }

    # ── FASE 9: Threshold + mínimo de experts ────────────────────────────────
    if cal_conf < threshold or votes_for_winner < MIN_VOTES_PANTHEON:
        parts = []
        if cal_conf < threshold:
            parts.append(f"edge {cal_conf:.0%} < thr {threshold:.0%}")
        if votes_for_winner < MIN_VOTES_PANTHEON:
            parts.append(f"experts {votes_for_winner}/{MIN_VOTES_PANTHEON}")
        top = [e["label"] for e in expert_results if e["vote"] is not None][:3]
        return {
            "action": "wait", "color": None, "confidence": cal_conf,
            "votes": expert_results,
            "reason": f"⏳ Aguardando: {'; '.join(parts)}. [{', '.join(top)}]",
            "kelly": 0.0, "vote_count": votes_for_winner,
            # Pantheon fields
            "micro_regime": current_regime_name,
            "ds_conflict": ds_conflict,
            "ds_mass_red": ds_mass_red,
            "ds_mass_black": ds_mass_black,
            "ds_mass_unc": ds_mass_unc,
            "oracle_weights": hybrid_weights,
            "oracle_q_states": len(_oracle_state.get("q_table", {})),
            "banca_level": banca_level,
        }

    # ── FASE 10: Sinal de entrada ─────────────────────────────────────────────
    alvo_label   = "VERMELHO 🔴" if ds_winner == 1 else "PRETO ⚫"
    contributing = [e["label"] for e in expert_results if e["vote"] == ds_winner]
    kelly_val    = kelly_fraction(cal_conf)
    after_str    = (f" [após {last_round['roll']} ({cs(last_round['color'])})]"
                    if last_round else "")

    conflict_str = f" D-S_K={ds_conflict:.2f}" if ds_conflict > 0.15 else ""

    return {
        "action":     "enter",
        "color":      ds_winner,
        "confidence": cal_conf,
        "votes":      expert_results,
        "reason":     (f"🌌 PANTHEON {alvo_label}{after_str} | "
                       f"Edge {cal_conf:.0%} | "
                       f"Experts {votes_for_winner}/{NUM_EXPERTS_PANTHEON} | "
                       f"D-S conf={ds_conf:.2f}{conflict_str} | "
                       f"[{'; '.join(contributing[:3])}]"),
        "kelly":      round(kelly_val * 100, 2),
        "vote_count": votes_for_winner,
        # ── Campos Pantheon para persistência ──────────────────────────
        "micro_regime":   current_regime_name,
        "ds_conflict":    ds_conflict,
        "ds_mass_red":    ds_mass_red,
        "ds_mass_black":  ds_mass_black,
        "ds_mass_unc":    ds_mass_unc,
        "oracle_weights": hybrid_weights,
        "oracle_q_states": len(_oracle_state.get("q_table", {})),
        "banca_level":    banca_level,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ATUALIZAÇÃO DE APRENDIZADO (substitui update_neural_weights no main loop)
# ═══════════════════════════════════════════════════════════════════════════════

def pantheon_learn(module_votes: dict, actual_color: int, won: bool, colors, regime):
    """
    Aprendizado completo do Pantheon:
    1. Atualiza pesos neurais (sistema original, correção de gradiente)
    2. Atualiza Q-Table do ORACLE (RL)
    3. Salva ORACLE periodicamente
    """
    # 1. Pesos neurais (sistema original mantido)
    update_neural_weights(module_votes, actual_color, won)

    # 2. Q-Learning ORACLE
    oracle_learn(won, colors, regime)

    # 3. Persiste ORACLE a cada 20 atualizações
    from agents.agent_oracle import _oracle_state
    if _oracle_state["total_updates"] % 20 == 0:
        oracle_save_state()
        log.info("🧠 ORACLE salvo | Q-states=%d", len(_oracle_state["q_table"]))


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def pantheon_init(colors=None):
    """
    Inicializa todos os componentes do Pantheon.
    Chamar no main() antes do loop principal.
    """
    log.info("═" * 90)
    log.info("  🌌 PANTHEON ENGINE v3.0 — 14 EXPERTS | D-S FUSION | ORACLE RL")
    log.info("  SYBIL(FFT) | CHAOS(PE) | HERMES(BOCPD) | ATLAS(CTW) | TITAN(FP)")
    log.info("  Dempster-Shafer | Correlation Guard | Regime-Shift Freeze")
    log.info("  Oracle Q-Learning | 18 Micro-Regimes | Hybrid Neural×RL Weights")
    log.info("═" * 90)

    oracle_load_state()

    if colors and len(colors) >= 100:
        nw = [c for c in colors if c != 0]
        atlas_rebuild_tree(nw)
        titan_rebuild(nw)
        log.info("✅ ATLAS e TITAN inicializados com %d rounds", len(nw))
    else:
        log.info("⚠️ Pantheon init: aguardando dados suficientes para ATLAS/TITAN")

    log.info("✅ Pantheon Engine v3.0 pronto.")