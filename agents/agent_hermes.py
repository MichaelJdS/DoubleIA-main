# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE HERMES — Bayesian Online Change-Point Detector
#  Baseado em: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
#  Detecta QUANDO o jogo mudou de regime, não só qual é o regime atual.
#  Isso permite reagir ANTES que os outros experts percebam a mudança.
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import threading
import numpy as np

log = logging.getLogger("hermes")


def bayes_prob(wins, n, alpha=1.5):
    return (wins + alpha) / (n + 2 * alpha)

HERMES_HAZARD_RATE  = 1 / 40   # taxa de mudança esperada (1 a cada ~40 rounds)
HERMES_WINDOW       = 150      # janela de observação
HERMES_MIN_CONF     = 0.64     # confiança mínima para votar após mudança
HERMES_WARMUP       = 30       # rounds mínimos após uma mudança para votar

_hermes_state = {
    "last_changepoint": 0,       # índice do último change-point detectado
    "changepoint_prob":  0.0,    # probabilidade atual de mudança
    "post_cp_rounds":    0,      # rounds desde último change-point
    "run_length_probs":  None,   # distribuição de run-lengths (interna BOCPD)
    "alpha_r":           1.5,    # prior Beta (pseudo-red)
    "beta_r":            1.5,    # prior Beta (pseudo-black)
    "total_rounds_seen": 0,
}
_hermes_lock = threading.Lock()

def _bocpd_update(x, run_length_probs, alpha, beta, hazard):
    """
    Atualização BOCPD para dados Bernoulli (vermelho vs preto).
    x: observação atual (1=vermelho, 0=preto)
    run_length_probs: P(run_length=r | data_{1:t-1}) para r=0,1,...,t-1
    Retorna: novos run_length_probs, evidência de change-point, (alpha, beta) atualizados
    """
    n = len(run_length_probs)
    new_probs = np.zeros(n + 1)

    # Predictive probability para cada run length (modelo Beta-Binomial)
    alphas = np.array([alpha + sum(
        1 for _ in range(min(r, 50)) if True  # simplificado
    ) for r in range(n)], dtype=float)

    # Versão eficiente: mantém suficientes estatísticas por run-length
    # Aqui usamos uma aproximação com Beta global atualizada incrementalmente
    pred_probs = np.zeros(n)
    for r in range(n):
        a = alpha + r * 0.5  # aproximação
        b = beta  + r * 0.5
        pred_probs[r] = (a / (a + b)) if x == 1 else (b / (a + b))

    # Probabilidade de crescimento do run (sem change-point)
    growth_probs = run_length_probs * pred_probs * (1.0 - hazard)

    # Probabilidade de change-point (reset do run)
    cp_prob = np.sum(run_length_probs * pred_probs * hazard)

    new_probs[1:] = growth_probs
    new_probs[0]  = cp_prob

    # Normaliza
    total = np.sum(new_probs)
    if total > 0:
        new_probs /= total

    # Evidência de change-point = P(run_length = 0)
    cp_evidence = float(new_probs[0])

    return new_probs, cp_evidence

def expert_hermes(colors, regime):
    """
    HERMES — Bayesian Change-Point Detector.

    Após detectar um change-point com alta probabilidade:
    1. Identifica a distribuição NOVA (pós-mudança) — janela curta
    2. Compara com a distribuição ANTIGA (pré-mudança) — janela longa
    3. Se a nova distribuição tem bias claro → vota nessa direção
    
    Conceito: o jogo acabou de mudar de "personalidade" → 
    os primeiros rounds do novo regime são os mais previsíveis.
    """
    nw = [c for c in colors if c != 0]
    if len(nw) < 50:
        return {"vote": None, "confidence": 0.0,
                "label": "hermes:insuf", "key": "hermes", "source": "hermes"}

    window  = nw[-HERMES_WINDOW:]
    n       = len(window)

    with _hermes_lock:
        # Inicializa run_length_probs se necessário
        if _hermes_state["run_length_probs"] is None or \
           len(_hermes_state["run_length_probs"]) > 300:
            _hermes_state["run_length_probs"] = np.array([1.0])

        rl_probs = _hermes_state["run_length_probs"].copy()
        alpha    = _hermes_state["alpha_r"]
        beta     = _hermes_state["beta_r"]

    # Processa os últimos 30 rounds para atualizar estado
    recent_process = window[-30:]
    cp_probs_trace = []

    for c in recent_process:
        x = 1 if c == 1 else 0
        rl_probs, cp_ev = _bocpd_update(x, rl_probs, alpha, beta, HERMES_HAZARD_RATE)
        cp_probs_trace.append(cp_ev)

        # Atualiza suficientes estatísticas
        if c == 1:
            alpha = min(alpha + 0.3, 50.0)
        else:
            beta  = min(beta + 0.3, 50.0)

    with _hermes_lock:
        _hermes_state["run_length_probs"]  = rl_probs
        _hermes_state["alpha_r"]           = alpha
        _hermes_state["beta_r"]            = beta
        _hermes_state["changepoint_prob"]  = cp_probs_trace[-1] if cp_probs_trace else 0.0
        _hermes_state["total_rounds_seen"] += 1

    cp_prob_now = cp_probs_trace[-1] if cp_probs_trace else 0.0
    cp_prob_max = max(cp_probs_trace) if cp_probs_trace else 0.0

    log.debug("🔴 HERMES: CP_prob=%.3f CP_max=%.3f", cp_prob_now, cp_prob_max)

    # DETECÇÃO DE MUDANÇA RECENTE (alta probabilidade de change-point)
    if cp_prob_max >= 0.35:
        # Identifica quando (aproximadamente) foi a mudança
        cp_idx = cp_probs_trace.index(cp_prob_max)
        rounds_since_cp = len(recent_process) - cp_idx

        with _hermes_lock:
            _hermes_state["last_changepoint"] = n - rounds_since_cp
            _hermes_state["post_cp_rounds"]   = rounds_since_cp

        if rounds_since_cp > HERMES_WARMUP:
            return {"vote": None, "confidence": 0.0,
                    "label": f"hermes:pós_mudança_estabilizando({rounds_since_cp}r)",
                    "key": "hermes", "source": "hermes"}

        # Distribuição no NOVO regime (post change-point)
        post_cp_window = nw[-rounds_since_cp:] if rounds_since_cp > 0 else nw[-10:]
        if len(post_cp_window) < 5:
            return {"vote": None, "confidence": 0.0,
                    "label": "hermes:novo_regime_dados_insuf",
                    "key": "hermes", "source": "hermes"}

        r_new = post_cp_window.count(1)
        b_new = post_cp_window.count(2)
        total_new = r_new + b_new

        if total_new == 0:
            return {"vote": None, "confidence": 0.0,
                    "label": "hermes:sem_dados_novo_regime",
                    "key": "hermes", "source": "hermes"}

        # Distribuição no regime ANTERIOR (pre change-point)
        pre_window = nw[-(rounds_since_cp + 60):-rounds_since_cp] if rounds_since_cp < len(nw) else nw[:60]
        r_old = pre_window.count(1) if pre_window else 0
        b_old = pre_window.count(2) if pre_window else 0
        total_old = r_old + b_old

        prob_r_new = bayes_prob(r_new, total_new)
        prob_b_new = bayes_prob(b_new, total_new)

        margin = abs(prob_r_new - prob_b_new)

        if margin < 0.10:
            return {"vote": None, "confidence": 0.0,
                    "label": f"hermes:mudança_sem_bias(CP={cp_prob_max:.2f})",
                    "key": "hermes", "source": "hermes"}

        winner = 1 if prob_r_new > prob_b_new else 2

        # Confiança: quanto maior o CP e o bias do novo regime, mais confiante
        conf = min(0.50 + cp_prob_max * 0.35 + margin * 0.80, 0.93)

        # Bônus: se o novo regime é radicalmente diferente do antigo
        if total_old > 10:
            prob_r_old = r_old / total_old
            regime_shift_magnitude = abs(prob_r_new - prob_r_old - 0.5)
            conf = min(conf + regime_shift_magnitude * 0.3, 0.95)

        return {
            "vote":       winner,
            "confidence": round(conf, 4),
            "label":      (f"⚡ HERMES mudança_régime CP={cp_prob_max:.2f} "
                           f"+{rounds_since_cp}r → {'V' if winner==1 else 'P'} "
                           f"bias={margin:.2f}"),
            "key":        f"hermes_cp_{cp_prob_max:.2f}",
            "source":     "hermes",
        }

    # MODO ESTÁVEL: jogo sem mudança → HERMES confirma outros experts
    # (não vota, mas sinaliza estabilidade via hot_bonus)
    if cp_prob_now < 0.05:
        return {
            "vote":       None,
            "confidence": 0.0,
            "label":      f"hermes:estável(CP={cp_prob_now:.3f})",
            "key":        "hermes",
            "source":     "hermes",
            "hot":        True,
            "hot_bonus":  0.03,   # pequeno boost para outros experts em regime estável
        }

    return {"vote": None, "confidence": 0.0,
            "label": f"hermes:neutro(CP={cp_prob_now:.3f})",
            "key": "hermes", "source": "hermes"}