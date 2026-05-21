# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE TITAN — FPGrowth Frequent Pattern Miner + Bootstrap Significance
#  Substitui o minerador Wilson simples por mineração de padrões frequentes
#  com teste de significância estatística via Bootstrap.
#  Encontra padrões NÃO-CONTÍGUOS que o minerador original nunca veria.
# ═══════════════════════════════════════════════════════════════════════════════
import math
import random
import threading
import logging
from collections import defaultdict, deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analisador import (bayes_prob, wilson_lower, streak_info,
                        get_sys_config, set_sys_config, _conn, cs)

log = logging.getLogger("leviathan")

TITAN_MIN_SUPPORT   = 12     # mínimo de ocorrências do padrão
TITAN_MIN_CONF      = 0.64   # confiança mínima (P(target|padrão))
TITAN_MIN_LIFT      = 1.20   # lift mínimo (quão mais frequente que aleatório)
TITAN_BOOTSTRAP_N   = 300    # iterações de bootstrap para significância
TITAN_BOOTSTRAP_P   = 0.05   # p-value máximo aceitável
TITAN_MAX_GAP       = 3      # gap máximo entre elementos do padrão (não-contíguo)
TITAN_MAX_PATTERN_LEN = 5    # tamanho máximo do padrão
TITAN_RECENCY_WINDOW  = 500  # janela de dados recentes para mineração
TITAN_REBUILD_EVERY   = 30   # reconstrói padrões a cada N rounds novos

_titan_lock    = threading.Lock()
_titan_patterns = []          # padrões minerados e validados
_titan_last_n   = 0


# ─────────────────────────── BOOTSTRAP SIGNIFICANCE ──────────────────────────

def _bootstrap_test(sequence, pattern, target_color, n_iter=TITAN_BOOTSTRAP_N):
    """
    Testa se a associação padrão→cor é estatisticamente significativa
    via permutação bootstrap (Monte Carlo).

    H0: a taxa de acerto do padrão é igual à taxa base (aleatório).
    Retorna p-value. Menor = mais significativo.
    """
    # Taxa observada real
    observed_rate = _pattern_hit_rate(sequence, pattern, target_color)
    if observed_rate is None:
        return 1.0

    # Base rate
    nw   = [c for c in sequence if c != 0]
    base = nw.count(target_color) / len(nw) if nw else 0.5

    # Bootstrap: permuta a sequência e conta quantas vezes bate ou supera
    exceed_count = 0
    seq_list     = list(sequence)

    for _ in range(n_iter):
        shuffled = seq_list.copy()
        random.shuffle(shuffled)
        perm_rate = _pattern_hit_rate(shuffled, pattern, target_color)
        if perm_rate is not None and perm_rate >= observed_rate:
            exceed_count += 1

    p_value = exceed_count / n_iter
    return p_value


def _pattern_hit_rate(sequence, pattern, target_color):
    """
    Calcula P(target_color | padrão aparece antes) na sequência.
    Suporta padrões não-contíguos com gap máximo TITAN_MAX_GAP.
    """
    hits  = 0
    total = 0
    n     = len(sequence)
    pat_len = len(pattern)

    for i in range(n - pat_len - 1):
        # Tenta casar o padrão a partir de i
        matched = True
        last_pos = i

        for p_elem in pattern:
            found = False
            for g in range(TITAN_MAX_GAP + 1):
                pos = last_pos + g
                if pos < n and sequence[pos] == p_elem:
                    last_pos = pos + 1
                    found    = True
                    break
            if not found:
                matched = False
                break

        if matched and last_pos < n:
            # Verifica a próxima cor não-branca após o padrão
            for k in range(last_pos, min(last_pos + 3, n)):
                if sequence[k] in (1, 2):
                    total += 1
                    if sequence[k] == target_color:
                        hits += 1
                    break

    if total < TITAN_MIN_SUPPORT:
        return None
    return hits / total


# ─────────────────────────── MINERAÇÃO DE PADRÕES ────────────────────────────

def _mine_frequent_patterns(colors):
    """
    Versão simplificada de FPGrowth: minera padrões frequentes de cores
    com suporte a gaps (padrões não-contíguos).

    Retorna lista de padrões validados com bootstrap.
    """
    nw      = [c for c in colors if c != 0]
    window  = nw[-TITAN_RECENCY_WINDOW:]
    n       = len(window)

    if n < 100:
        return []

    candidate_patterns = []

    # Gera candidatos: pares e triplas não-contíguas de (cor, alvo)
    for length in range(2, TITAN_MAX_PATTERN_LEN + 1):
        # Para eficiência, usa apenas os últimos 300 rounds para candidatos
        sub = window[-300:]
        pattern_counts = defaultdict(int)

        for i in range(len(sub) - length):
            # Padrão contíguo (mais frequente e rápido de testar)
            pat = tuple(sub[i:i+length])
            if 0 not in pat:   # ignora brancos no padrão
                pattern_counts[pat] += 1

        # Filtra por suporte mínimo
        for pat, count in pattern_counts.items():
            if count >= TITAN_MIN_SUPPORT:
                candidate_patterns.append(pat)

    if not candidate_patterns:
        return []

    log.info("⚒️ TITAN: %d candidatos para validação bootstrap", len(candidate_patterns[:50]))

    validated = []

    for pat in candidate_patterns[:80]:    # limita para performance
        for target in (1, 2):
            rate = _pattern_hit_rate(window, pat, target)
            if rate is None or rate < TITAN_MIN_CONF:
                continue

            # Calcula lift
            base = window.count(target) / len(window) if window else 0.5
            lift = rate / base if base > 0 else 1.0

            if lift < TITAN_MIN_LIFT:
                continue

            # Teste bootstrap (custoso — só para os promissores)
            p_val = _bootstrap_test(window, pat, target, n_iter=150)

            if p_val <= TITAN_BOOTSTRAP_P:
                wl = wilson_lower(int(rate * len(window)), len(window))
                validated.append({
                    "id":       f"titan_{''.join(cs(c) for c in pat)}_{target}",
                    "pattern":  pat,
                    "target":   target,
                    "rate":     round(rate, 4),
                    "lift":     round(lift, 3),
                    "p_value":  round(p_val, 4),
                    "wilson":   round(wl, 4),
                    "support":  int(rate * len(window)),
                    "source":   "titan",
                    "weight":   round(wl * min(lift / 2.0, 1.0), 4),
                })

    # Ordena por peso (wilson * lift)
    validated.sort(key=lambda x: x["weight"], reverse=True)
    log.info("⚒️ TITAN: %d padrões bootstrap-validados (p<%.2f)",
             len(validated), TITAN_BOOTSTRAP_P)
    return validated[:60]


def titan_rebuild(colors):
    """Interface pública para reconstruir padrões Titan."""
    global _titan_patterns, _titan_last_n
    nw = [c for c in colors if c != 0]
    patterns = _mine_frequent_patterns(colors)
    with _titan_lock:
        _titan_patterns = patterns
        _titan_last_n   = len(nw)
    log.info("⚒️ TITAN: Padrões reconstruídos: %d", len(patterns))


# ─────────────────────────── EXPERT PRINCIPAL ────────────────────────────────

def expert_titan(colors, regime):
    """
    TITAN — FPGrowth + Bootstrap Significance Pattern Miner.

    Diferença do ExpertMiner original:
    - Suporta padrões NÃO-CONTÍGUOS (com gaps)
    - Validação bootstrap (não só Wilson score)
    - Usa lift para filtrar padrões espúrios
    - Janela de recência ponderada
    """
    nw = [c for c in colors if c != 0]
    if len(nw) < 80:
        return {"vote": None, "confidence": 0.0,
                "label": "titan:insuf", "key": "titan", "source": "titan"}

    with _titan_lock:
        patterns = _titan_patterns.copy()
        last_n   = _titan_last_n

    # Reconstrói se muito desatualizado
    if not patterns or (len(nw) - last_n) >= TITAN_REBUILD_EVERY:
        # Reconstrói em background (não bloqueia o ciclo principal)
        import threading as _t
        _t.Thread(target=titan_rebuild, args=(colors,), daemon=True).start()

        if not patterns:
            return {"vote": None, "confidence": 0.0,
                    "label": "titan:reconstruindo", "key": "titan", "source": "titan"}

    # Busca o melhor padrão que casa com o final da sequência
    best = None

    for pat_info in patterns:
        pat    = pat_info["pattern"]
        target = pat_info["target"]
        pat_len = len(pat)

        # Testa match contíguo no final
        if len(nw) >= pat_len:
            tail = tuple(nw[-pat_len:])
            if tail == pat:
                weight = pat_info["weight"]
                if best is None or weight > best["weight"]:
                    best = pat_info

        # Testa match não-contíguo no final (gap ≤ 2)
        if best is None and len(nw) >= pat_len + 2:
            for offset in range(1, 3):
                window_check = nw[-(pat_len + offset):]
                rate = _pattern_hit_rate(window_check, pat, target)
                if rate and rate >= TITAN_MIN_CONF:
                    weight = pat_info["weight"] * 0.85   # penalidade por gap
                    if best is None or weight > best.get("weight", 0):
                        best = {**pat_info, "weight": weight, "gap_match": True}
                    break

    if best is None:
        return {"vote": None, "confidence": 0.0,
                "label": "titan:sem_match", "key": "titan", "source": "titan"}

    conf = min(best["wilson"] * (1.0 + (best["lift"] - 1.0) * 0.3), 0.94)

    # Bônus de regime
    regime_bonus = 1.10 if regime["name"] == "balanced" else 1.0
    conf = min(conf * regime_bonus, 0.94)

    if conf < TITAN_MIN_CONF:
        return {"vote": None, "confidence": 0.0,
                "label": f"titan:conf_baixa({conf:.2f})",
                "key": "titan", "source": "titan"}

    gap_str = " [gap]" if best.get("gap_match") else ""
    return {
        "vote":       best["target"],
        "confidence": round(conf, 4),
        "label":      (f"⚒️ TITAN {''.join(cs(c) for c in best['pattern'])}"
                       f"→{'V' if best['target']==1 else 'P'} "
                       f"lift={best['lift']:.2f} p={best['p_value']:.3f}{gap_str}"),
        "key":        best["id"],
        "source":     "titan",
    }