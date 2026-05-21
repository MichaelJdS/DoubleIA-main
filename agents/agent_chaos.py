# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE CHAOS — Permutation Entropy Collapse Predictor
#  Usa Entropia de Permutação (Bandt-Pompe 2002) para detectar transições
#  caos→ordem antes que qualquer outro expert perceba.
#  Referência: Bandt & Pompe, PRL 88 (2002) 174102
# ═══════════════════════════════════════════════════════════════════════════════
import logging
import math
from collections import deque
from math import factorial
from itertools import permutations

log = logging.getLogger("chaos")

CHAOS_EMBED_DIM     = 4     # dimensão de embedding (m=4 → 4! = 24 padrões)
CHAOS_LAG           = 1     # lag temporal
CHAOS_WINDOW        = 60    # janela para cálculo de entropia atual
CHAOS_HISTORY_WIN   = 200   # janela para referência histórica
CHAOS_COLLAPSE_THR  = 0.18  # queda de entropia que sinaliza colapso iminente
CHAOS_MIN_HISTORY   = 5     # mínimo de medições para detectar tendência

_chaos_entropy_history = deque(maxlen=30)

def _permutation_entropy(series, m=4, lag=1):
    """
    Calcula a Entropia de Permutação (PE) de uma série temporal.
    PE próxima de 1.0 = máximo caos (aleatório)
    PE próxima de 0.0 = máxima ordem (determinístico)
    
    Bandt & Pompe (2002) — método robusto para sinais não-estacionários.
    """
    n = len(series)
    if n < m * lag + 1:
        return 1.0  # insuficiente → assume máximo caos

    # Gera todos os padrões de ordem possíveis
    all_perms = list(permutations(range(m)))
    perm_counts = {p: 0 for p in all_perms}

    # Conta frequência de cada padrão de ordem
    for i in range(n - (m - 1) * lag):
        # Extrai sub-série com lag
        sub = [series[i + j * lag] for j in range(m)]
        # Padrão ordinal (ranking dos valores)
        pattern = tuple(sorted(range(m), key=lambda x: sub[x]))
        if pattern in perm_counts:
            perm_counts[pattern] += 1

    total = sum(perm_counts.values())
    if total == 0:
        return 1.0

    # Entropia de Shannon normalizada
    max_entropy = math.log2(factorial(m))
    entropy = 0.0
    for count in perm_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy / max_entropy if max_entropy > 0 else 1.0

def _entropy_trend(history):
    """
    Calcula a tendência da entropia (regressão linear simples).
    Retorna o coeficiente angular (negativo = entropia caindo = ordem emergindo).
    """
    n = len(history)
    if n < 3:
        return 0.0

    x     = list(range(n))
    x_avg = sum(x) / n
    y_avg = sum(history) / n

    num = sum((x[i] - x_avg) * (history[i] - y_avg) for i in range(n))
    den = sum((x[i] - x_avg) ** 2 for i in range(n))

    return num / den if den != 0 else 0.0

def expert_chaos(colors, regime):
    """
    CHAOS — Permutation Entropy Collapse Predictor.
    
    Lógica:
    1. Calcula PE atual na janela recente
    2. Compara com PE histórica (referência)
    3. Se PE está CAINDO rapidamente → ordem emergindo → moment de entrada
    4. Usa a direção da ordem emergente para votar
    """
    nw = [c for c in colors if c != 0]
    if len(nw) < CHAOS_HISTORY_WIN:
        return {"vote": None, "confidence": 0.0,
                "label": "chaos:histórico_insuf", "key": "chaos", "source": "chaos"}

    # Codifica como série numérica (vermelho=1, preto=2)
    series_full   = nw[-CHAOS_HISTORY_WIN:]
    series_recent = nw[-CHAOS_WINDOW:]

    pe_recent = _permutation_entropy(series_recent, m=CHAOS_EMBED_DIM, lag=CHAOS_LAG)
    pe_hist   = _permutation_entropy(series_full,   m=CHAOS_EMBED_DIM, lag=CHAOS_LAG)

    # Registra histórico de PE
    _chaos_entropy_history.append(pe_recent)
    hist_list = list(_chaos_entropy_history)

    # Tendência de queda de entropia
    trend = _entropy_trend(hist_list)

    # Delta entre PE histórica e recente
    pe_delta = pe_hist - pe_recent  # positivo = PE caiu = mais ordem recente

    log.debug("🌀 CHAOS: PE_recente=%.3f PE_hist=%.3f delta=%.3f trend=%.4f",
              pe_recent, pe_hist, pe_delta, trend)

    # MODO 1: Colapso iminente (PE caindo rápido)
    if pe_delta >= CHAOS_COLLAPSE_THR and trend < -0.005:
        # Ordem emergindo: descobre qual cor está se tornando dominante
        recent_50 = nw[-50:]
        r_count = recent_50.count(1)
        b_count = recent_50.count(2)
        total   = r_count + b_count

        if total == 0:
            return {"vote": None, "confidence": 0.0,
                    "label": "chaos:sem_dados", "key": "chaos", "source": "chaos"}

        r_rate = r_count / total
        b_rate = b_count / total

        # Vota na cor que está *dominando* a ordem emergente
        if abs(r_rate - b_rate) < 0.08:
            return {"vote": None, "confidence": 0.0,
                    "label": f"chaos:colapso_sem_direção(δ={pe_delta:.2f})",
                    "key": "chaos", "source": "chaos"}

        winner = 1 if r_rate > b_rate else 2
        conf   = min(0.55 + pe_delta * 1.8 + abs(trend) * 20, 0.93)

        return {
            "vote":       winner,
            "confidence": round(conf, 4),
            "label":      f"🌀 CHAOS colapso PE={pe_recent:.3f}↓{pe_delta:.2f} → {'V' if winner==1 else 'P'}",
            "key":        "chaos_collapse",
            "source":     "chaos",
        }

    # MODO 2: Alta ordem detectada (PE já baixa) → sinal de continuação
    if pe_recent <= 0.72 and pe_hist <= 0.78:
        nw_last = nw[-1]
        # Em alta ordem, a última cor tende a continuar (streak ou alternância)
        alt = sum(1 for i in range(1, min(len(nw), 10)) if nw[-i] != nw[-i-1]) / min(len(nw)-1, 9)

        if alt >= 0.70:
            vote = 2 if nw_last == 1 else 1  # ordem alternante
        else:
            vote = nw_last  # ordem de streak

        conf = min(0.50 + (0.80 - pe_recent) * 1.5, 0.88)

        return {
            "vote":       vote,
            "confidence": round(conf, 4),
            "label":      f"🌀 CHAOS ordem_alta PE={pe_recent:.3f} → {'V' if vote==1 else 'P'}",
            "key":        "chaos_order",
            "source":     "chaos",
        }

    # MODO 3: Caos puro → VETO (protege a banca)
    if pe_recent >= 0.94 and regime["name"] == "chaotic":
        return {
            "vote":       None,
            "confidence": 0.0,
            "label":      f"🌀 CHAOS máximo(PE={pe_recent:.3f}) — VETO",
            "key":        "chaos_veto",
            "source":     "chaos",
            "veto":       True,
        }

    return {"vote": None, "confidence": 0.0,
            "label": f"chaos:neutro(PE={pe_recent:.3f})", "key": "chaos", "source": "chaos"}