# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE SYBIL — Fourier Spectral Cycle Detector
#  Aplica FFT na sequência para detectar periodicidades ocultas.
#  Se existe um ciclo de período K dominante, prediz com base nele.
# ═══════════════════════════════════════════════════════════════════════════════
import logging
import numpy as np
from collections import deque

log = logging.getLogger("sybil")

SYBIL_WINDOW     = 256   # potência de 2 para FFT eficiente
SYBIL_MIN_POWER  = 3.5   # amplitude mínima para considerar ciclo real
SYBIL_MIN_CONF   = 0.63  # confiança mínima para votar
SYBIL_CACHE_SIZE = 5     # últimos ciclos detectados (estabilidade)

_sybil_cycle_history = deque(maxlen=SYBIL_CACHE_SIZE)

def _encode_colors_for_fft(colors):
    """
    Transforma cores em sinal numérico bipolar:
      VERMELHO=+1, PRETO=-1, BRANCO=0
    Branco é neutro — não distorce o sinal.
    """
    mapping = {1: 1.0, 2: -1.0, 0: 0.0}
    return np.array([mapping.get(c, 0.0) for c in colors], dtype=np.float64)

def _dominant_cycles(signal, top_n=4):
    """
    Retorna os top_n períodos dominantes (em rounds) e suas amplitudes.
    Ignora DC (freq 0) e frequências acima de Nyquist/2.
    """
    n = len(signal)
    if n < 16:
        return []

    # Janela Hann para reduzir vazamento espectral
    window  = np.hanning(n)
    windowed = signal * window

    fft_vals = np.fft.rfft(windowed)
    freqs    = np.fft.rfftfreq(n)
    power    = np.abs(fft_vals)

    # Ignora DC (idx=0) e frequências muito altas (período < 3 rounds)
    valid_mask = (freqs > 0) & (freqs < 0.33)
    valid_idx  = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return []

    valid_power = power[valid_idx]
    top_idx     = valid_idx[np.argsort(valid_power)[::-1][:top_n]]

    cycles = []
    for idx in top_idx:
        freq   = freqs[idx]
        period = round(1.0 / freq) if freq > 0 else 0
        amp    = float(power[idx])
        phase  = float(np.angle(fft_vals[idx]))
        if period >= 3 and amp >= SYBIL_MIN_POWER:
            cycles.append({"period": period, "amplitude": amp, "phase": phase, "freq": float(freq)})

    return cycles

def _predict_from_cycle(colors, cycle):
    """
    Dado um ciclo dominante de período K e fase φ,
    estima a cor esperada na posição atual.
    """
    period = cycle["period"]
    phase  = cycle["phase"]
    n      = len(colors)

    # Posição atual no ciclo
    pos_in_cycle = n % period

    # Lê os valores históricos nessa mesma posição do ciclo
    votes = {1: 0.0, 2: 0.0}
    samples = 0
    for i in range(len(colors) - 1, -1, -1):
        if i % period == pos_in_cycle:
            c = colors[i]
            if c in (1, 2):
                # Pesos maiores para amostras mais recentes
                weight = 1.0 + (i / len(colors)) * 0.5
                votes[c] += weight
                samples  += 1
        if samples >= 20:
            break

    total = votes[1] + votes[2]
    if total < 5:
        return None, 0.0

    prob_r = votes[1] / total
    prob_b = votes[2] / total

    if abs(prob_r - prob_b) < 0.10:
        return None, 0.0

    winner = 1 if prob_r > prob_b else 2
    conf   = max(prob_r, prob_b)
    return winner, conf

def expert_sybil(colors, regime):
    """
    SYBIL — Fourier Spectral Analyzer.
    Detecta ciclos com FFT e vota quando há periodicidade dominante confiável.
    Mais poderoso em regimes 'balanced' e 'chaotic' (onde outros experts ficam cegos).
    """
    nw = [c for c in colors if c != 0]
    if len(nw) < 64:
        return {"vote": None, "confidence": 0.0,
                "label": "sybil:histórico_insuf", "key": "sybil", "source": "sybil"}

    # Usa os últimos SYBIL_WINDOW rounds (ou tudo se menor)
    window = min(SYBIL_WINDOW, len(nw))
    signal = _encode_colors_for_fft(nw[-window:])

    cycles = _dominant_cycles(signal, top_n=4)

    if not cycles:
        return {"vote": None, "confidence": 0.0,
                "label": "sybil:sem_ciclos", "key": "sybil", "source": "sybil"}

    # Tenta prever pelo ciclo mais forte
    best_vote, best_conf = None, 0.0
    best_cycle = None

    for cyc in cycles:
        vote, conf = _predict_from_cycle(nw, cyc)
        if vote is not None and conf > best_conf:
            best_vote  = vote
            best_conf  = conf
            best_cycle = cyc

    if best_vote is None or best_conf < SYBIL_MIN_CONF:
        return {"vote": None, "confidence": 0.0,
                "label": f"sybil:confiança_baixa({best_conf:.2f})", "key": "sybil", "source": "sybil"}

    # Registra ciclo detectado no histórico (estabilidade)
    _sybil_cycle_history.append(best_cycle["period"])

    # Bônus de estabilidade: se o mesmo período aparece em calls consecutivos
    period_hist = list(_sybil_cycle_history)
    stability_bonus = 1.0
    if len(period_hist) >= 3:
        most_common_count = max(period_hist.count(p) for p in set(period_hist))
        stability_ratio   = most_common_count / len(period_hist)
        if stability_ratio >= 0.6:
            stability_bonus = 1.12
            log.info("🔮 SYBIL: Ciclo estável detectado período=%d (%.0f%% estável)",
                     best_cycle["period"], stability_ratio * 100)

    # Bônus em regime caótico — onde SYBIL é único que enxerga
    regime_bonus = 1.15 if regime["name"] == "chaotic" else 1.0

    final_conf = min(best_conf * stability_bonus * regime_bonus, 0.95)

    return {
        "vote":       best_vote,
        "confidence": round(final_conf, 4),
        "label":      f"🔮 SYBIL ciclo={best_cycle['period']}rounds amp={best_cycle['amplitude']:.1f} → {'V' if best_vote==1 else 'P'}",
        "key":        f"sybil_period_{best_cycle['period']}",
        "source":     "sybil",
    }