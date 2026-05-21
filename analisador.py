"""=============================================================================
BLAZE DOUBLE AI — LEVIATHAN ENGINE v2.0 (MIXTURE OF EXPERTS + CONTINUAL LEARNING)
Arquitetura MoE com Especialistas por Cor, Gating Network, Concept Drift Detection
e Aprendizado Contínuo relendo o banco de dados completo.
=============================================================================

ARQUITETURA:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  LEVIATHAN ENGINE v2.0 — MoE                       │
  │                                                                     │
  │  ┌──────────────┐   ┌─────────────────────────────────────────┐    │
  │  │   GATING     │   │         CONTINUAL LEARNER               │    │
  │  │   NETWORK    │   │  • Relê DB completo (cada 50 rounds)    │    │
  │  │  (Router)    │   │  • Concept Drift Detection              │    │
  │  └──────────────┘   │  • Janelas: 500 / 2000 / full DB       │    │
  │         ↓           └─────────────────────────────────────────┘    │
  │  ┌───────────────────────────────────────┐                         │
  │  │         MIXTURE OF EXPERTS            │                         │
  │  │  ┌──────────┐ ┌──────────┐ ┌───────┐ │                         │
  │  │  │🔴 RED    │ │⚫ BLACK  │ │⚪WHITE│ │                         │
  │  │  │ Expert   │ │ Expert   │ │ Expert│ │                         │
  │  │  └──────────┘ └──────────┘ └───────┘ │                         │
  │  └───────────────────────────────────────┘                         │
  │         ↓                                                           │
  │  ┌─────────────────────────────────────────────────────────────┐   │
  │  │   META-LEARNER NEURAL (pesos por Expert × Cor × Regime)    │   │
  │  └─────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────┘
"""

import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import requests

try:
    from notificador import notificar_sinal
    TELEGRAM_OK = True
except Exception:
    TELEGRAM_OK = False
    def notificar_sinal(*a, **k):
        return False


# ─────────────────────────── CONFIGURAÇÕES ────────────────────────────────────
DB_PATH = "blaze_double.db"
LOG_FILE = "leviathan.log"
LOOP_INTERVAL = 2
MIN_HISTORY = 80

THRESHOLD_MIN = 0.58
THRESHOLD_START = 0.68
THRESHOLD_MAX = 0.85
THRESHOLD_STEP_UP = 0.02
THRESHOLD_STEP_DOWN = 0.01

AUTO_MUTE_LOSSES = 4
AUTO_MUTE_ROUNDS = 20
MUTE_SCORE_NEEDED = 3

MIN_VOTES_TO_ENTER = 2
ENSEMBLE_MODULES = 5

MINER_INTERVAL = 25
MINER_MIN_MATCHES = 6
MINER_MIN_BAYES = 0.63

# Continual Learning
CL_RELEARN_EVERY = 50       # rounds novos para re-aprender
CL_DRIFT_WINDOW = 100       # janela para drift detection
CL_DRIFT_THRESHOLD = 0.12   # diferença de distribuição que dispara re-learn

DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

# ─────────────────────────── ESTADO GLOBAL ───────────────────────────────────
_miner_lock = threading.Lock()
_catalog_lock = threading.Lock()
_cl_lock = threading.Lock()
_neural_weights_lock = threading.Lock()

GLOBAL_MINED_STRATS = []
GLOBAL_CATALOG_STRATS = []

# Estado do Continual Learner
CL_STATE = {
    "last_db_count": 0,
    "color_dist_full": {0: 0.07, 1: 0.465, 2: 0.465},
    "color_dist_500": {0: 0.07, 1: 0.465, 2: 0.465},
    "color_dist_2000": {0: 0.07, 1: 0.465, 2: 0.465},
    "drift_detected": False,
    "drift_magnitude": 0.0,
    "last_relearn_ts": None,
    "expert_stats": {
        "red":   {"total": 0, "wins": 0, "by_regime": defaultdict(lambda: {"total": 0, "wins": 0})},
        "black": {"total": 0, "wins": 0, "by_regime": defaultdict(lambda: {"total": 0, "wins": 0})},
        "white": {"total": 0, "wins": 0, "by_regime": defaultdict(lambda: {"total": 0, "wins": 0})},
    },
    "transition_matrix": {},   # P(cor_atual | cor_anterior)
    "ngram_cache": {},         # Cache de n-gramas do DB completo
}

_threshold_state = {
    "value": THRESHOLD_START,
    "consecutive_losses": 0,
    "consecutive_wins": 0,
    "mute_until_round": 0,
    "mute_win_counter": 0,
    "total_rounds_seen": 0,
    "history": deque(maxlen=200),
}
_threshold_lock = threading.Lock()

_pattern_perf_memory: dict = defaultdict(lambda: deque(maxlen=30))
_perf_lock = threading.Lock()

_NEURAL_LEARNING_RATE = 0.05
_DEFAULT_NEURAL_WEIGHTS = {
    # Pesos por (expert_cor, sub_modulo)
    "red_miner": 1.0, "red_catalog": 1.2, "red_markov": 0.9,
    "red_streak": 0.85, "red_cl": 1.1,
    "black_miner": 1.0, "black_catalog": 1.2, "black_markov": 0.9,
    "black_streak": 0.85, "black_cl": 1.1,
    "white_poisson": 1.3, "white_gap": 1.1, "white_post": 1.0, "white_cl": 1.2,
}

# ─────────────────────────── LOGGING ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("leviathan")


# ─────────────────────────── DB HELPERS ──────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH, timeout=20)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def _add_column_safe(conn, table, column, definition):
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()
    except Exception:
        pass


def get_sys_config(key, default=None):
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except sqlite3.OperationalError:
        return default


def set_sys_config(key, value):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def get_mode():
    return get_sys_config("mode", "moe")


def get_max_gales():
    try:
        return int(get_sys_config("max_gales", "0"))
    except Exception:
        return 0


def get_groq_models():
    custom = os.environ.get("GROQ_MODEL", "").strip()
    if custom:
        return [x.strip() for x in custom.split(",") if x.strip()]
    return DEFAULT_GROQ_MODELS[:]


# ─────────────────────────── NEURAL WEIGHTS ──────────────────────────────────
def get_neural_weights():
    weights_json = get_sys_config("neural_weights_v2", "")
    if weights_json:
        try:
            return json.loads(weights_json)
        except Exception:
            pass
    return _DEFAULT_NEURAL_WEIGHTS.copy()


def update_neural_weights(expert_key: str, won: bool, confidence: float):
    with _neural_weights_lock:
        weights = get_neural_weights()
        error = 1.0 if won else -1.0
        adjustment = _NEURAL_LEARNING_RATE * error * confidence
        weights[expert_key] = max(0.1, min(3.0, weights.get(expert_key, 1.0) + adjustment))
        set_sys_config("neural_weights_v2", json.dumps(weights))
    log.debug("🧠 Neural [%s]: %.3f", expert_key, weights[expert_key])


# ─────────────────────────── INIT TABLES ─────────────────────────────────────
def init_tables():
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS analysis_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT NOT NULL,
        total_rounds    INTEGER,
        last_round_id   TEXT,
        prob_red        REAL,
        prob_black      REAL,
        prob_white      REAL,
        signal_color    INTEGER,
        signal_conf     REAL,
        signal_action   TEXT,
        signal_reason   TEXT,
        regime          TEXT,
        regime_strength REAL,
        white_hazard    REAL,
        dist_last_white INTEGER,
        features_json   TEXT,
        patterns_json   TEXT,
        mode_used       TEXT DEFAULT 'moe_v2',
        votes_json      TEXT DEFAULT '[]',
        threshold_used  REAL DEFAULT 0.68
    );

    CREATE TABLE IF NOT EXISTS prediction_performance (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id INTEGER,
        ts          TEXT NOT NULL,
        predicted   INTEGER,
        confidence  REAL,
        actual      INTEGER,
        correct     INTEGER,
        action      TEXT,
        mode        TEXT DEFAULT 'moe_v2',
        pattern_key TEXT DEFAULT '',
        expert_used TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS leviathan_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS cl_snapshots (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT NOT NULL,
        db_count    INTEGER,
        dist_full   TEXT,
        dist_500    TEXT,
        dist_2000   TEXT,
        drift_mag   REAL,
        expert_stats TEXT
    );
    """)
    conn.commit()

    _add_column_safe(conn, "analysis_snapshots", "mode_used", "TEXT DEFAULT 'moe_v2'")
    _add_column_safe(conn, "analysis_snapshots", "votes_json", "TEXT DEFAULT '[]'")
    _add_column_safe(conn, "analysis_snapshots", "threshold_used", "REAL DEFAULT 0.68")
    _add_column_safe(conn, "prediction_performance", "expert_used", "TEXT DEFAULT ''")
    conn.close()
    log.info("Tabelas MoE v2 verificadas.")


def load_meta(key, default="0"):
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT value FROM leviathan_meta WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception:
        return default


def save_meta(key, value):
    try:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO leviathan_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────── HELPERS ─────────────────────────────────────────
def color_short(c):
    return {0: "B", 1: "V", 2: "P"}.get(c, "?")


def color_name(c):
    return {0: "BRANCO", 1: "VERMELHO", 2: "PRETO"}.get(c, "—")


def bayes_prob(wins, matches, alpha=1.5):
    return (wins + alpha) / (matches + 2 * alpha)


def kelly_fraction(prob, frac=0.25):
    if prob <= 0.50:
        return 0.0
    edge = 2.0 * prob - 1.0
    return max(0.0, min(edge * frac, 0.30))


def poisson_white_hazard(colors):
    gaps = []
    cur = 0
    for c in colors:
        if c == 0:
            gaps.append(cur)
            cur = 0
        else:
            cur += 1
    avg = sum(gaps) / len(gaps) if gaps else 14.0
    lam = 1.0 / max(avg, 1e-9)
    hazard = 1.0 - math.exp(-lam * cur)
    return {
        "dist": cur,
        "hazard": round(min(hazard, 0.99), 4),
        "avg_gap": round(avg, 2),
        "post_white": cur == 0,
    }


def markov_prob(colors, target, order=2):
    nw = [c for c in colors if c != 0]
    if len(nw) < order + 1:
        return 0.465
    key = tuple(nw[-order:])
    counts = defaultdict(int)
    total = 0
    for i in range(len(nw) - order):
        k = tuple(nw[i:i + order])
        if k == key:
            counts[nw[i + order]] += 1
            total += 1
    if total == 0:
        return 0.465
    return (counts[target] + 1) / (total + 2)


def entropy_regime(colors, window=20):
    recent = [c for c in colors[-window:] if c != 0]
    if len(recent) < 4:
        return 1.0
    counts = defaultdict(int)
    for c in recent:
        counts[c] += 1
    n = len(recent)
    ent = -sum((v / n) * math.log2(v / n + 1e-10) for v in counts.values())
    return round(min(ent / 1.0, 1.0), 4)


def streak_info(colors):
    nw = [c for c in colors if c != 0]
    if not nw:
        return {"color": None, "length": 0}
    anchor = nw[-1]
    length = 0
    for c in reversed(nw):
        if c == anchor:
            length += 1
        else:
            break
    return {"color": anchor, "length": length}


def alternation_ratio(colors, window=10):
    nw = [c for c in colors[-50:] if c != 0][-window:]
    if len(nw) < 2:
        return 0.5
    flips = sum(1 for i in range(1, len(nw)) if nw[i] != nw[i - 1])
    return flips / (len(nw) - 1)


def load_sequence(limit=4000):
    conn = _conn()
    try:
        c = conn.cursor()
        c.execute(
            "SELECT id, round_id, color, roll, created_at FROM results_raw ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()
        return [
            {"id": r[0], "round_id": r[1], "color": r[2], "roll": r[3], "created_at": r[4]}
            for r in reversed(rows)
        ]
    finally:
        conn.close()


def get_last_round_id():
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT round_id FROM results_raw ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_last_analyzed_round_id():
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT last_round_id FROM analysis_snapshots ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# ─────────────────────────── DETECTOR DE REGIME ──────────────────────────────
def detect_regime(colors):
    ent = entropy_regime(colors, 20)
    strk = streak_info(colors)
    alt = alternation_ratio(colors, 12)
    wh = poisson_white_hazard(colors)
    nw = [c for c in colors if c != 0]

    if wh["dist"] >= 18 and wh["hazard"] >= 0.72:
        return {"name": "white_zone", "label": "⚪ Zona Branca", "strength": wh["hazard"], "data": wh}
    if strk["length"] >= 4:
        return {"name": "streak_hot", "label": f"🔥 Streak {strk['length']}x",
                "strength": min(strk["length"] / 8.0, 0.99), "data": strk}
    if alt >= 0.78 and len(nw) >= 6:
        return {"name": "alternating", "label": "↔️ Alternância",
                "strength": round(alt, 3), "data": {"alt_ratio": alt}}
    if ent >= 0.88:
        return {"name": "chaotic", "label": "🌀 Caótico", "strength": round(ent, 3), "data": {"entropy": ent}}
    return {"name": "balanced", "label": "⚖️ Equilibrado", "strength": 0.5, "data": {}}


# ─────────────────────────── CONTINUAL LEARNER ────────────────────────────────
def _compute_color_dist(colors):
    n = len(colors)
    if n == 0:
        return {0: 0.07, 1: 0.465, 2: 0.465}
    counts = defaultdict(int)
    for c in colors:
        counts[c] += 1
    return {k: round(counts[k] / n, 4) for k in (0, 1, 2)}


def _compute_transition_matrix(colors):
    """Calcula P(next | current) para todas as cores incluindo branco."""
    matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(colors) - 1):
        matrix[colors[i]][colors[i + 1]] += 1
    result = {}
    for from_c, to_counts in matrix.items():
        total = sum(to_counts.values())
        result[from_c] = {to_c: round(cnt / total, 4) for to_c, cnt in to_counts.items()}
    return result


def _compute_ngram_cache(colors, max_len=6):
    """Computa n-gramas do DB completo para uso pelos experts."""
    cache = defaultdict(lambda: {1: 0, 2: 0, 0: 0, "total": 0})
    n = len(colors)
    for length in range(2, max_len + 1):
        for i in range(length - 1, n - 1):
            seq = tuple(colors[i - length + 1: i + 1])
            next_c = colors[i + 1]
            cache[seq][next_c] += 1
            cache[seq]["total"] += 1
    return dict(cache)


def _detect_concept_drift(old_dist, new_dist):
    """KL Divergence simplificada entre distribuições."""
    magnitude = sum(
        abs(new_dist.get(k, 0) - old_dist.get(k, 0)) for k in (0, 1, 2)
    )
    return magnitude


def continual_learner_thread():
    """Thread que relê o DB completo periodicamente e re-adapta todos os experts."""
    global CL_STATE
    log.info("🧬 Continual Learner iniciado.")

    while True:
        try:
            conn = _conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM results_raw")
            total_count = c.fetchone()[0]

            with _cl_lock:
                last_count = CL_STATE["last_db_count"]

            # Só re-aprende se tiver N novos rounds OU no primeiro boot
            if total_count - last_count >= CL_RELEARN_EVERY or last_count == 0:
                log.info("🧬 Continual Learner: relendo %d rounds do DB...", total_count)

                c.execute("SELECT color FROM results_raw ORDER BY id ASC")
                all_colors = [r[0] for r in c.fetchall()]

                if len(all_colors) < 100:
                    conn.close()
                    time.sleep(15)
                    continue

                # Distribuições por janela
                dist_full = _compute_color_dist(all_colors)
                dist_2000 = _compute_color_dist(all_colors[-2000:])
                dist_500 = _compute_color_dist(all_colors[-500:])

                # Matriz de transição
                transition = _compute_transition_matrix(all_colors)

                # N-grama cache (DB completo)
                ngram = _compute_ngram_cache(all_colors, max_len=6)

                # Concept Drift Detection
                with _cl_lock:
                    old_dist = CL_STATE["color_dist_500"].copy()

                drift_mag = _detect_concept_drift(old_dist, dist_500)
                drift_detected = drift_mag >= CL_DRIFT_THRESHOLD

                if drift_detected:
                    log.warning(
                        "⚠️ CONCEPT DRIFT DETECTADO! Magnitude=%.4f — Re-calibrando experts...",
                        drift_mag
                    )

                # Estatísticas dos experts por regime (lendo prediction_performance)
                c.execute("""
                    SELECT pp.predicted, pp.actual, pp.correct, pp.expert_used,
                           a.regime
                    FROM prediction_performance pp
                    LEFT JOIN analysis_snapshots a ON a.id = pp.snapshot_id
                    WHERE pp.action IN ('win', 'loss')
                    ORDER BY pp.id DESC
                    LIMIT 2000
                """)
                perf_rows = c.fetchall()

                expert_stats = {
                    "red":   {"total": 0, "wins": 0, "by_regime": defaultdict(lambda: {"total": 0, "wins": 0})},
                    "black": {"total": 0, "wins": 0, "by_regime": defaultdict(lambda: {"total": 0, "wins": 0})},
                    "white": {"total": 0, "wins": 0, "by_regime": defaultdict(lambda: {"total": 0, "wins": 0})},
                }

                for predicted, actual, correct, expert_used, regime_label in perf_rows:
                    exp_key = {1: "red", 2: "black", 0: "white"}.get(predicted, "red")
                    regime_key = regime_label or "balanced"
                    expert_stats[exp_key]["total"] += 1
                    expert_stats[exp_key]["by_regime"][regime_key]["total"] += 1
                    if correct:
                        expert_stats[exp_key]["wins"] += 1
                        expert_stats[exp_key]["by_regime"][regime_key]["wins"] += 1

                # Salvar snapshot CL no DB
                conn.execute(
                    """INSERT INTO cl_snapshots
                       (ts, db_count, dist_full, dist_500, dist_2000, drift_mag, expert_stats)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        total_count,
                        json.dumps(dist_full),
                        json.dumps(dist_500),
                        json.dumps(dist_2000),
                        drift_mag,
                        json.dumps({
                            k: {"total": v["total"], "wins": v["wins"]}
                            for k, v in expert_stats.items()
                        }),
                    )
                )
                conn.commit()

                # Atualizar estado global
                with _cl_lock:
                    CL_STATE["last_db_count"] = total_count
                    CL_STATE["color_dist_full"] = dist_full
                    CL_STATE["color_dist_500"] = dist_500
                    CL_STATE["color_dist_2000"] = dist_2000
                    CL_STATE["drift_detected"] = drift_detected
                    CL_STATE["drift_magnitude"] = drift_mag
                    CL_STATE["last_relearn_ts"] = datetime.now().strftime("%H:%M:%S")
                    CL_STATE["transition_matrix"] = transition
                    CL_STATE["ngram_cache"] = ngram
                    CL_STATE["expert_stats"] = expert_stats

                log.info(
                    "🧬 CL Completo | Dist500=%s | Drift=%.4f %s | Transições=%d | NGrams=%d",
                    dist_500, drift_mag,
                    "⚠️ DRIFT!" if drift_detected else "✅ Estável",
                    len(transition),
                    len(ngram),
                )

            conn.close()

        except Exception as e:
            log.error("Erro no Continual Learner: %s", e, exc_info=True)

        time.sleep(10)


# ─────────────────────────── GATING NETWORK (ROUTER MoE) ─────────────────────
def gating_network(regime: dict, colors: list) -> dict:
    """
    Decide quais experts ativar e com qual peso base,
    baseado no regime atual e no estado do Continual Learner.
    Retorna: {"red": float, "black": float, "white": float}
    """
    name = regime["name"]
    strength = regime["strength"]
    wh = poisson_white_hazard(colors)

    with _cl_lock:
        dist_500 = CL_STATE["color_dist_500"].copy()
        drift = CL_STATE["drift_detected"]
        trans = CL_STATE["transition_matrix"].copy()
        expert_stats = CL_STATE["expert_stats"]

    # Pesos base por regime
    weights = {
        "white_zone":  {"red": 0.2,  "black": 0.2,  "white": 1.0},
        "streak_hot":  {"red": 0.9,  "black": 0.9,  "white": 0.3},
        "alternating": {"red": 0.85, "black": 0.85, "white": 0.2},
        "chaotic":     {"red": 0.6,  "black": 0.6,  "white": 0.4},
        "balanced":    {"red": 0.8,  "black": 0.8,  "white": 0.5},
    }.get(name, {"red": 0.7, "black": 0.7, "white": 0.4})

    # Ajuste por distribuição recente (Continual Learner)
    # Se vermelho está aparecendo muito, o expert vermelho fica menos confiante (mean reversion)
    red_bias = dist_500.get(1, 0.465)
    black_bias = dist_500.get(2, 0.465)

    if red_bias > 0.52:
        weights["red"] *= 0.85
        weights["black"] *= 1.10
    elif red_bias < 0.42:
        weights["red"] *= 1.10
        weights["black"] *= 0.90

    # Ajuste por performance histórica dos experts (vindo do CL)
    for color_key, exp_data in expert_stats.items():
        total = exp_data.get("total", 0)
        wins = exp_data.get("wins", 0)
        if total >= 20:
            acc = wins / total
            bonus = (acc - 0.50) * 0.4  # ±20% de ajuste
            weights[color_key] = max(0.1, min(1.5, weights[color_key] + bonus))

    # Se drift detectado, reduz confiança geral e aumenta white (incerteza)
    if drift:
        weights["red"] *= 0.80
        weights["black"] *= 0.80
        weights["white"] *= 1.20

    # Normalizar para [0, 1]
    max_w = max(weights.values()) or 1.0
    return {k: round(v / max_w, 3) for k, v in weights.items()}


# ─────────────────────────── EXPERT: RED 🔴 ──────────────────────────────────
def red_expert(colors: list, regime: dict) -> dict:
    """Especialista dedicado a prever VERMELHO."""
    nw = [c for c in colors if c != 0]
    if len(nw) < 10:
        return {"confidence": 0.0, "signals": [], "color": 1}

    weights = get_neural_weights()
    signals = []
    score = 0.0
    weight_sum = 0.0

    # Sub-módulo 1: Markov
    p_red_m1 = markov_prob(nw, 1, order=1)
    p_red_m2 = markov_prob(nw, 1, order=2)
    p_red_markov = p_red_m1 * 0.35 + p_red_m2 * 0.65
    w_markov = weights.get("red_markov", 0.9)
    score += p_red_markov * w_markov
    weight_sum += w_markov
    signals.append({"name": "red_markov", "conf": round(p_red_markov, 3)})

    # Sub-módulo 2: Streak (reversão ou continuação para vermelho)
    strk = streak_info(nw)
    w_streak = weights.get("red_streak", 0.85)
    if strk["color"] == 1 and strk["length"] >= 3:
        # Streak vermelho — pode continuar
        cont_prob = bayes_prob(
            sum(1 for i in range(len(nw) - 1) if nw[i] == 1 and nw[i + 1] == 1),
            sum(1 for c in nw if c == 1), alpha=1.0
        )
        score += cont_prob * w_streak
        weight_sum += w_streak
        signals.append({"name": "red_streak_cont", "conf": round(cont_prob, 3)})
    elif strk["color"] == 2 and strk["length"] >= 3:
        # Streak preto longo — reversão para vermelho
        rev_total = sum(1 for i in range(len(nw) - strk["length"])
                        if all(nw[i + j] == 2 for j in range(strk["length"])) and
                        i + strk["length"] < len(nw))
        rev_red = sum(1 for i in range(len(nw) - strk["length"])
                      if all(nw[i + j] == 2 for j in range(strk["length"])) and
                      i + strk["length"] < len(nw) and nw[i + strk["length"]] == 1)
        rev_prob = bayes_prob(rev_red, rev_total) if rev_total > 3 else 0.5
        score += rev_prob * w_streak
        weight_sum += w_streak
        signals.append({"name": "red_streak_rev", "conf": round(rev_prob, 3)})

    # Sub-módulo 3: Miner (n-gramas locais)
    w_miner = weights.get("red_miner", 1.0)
    with _miner_lock:
        best_miner = None
        for s in GLOBAL_MINED_STRATS:
            if s["target"] != 1:
                continue
            seq = s["seq"]
            if len(colors) >= len(seq) and tuple(colors[-len(seq):]) == seq:
                if best_miner is None or s["prob"] > best_miner["prob"]:
                    best_miner = s
    if best_miner:
        score += best_miner["prob"] * w_miner
        weight_sum += w_miner
        signals.append({"name": "red_miner", "conf": round(best_miner["prob"], 3)})

    # Sub-módulo 4: Continual Learner (n-grama DB completo)
    w_cl = weights.get("red_cl", 1.1)
    with _cl_lock:
        ngram_cache = CL_STATE["ngram_cache"]
        trans = CL_STATE["transition_matrix"]

    # Usar matriz de transição do CL
    last_color = colors[-1] if colors else None
    if last_color is not None and last_color in trans:
        cl_prob_red = trans[last_color].get(1, 0.465)
        score += cl_prob_red * w_cl
        weight_sum += w_cl
        signals.append({"name": "red_cl_transition", "conf": round(cl_prob_red, 3)})

    # N-grama do CL para vermelho
    for length in range(3, 7):
        if len(colors) >= length:
            key = tuple(colors[-length:])
            if key in ngram_cache:
                entry = ngram_cache[key]
                total_ng = entry.get("total", 0)
                red_ng = entry.get(1, 0)
                if total_ng >= 8:
                    ng_prob = bayes_prob(red_ng, total_ng)
                    if ng_prob >= 0.60:
                        score += ng_prob * w_cl * 0.8
                        weight_sum += w_cl * 0.8
                        signals.append({"name": f"red_cl_ngram_{length}", "conf": round(ng_prob, 3)})
                        break

    # Catalog
    w_catalog = weights.get("red_catalog", 1.2)
    with _catalog_lock:
        best_cat = None
        for s in GLOBAL_CATALOG_STRATS:
            if s.get("target") != 1:
                continue
            if catalog_strategy_matches(s, colors):
                if best_cat is None or s.get("wf_acc", 0) > best_cat.get("wf_acc", 0):
                    best_cat = s
    if best_cat:
        cat_conf = best_cat.get("wf_acc", 0.6)
        score += cat_conf * w_catalog
        weight_sum += w_catalog
        signals.append({"name": "red_catalog", "conf": round(cat_conf, 3)})

    if weight_sum == 0:
        return {"confidence": 0.0, "signals": signals, "color": 1}

    final_conf = min(score / weight_sum, 0.99)
    return {"confidence": round(final_conf, 4), "signals": signals, "color": 1}


# ─────────────────────────── EXPERT: BLACK ⚫ ─────────────────────────────────
def black_expert(colors: list, regime: dict) -> dict:
    """Especialista dedicado a prever PRETO."""
    nw = [c for c in colors if c != 0]
    if len(nw) < 10:
        return {"confidence": 0.0, "signals": [], "color": 2}

    weights = get_neural_weights()
    signals = []
    score = 0.0
    weight_sum = 0.0

    # Markov
    p_black_m1 = markov_prob(nw, 2, order=1)
    p_black_m2 = markov_prob(nw, 2, order=2)
    p_black_markov = p_black_m1 * 0.35 + p_black_m2 * 0.65
    w_markov = weights.get("black_markov", 0.9)
    score += p_black_markov * w_markov
    weight_sum += w_markov
    signals.append({"name": "black_markov", "conf": round(p_black_markov, 3)})

    # Streak
    strk = streak_info(nw)
    w_streak = weights.get("black_streak", 0.85)
    if strk["color"] == 2 and strk["length"] >= 3:
        cont_prob = bayes_prob(
            sum(1 for i in range(len(nw) - 1) if nw[i] == 2 and nw[i + 1] == 2),
            sum(1 for c in nw if c == 2), alpha=1.0
        )
        score += cont_prob * w_streak
        weight_sum += w_streak
        signals.append({"name": "black_streak_cont", "conf": round(cont_prob, 3)})
    elif strk["color"] == 1 and strk["length"] >= 3:
        rev_total = sum(1 for i in range(len(nw) - strk["length"])
                        if all(nw[i + j] == 1 for j in range(strk["length"])) and
                        i + strk["length"] < len(nw))
        rev_black = sum(1 for i in range(len(nw) - strk["length"])
                        if all(nw[i + j] == 1 for j in range(strk["length"])) and
                        i + strk["length"] < len(nw) and nw[i + strk["length"]] == 2)
        rev_prob = bayes_prob(rev_black, rev_total) if rev_total > 3 else 0.5
        score += rev_prob * w_streak
        weight_sum += w_streak
        signals.append({"name": "black_streak_rev", "conf": round(rev_prob, 3)})

    # Miner
    w_miner = weights.get("black_miner", 1.0)
    with _miner_lock:
        best_miner = None
        for s in GLOBAL_MINED_STRATS:
            if s["target"] != 2:
                continue
            seq = s["seq"]
            if len(colors) >= len(seq) and tuple(colors[-len(seq):]) == seq:
                if best_miner is None or s["prob"] > best_miner["prob"]:
                    best_miner = s
    if best_miner:
        score += best_miner["prob"] * w_miner
        weight_sum += w_miner
        signals.append({"name": "black_miner", "conf": round(best_miner["prob"], 3)})

    # Continual Learner
    w_cl = weights.get("black_cl", 1.1)
    with _cl_lock:
        ngram_cache = CL_STATE["ngram_cache"]
        trans = CL_STATE["transition_matrix"]

    last_color = colors[-1] if colors else None
    if last_color is not None and last_color in trans:
        cl_prob_black = trans[last_color].get(2, 0.465)
        score += cl_prob_black * w_cl
        weight_sum += w_cl
        signals.append({"name": "black_cl_transition", "conf": round(cl_prob_black, 3)})

    for length in range(3, 7):
        if len(colors) >= length:
            key = tuple(colors[-length:])
            if key in ngram_cache:
                entry = ngram_cache[key]
                total_ng = entry.get("total", 0)
                black_ng = entry.get(2, 0)
                if total_ng >= 8:
                    ng_prob = bayes_prob(black_ng, total_ng)
                    if ng_prob >= 0.60:
                        score += ng_prob * w_cl * 0.8
                        weight_sum += w_cl * 0.8
                        signals.append({"name": f"black_cl_ngram_{length}", "conf": round(ng_prob, 3)})
                        break

    # Catalog
    w_catalog = weights.get("black_catalog", 1.2)
    with _catalog_lock:
        best_cat = None
        for s in GLOBAL_CATALOG_STRATS:
            if s.get("target") != 2:
                continue
            if catalog_strategy_matches(s, colors):
                if best_cat is None or s.get("wf_acc", 0) > best_cat.get("wf_acc", 0):
                    best_cat = s
    if best_cat:
        cat_conf = best_cat.get("wf_acc", 0.6)
        score += cat_conf * w_catalog
        weight_sum += w_catalog
        signals.append({"name": "black_catalog", "conf": round(cat_conf, 3)})

    if weight_sum == 0:
        return {"confidence": 0.0, "signals": signals, "color": 2}

    final_conf = min(score / weight_sum, 0.99)
    return {"confidence": round(final_conf, 4), "signals": signals, "color": 2}


# ─────────────────────────── EXPERT: WHITE ⚪ ─────────────────────────────────
def white_expert(colors: list, regime: dict) -> dict:
    """Especialista dedicado a prever BRANCO."""
    weights = get_neural_weights()
    signals = []
    score = 0.0
    weight_sum = 0.0

    # Sub-módulo 1: Poisson Hazard
    wh = poisson_white_hazard(colors)
    w_poisson = weights.get("white_poisson", 1.3)
    score += wh["hazard"] * w_poisson
    weight_sum += w_poisson
    signals.append({"name": "white_poisson", "conf": round(wh["hazard"], 3)})

    # Sub-módulo 2: Gap timing (baseado no gap médio histórico)
    w_gap = weights.get("white_gap", 1.1)
    avg_gap = wh["avg_gap"]
    dist = wh["dist"]
    if avg_gap > 0:
        gap_timing_score = min(dist / avg_gap, 1.5) / 1.5
        score += gap_timing_score * w_gap
        weight_sum += w_gap
        signals.append({"name": "white_gap_timing", "conf": round(gap_timing_score, 3)})

    # Sub-módulo 3: Pós-branco (se acabou de sair branco)
    w_post = weights.get("white_post", 1.0)
    if wh["post_white"] or dist <= 2:
        # Pouco após um branco, a prob de outro branco é baixa
        score += 0.05 * w_post
        weight_sum += w_post
        signals.append({"name": "white_post_recent", "conf": 0.05})

    # Sub-módulo 4: Continual Learner (transição para branco)
    w_cl = weights.get("white_cl", 1.2)
    with _cl_lock:
        trans = CL_STATE["transition_matrix"]
        ngram_cache = CL_STATE["ngram_cache"]
        dist_full = CL_STATE["color_dist_full"]

    last_color = colors[-1] if colors else None
    if last_color is not None and last_color in trans:
        cl_prob_white = trans[last_color].get(0, 0.07)
        score += cl_prob_white * w_cl * 2.0  # Amplifica pois base é baixa
        weight_sum += w_cl
        signals.append({"name": "white_cl_transition", "conf": round(cl_prob_white, 3)})

    # N-grama para branco
    for length in range(3, 7):
        if len(colors) >= length:
            key = tuple(colors[-length:])
            if key in ngram_cache:
                entry = ngram_cache[key]
                total_ng = entry.get("total", 0)
                white_ng = entry.get(0, 0)
                if total_ng >= 8 and white_ng >= 1:
                    ng_prob = bayes_prob(white_ng, total_ng, alpha=0.5)
                    if ng_prob >= 0.12:
                        score += ng_prob * w_cl
                        weight_sum += w_cl * 0.5
                        signals.append({"name": f"white_cl_ngram_{length}", "conf": round(ng_prob, 3)})
                        break

    if weight_sum == 0:
        return {"confidence": 0.0, "signals": signals, "color": 0}

    final_conf = min(score / weight_sum, 0.99)
    return {"confidence": round(final_conf, 4), "signals": signals, "color": 0}


# ─────────────────────────── CATÁLOGO + MINER HELPERS ────────────────────────
def load_catalog_strategies():
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_catalog'")
        if not c.fetchone():
            conn.close()
            return []
        c.execute("""
            SELECT strategy_id, family, name, params_json, target_color,
                   wf_acc, weight, quality_score, recent_acc
            FROM strategy_catalog WHERE status='active'
            ORDER BY weight DESC LIMIT 80
        """)
        rows = c.fetchall()
        conn.close()
        strats = []
        for r in rows:
            try:
                params = json.loads(r[3])
                strats.append({
                    "id": r[0], "family": r[1], "name": r[2], "params": params,
                    "target": r[4], "wf_acc": r[5] or 0.0, "weight": r[6] or 0.0,
                    "quality": r[7] or 0.0, "recent_acc": r[8] or 0.0, "source": "catalog",
                })
            except Exception:
                pass
        return strats
    except Exception:
        return []


def catalog_strategy_matches(strat, colors):
    family = strat["family"]
    p = strat["params"]

    if family == "exact_ngram":
        seq = p.get("seq", [])
        lag = len(seq)
        return colors[len(colors) - lag:] == seq if len(colors) >= lag else False

    if family == "run_edge":
        strk = streak_info(colors)
        return strk["color"] == p.get("run_color") and strk["length"] == p.get("run_size")

    if family == "white_gap":
        wh = poisson_white_hazard(colors)
        gap = wh["dist"]
        gb = ("0_4" if gap <= 4 else "5_9" if gap <= 9 else
              "10_14" if gap <= 14 else "15_22" if gap <= 22 else "23_plus")
        nw = [c for c in colors if c != 0]
        last_nw = nw[-1] if nw else None
        return gb == p.get("gap_bucket") and last_nw == p.get("last_color")

    if family == "alternation_edge":
        nw = [c for c in colors if c != 0]
        last_nw = nw[-1] if nw else None
        fr = alternation_ratio(colors, int(p.get("window", 6)))
        return last_nw == p.get("last_color") and fr >= float(p.get("min_flip_ratio", 0.75))

    return False


def refresh_catalog():
    global GLOBAL_CATALOG_STRATS
    while True:
        try:
            strats = load_catalog_strategies()
            with _catalog_lock:
                GLOBAL_CATALOG_STRATS = strats
            log.info("Catálogo atualizado: %d estratégias", len(strats))
        except Exception as e:
            log.warning("Erro catálogo: %s", e)
        time.sleep(60)


def mine_local_strategies(colors):
    global GLOBAL_MINED_STRATS
    n = len(colors)
    max_g = get_max_gales()
    patterns = defaultdict(lambda: {1: 0, 2: 0, "m": 0})
    recency_start = max(0, n - 300)

    for length in range(2, 8):
        for i in range(length - 1, n - 1 - max_g):
            seq = tuple(colors[i - length + 1: i + 1])
            d = patterns[seq]
            d["m"] += 1
            bonus = 2 if i >= recency_start else 1
            for alvo in (1, 2):
                for step in range(1 + max_g):
                    idx = i + 1 + step
                    if idx < n and colors[idx] == alvo:
                        d[alvo] += bonus
                        break

    strats = []
    for seq, d in patterns.items():
        m = d["m"]
        if m < MINER_MIN_MATCHES:
            continue
        for alvo in (1, 2):
            w = d[alvo]
            b = bayes_prob(w, m)
            if b >= MINER_MIN_BAYES:
                txt = "-".join(color_short(c) for c in seq)
                strats.append({
                    "id": f"mined_{txt}_{alvo}", "family": "exact_ngram_local",
                    "name": f"Local [{txt}]→{'V' if alvo==1 else 'P'}",
                    "seq": seq, "target": alvo, "prob": b, "matches": m, "wins": w,
                    "source": "miner", "wf_acc": 0.0, "weight": b * min(m / 30.0, 1.0),
                })

    strats.sort(key=lambda x: x["weight"], reverse=True)
    with _miner_lock:
        GLOBAL_MINED_STRATS = strats[:80]
    log.info("Miner: %d padrões", len(strats[:80]))


def miner_thread():
    last_count = 0
    while True:
        try:
            conn = _conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM results_raw")
            n = c.fetchone()[0]
            c.execute("SELECT color FROM results_raw ORDER BY id ASC")
            colors = [r[0] for r in c.fetchall()]
            conn.close()
            if n >= 100 and (n - last_count >= MINER_INTERVAL or last_count == 0):
                mine_local_strategies(colors)
                last_count = n
        except Exception as e:
            log.error("Miner erro: %s", e)
        time.sleep(8)


# ─────────────────────────── MoE ENGINE PRINCIPAL ────────────────────────────
def run_moe_engine(colors: list, regime: dict, last_round: dict = None) -> dict:
    """
    Executa o Mixture of Experts:
    1. Gating Network decide pesos por expert
    2. Cada expert (Red, Black, White) calcula sua confiança
    3. Meta-Learner combina com pesos neurais
    4. Retorna sinal final
    """
    gate_weights = gating_network(regime, colors)

    red_result = red_expert(colors, regime)
    black_result = black_expert(colors, regime)
    white_result = white_expert(colors, regime)

    # Aplicar pesos do Gating Network
    red_score = red_result["confidence"] * gate_weights["red"]
    black_score = black_result["confidence"] * gate_weights["black"]
    white_score = white_result["confidence"] * gate_weights["white"]

    # White Veto — se hazard muito alto, bloqueia tudo
    wh = poisson_white_hazard(colors)
    if wh["hazard"] >= 0.82:
        return {
            "action": "block", "color": None, "confidence": white_score,
            "expert_used": "white_veto",
            "reason": f"⚪ VETO BRANCO — Hazard {wh['hazard']:.0%} crítico",
            "kelly": 0.0, "gate_weights": gate_weights,
            "experts": {"red": red_result, "black": black_result, "white": white_result},
        }

    # Determinar vencedor
    scores = {1: red_score, 2: black_score, 0: white_score}
    winner = max(scores, key=scores.get)
    winner_conf = scores[winner]

    with _threshold_lock:
        threshold = _threshold_state["value"]
        mute_until = _threshold_state["mute_until_round"]
        total_rounds = _threshold_state["total_rounds_seen"]

    # Auto-Mute
    if total_rounds < mute_until:
        remaining = mute_until - total_rounds
        return {
            "action": "wait", "color": None, "confidence": winner_conf,
            "expert_used": "",
            "reason": f"🔇 Auto-Mute ativo ({remaining} rounds restantes)",
            "kelly": 0.0, "gate_weights": gate_weights,
            "experts": {"red": red_result, "black": black_result, "white": white_result},
        }

    # Threshold
    if winner_conf < threshold:
        return {
            "action": "wait", "color": None, "confidence": winner_conf,
            "expert_used": "",
            "reason": f"⏳ Edge {winner_conf:.0%} < threshold {threshold:.0%}",
            "kelly": 0.0, "gate_weights": gate_weights,
            "experts": {"red": red_result, "black": black_result, "white": white_result},
        }

    # Sinal de entrada
    expert_name = {1: "🔴 RED Expert", 2: "⚫ BLACK Expert", 0: "⚪ WHITE Expert"}[winner]
    color_label = {1: "VERMELHO 🔴", 2: "PRETO ⚫", 0: "BRANCO ⚪"}[winner]
    after_str = f" [Após {last_round['roll']} ({color_short(last_round['color'])})]" if last_round else ""

    # Kelly ajustado pelo gate weight
    kelly_val = kelly_fraction(winner_conf) * gate_weights[{1: "red", 2: "black", 0: "white"}[winner]]

    with _cl_lock:
        drift = CL_STATE["drift_detected"]

    drift_warning = " ⚠️ DRIFT ATIVO" if drift else ""

    return {
        "action": "enter", "color": winner, "confidence": round(winner_conf, 4),
        "expert_used": {1: "red", 2: "black", 0: "white"}[winner],
        "reason": f"🎯 {color_label}{after_str} | {expert_name} | Edge {winner_conf:.0%}{drift_warning}",
        "kelly": round(kelly_val * 100, 2),
        "gate_weights": gate_weights,
        "experts": {"red": red_result, "black": black_result, "white": white_result},
    }


# ─────────────────────────── GROQ VALIDATOR ──────────────────────────────────
def ask_groq_analyst(colors: list, moe_result: dict, regime: dict) -> dict:
    groq_key = (get_sys_config("groq_key", "") or "").strip()
    llm_enabled = get_sys_config("llm_enabled", "0")

    if llm_enabled != "1" or not groq_key:
        return {"status": "disabled", "model": "", "reason": "LLM desativado"}

    seq_str = "-".join(color_short(c) for c in colors[-25:])
    alvo_str = {1: "VERMELHO", 2: "PRETO", 0: "BRANCO"}.get(moe_result["color"], "?")
    gate = moe_result.get("gate_weights", {})

    with _threshold_lock:
        hist = list(_threshold_state["history"])[-15:]
    wr_str = f"{sum(hist)}/{len(hist)}" if hist else "sem histórico"

    with _cl_lock:
        drift = CL_STATE["drift_detected"]
        drift_mag = CL_STATE["drift_magnitude"]

    prompt = f"""Você é analista quantitativo especialista em Blaze Double.
Analise e responda APENAS: CONFIRMAR, REDUZIR ou VETAR.

• Últimas 25: {seq_str}
• Regime: {regime['label']}
• Expert ativo: {moe_result.get('expert_used', '?')} → {alvo_str}
• Edge: {moe_result['confidence']:.1%}
• Gate weights: R={gate.get('red',0):.2f} B={gate.get('black',0):.2f} W={gate.get('white',0):.2f}
• Performance recente: {wr_str}
• Concept Drift: {'SIM magnitude=' + str(round(drift_mag,3)) if drift else 'NÃO'}"""

    for model in get_groq_models():
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": model, "temperature": 0.05, "max_tokens": 6,
                    "messages": [
                        {"role": "system", "content": "Responda CONFIRMAR, REDUZIR ou VETAR."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=6,
            )
            if resp.status_code != 200:
                continue
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            if "VETAR" in answer: return {"status": "vetar", "model": model, "reason": answer}
            if "REDUZIR" in answer: return {"status": "reduzir", "model": model, "reason": answer}
            if "CONFIRMAR" in answer: return {"status": "confirmar", "model": model, "reason": answer}
            return {"status": "inconclusivo", "model": model, "reason": answer}
        except Exception as e:
            log.debug("Groq %s falhou: %s", model, e)

    return {"status": "error", "model": "", "reason": "Groq indisponível"}


# ─────────────────────────── ADAPTIVE THRESHOLD ──────────────────────────────
def update_threshold(won: bool):
    with _threshold_lock:
        _threshold_state["history"].append(won)
        _threshold_state["total_rounds_seen"] += 1

        if won:
            _threshold_state["consecutive_losses"] = 0
            _threshold_state["consecutive_wins"] += 1
            _threshold_state["mute_win_counter"] += 1
            if _threshold_state["mute_win_counter"] >= MUTE_SCORE_NEEDED:
                _threshold_state["mute_until_round"] = 0
                _threshold_state["mute_win_counter"] = 0
                log.info("🔊 Auto-Mute encerrado")
            new_t = max(THRESHOLD_MIN, _threshold_state["value"] - THRESHOLD_STEP_DOWN)
        else:
            _threshold_state["consecutive_wins"] = 0
            _threshold_state["consecutive_losses"] += 1
            if _threshold_state["consecutive_losses"] >= AUTO_MUTE_LOSSES:
                mute_end = _threshold_state["total_rounds_seen"] + AUTO_MUTE_ROUNDS
                _threshold_state["mute_until_round"] = mute_end
                _threshold_state["mute_win_counter"] = 0
                log.warning("🔇 AUTO-MUTE ATIVADO por %d rounds", AUTO_MUTE_ROUNDS)
            new_t = min(THRESHOLD_MAX, _threshold_state["value"] + THRESHOLD_STEP_UP)

        _threshold_state["value"] = round(new_t, 4)

    save_meta("threshold", str(_threshold_state["value"]))
    save_meta("consecutive_loss", str(_threshold_state["consecutive_losses"]))
    save_meta("consecutive_win", str(_threshold_state["consecutive_wins"]))
    save_meta("mute_until", str(_threshold_state["mute_until_round"]))
    save_meta("mute_wins", str(_threshold_state["mute_win_counter"]))


def load_threshold_state():
    with _threshold_lock:
        _threshold_state["value"] = float(load_meta("threshold", str(THRESHOLD_START)))
        _threshold_state["consecutive_losses"] = int(load_meta("consecutive_loss", "0"))
        _threshold_state["consecutive_wins"] = int(load_meta("consecutive_win", "0"))
        _threshold_state["mute_until_round"] = int(load_meta("mute_until", "0"))
        _threshold_state["mute_win_counter"] = int(load_meta("mute_wins", "0"))
    log.info("Threshold restaurado: %.2f%%", _threshold_state["value"] * 100)


# ─────────────────────────── ENGINE WRAPPER ──────────────────────────────────
def run_engine(seq: list) -> dict:
    colors = [r["color"] for r in seq]
    wh = poisson_white_hazard(colors)
    regime = detect_regime(colors)
    result = run_moe_engine(colors, regime, seq[-1])

    groq_status = ""
    groq_model = ""

    if result["action"] == "enter":
        groq = ask_groq_analyst(colors, result, regime)
        groq_status = groq["status"]
        groq_model = groq["model"]

        if groq_status == "vetar":
            result["action"] = "block"
            result["confidence"] = min(result["confidence"] * 0.5, 0.45)
            result["reason"] = f"🤖 VETO LLM ({groq_model}): {result['reason']}"
            result["kelly"] = 0.0
        elif groq_status == "reduzir":
            result["kelly"] = round(result["kelly"] * 0.5, 2)
            result["reason"] = f"⚡ KELLY REDUZIDO: {result['reason']}"
        elif groq_status == "confirmar":
            result["kelly"] = round(min(result["kelly"] * 1.3, 5.0), 2)
            result["reason"] = f"✅ LLM CONFIRMADO: {result['reason']}"

    white_p = 0.07
    c = result["color"]
    conf = max(0.0, min(result["confidence"], 1.0 - white_p))
    if c == 1:
        probs = {"red": conf, "black": 1.0 - white_p - conf, "white": white_p}
    elif c == 2:
        probs = {"red": 1.0 - white_p - conf, "black": conf, "white": white_p}
    else:
        probs = {"red": 0.465, "black": 0.465, "white": 0.07}

    gate = result.get("gate_weights", {})
    experts = result.get("experts", {})

    with _cl_lock:
        cl_info = {
            "dist_500": CL_STATE["color_dist_500"],
            "drift": CL_STATE["drift_detected"],
            "drift_mag": round(CL_STATE["drift_magnitude"], 4),
            "last_relearn": CL_STATE["last_relearn_ts"],
        }

    return {
        "n": len(seq),
        "signal": {
            "action": result["action"], "color": result["color"],
            "confidence": result["confidence"], "kelly": result["kelly"],
            "reason": result["reason"], "expert_used": result.get("expert_used", ""),
        },
        "probs": probs,
        "regime": {"regime": regime["name"], "label": regime["label"], "strength": regime["strength"]},
        "tests": {"white": wh},
        "features": {
            "llm_status": groq_status, "llm_model": groq_model,
            "kelly_pct": result["kelly"],
            "gate_red": gate.get("red", 0), "gate_black": gate.get("black", 0),
            "gate_white": gate.get("white", 0),
            "expert_red_conf": experts.get("red", {}).get("confidence", 0),
            "expert_black_conf": experts.get("black", {}).get("confidence", 0),
            "expert_white_conf": experts.get("white", {}).get("confidence", 0),
            "threshold_used": _threshold_state["value"],
            "regime_name": regime["name"],
            "miner_count": len(GLOBAL_MINED_STRATS),
            "catalog_count": len(GLOBAL_CATALOG_STRATS),
            "cl_drift": cl_info["drift"],
            "cl_drift_mag": cl_info["drift_mag"],
            "cl_last_relearn": cl_info["last_relearn"] or "",
            "votes_json": json.dumps([
                {"module": k, "conf": round(v.get("confidence", 0), 3)}
                for k, v in experts.items()
            ], ensure_ascii=False),
        },
    }


# ─────────────────────────── SALVAR SNAPSHOT ─────────────────────────────────
def save_snapshot(seq: list, engine_result: dict) -> int:
    lid = seq[-1]["round_id"] if seq else None
    s = engine_result["signal"]
    p = engine_result["probs"]
    reg = engine_result["regime"]
    wh = engine_result["tests"]["white"]
    feat = engine_result["features"]

    conn = _conn()
    cur = conn.execute(
        """INSERT INTO analysis_snapshots
        (ts, total_rounds, last_round_id, prob_red, prob_black, prob_white,
         signal_color, signal_conf, signal_action, signal_reason,
         regime, regime_strength, white_hazard, dist_last_white,
         features_json, patterns_json, mode_used, votes_json, threshold_used)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(), len(seq), lid,
            p["red"], p["black"], p["white"],
            s.get("color"), s.get("confidence"), s.get("action"), s.get("reason"),
            reg.get("label"), reg["strength"], wh["hazard"], wh["dist"],
            json.dumps(feat, ensure_ascii=False, default=str),
            json.dumps([{"name": s.get("reason", "")[:80], "strength": s.get("confidence", 0)}]),
            "moe_v2", feat.get("votes_json", "[]"), feat.get("threshold_used", THRESHOLD_START),
        ),
    )
    snap_id = cur.lastrowid
    conn.commit()
    conn.close()
    return snap_id


# ─────────────────────────── VALIDAÇÃO DE PERFORMANCE ────────────────────────
def validate_previous(current_color: int, current_round_id: str):
    conn = _conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, last_round_id, signal_color, signal_action, features_json
            FROM analysis_snapshots
            WHERE signal_action IN ('enter', 'gale_1', 'gale_2')
            ORDER BY id DESC LIMIT 1
        """)
        row = c.fetchone()
        if not row: return

        snap_id, pred_last_rid, pred_color, s_action, feat_json_str = row

        c.execute("SELECT action FROM prediction_performance WHERE snapshot_id=?", (snap_id,))
        if c.fetchone(): return

        c.execute("SELECT id FROM results_raw WHERE round_id=?", (pred_last_rid,))
        pred_db = c.fetchone()
        c.execute("SELECT id FROM results_raw WHERE round_id=?", (current_round_id,))
        curr_db = c.fetchone()
        if not pred_db or not curr_db: return

        gap = curr_db[0] - pred_db[0]
        if not (1 <= gap <= max(2, get_max_gales() + 1)): return

        if current_color == 0:
            correct = 1
            action_res = "empate_branco"
            won = True
        else:
            correct = 1 if current_color == pred_color else 0
            won = bool(correct)
            if correct:
                action_res = "win"
            else:
                gale_step = {"enter": 0, "gale_1": 1, "gale_2": 2}.get(s_action, 0)
                action_res = "gale_pending" if gale_step < get_max_gales() else "loss"

        try:
            feat = json.loads(feat_json_str or "{}")
            expert_used = feat.get("expert_used", "")
        except Exception:
            expert_used = ""

        c.execute("""
            INSERT INTO prediction_performance
            (snapshot_id, ts, predicted, actual, correct, action, mode, pattern_key, expert_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snap_id, datetime.now(timezone.utc).isoformat(),
            pred_color, current_color, correct, action_res, "moe_v2", "", expert_used,
        ))
        conn.commit()

        if action_res in ("win", "loss", "empate_branco"):
            update_threshold(won)

            # Atualizar pesos neurais do expert específico
            if expert_used and pred_color is not None:
                color_key = {1: "red", 2: "black", 0: "white"}.get(pred_color, "")
                if color_key:
                    try:
                        feat = json.loads(feat_json_str or "{}")
                        conf = feat.get(f"expert_{color_key}_conf", 0.5)
                        for sub in ["markov", "miner", "catalog", "cl", "streak",
                                    "poisson", "gap", "post"]:
                            key = f"{color_key}_{sub}"
                            update_neural_weights(key, won, float(conf))
                    except Exception:
                        pass

            log.info(
                "📊 MoE Performance: %s | expert=%s | predicted=%s actual=%s | threshold=%.2f%%",
                action_res.upper(), expert_used, color_name(pred_color),
                color_name(current_color), _threshold_state["value"] * 100,
            )
    finally:
        conn.close()


# ─────────────────────────── GALE INTELIGENTE ────────────────────────────────
def check_pending_gale(seq: list):
    max_gales = get_max_gales()
    if max_gales == 0: return None

    conn = _conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, last_round_id, signal_color, signal_action
            FROM analysis_snapshots
            WHERE signal_action IN ('enter', 'gale_1', 'gale_2')
            ORDER BY id DESC LIMIT 1
        """)
        row = c.fetchone()
        if not row: return None

        snap_id, pred_rid, pred_color, action = row

        c.execute("SELECT action FROM prediction_performance WHERE snapshot_id=?", (snap_id,))
        perf = c.fetchone()
        if not perf or perf[0] in ("win", "empate_branco"): return None

        c.execute("SELECT id FROM results_raw WHERE round_id=?", (pred_rid,))
        db_id_row = c.fetchone()
        if not db_id_row: return None

        gap = seq[-1]["id"] - db_id_row[0]
        gale_step = {"enter": 0, "gale_1": 1, "gale_2": 2}.get(action, 0)

        if gap == 1 and gale_step < max_gales:
            colors = [r["color"] for r in seq]
            regime = detect_regime(colors)
            moe_check = run_moe_engine(colors, regime, seq[-1])

            still_valid = (moe_check["action"] == "enter" and moe_check["color"] == pred_color)
            next_gale = gale_step + 1
            last_round = seq[-1]
            after_str = f" [Após {last_round['roll']} ({color_short(last_round['color'])})]"

            if still_valid:
                kelly_g = round(kelly_fraction(moe_check["confidence"]) * 2 * 100, 2)
                return {
                    "action": f"gale_{next_gale}", "color": pred_color,
                    "confidence": moe_check["confidence"], "kelly": kelly_g,
                    "reason": f"🔥 Gale {next_gale} CONFIRMADO{after_str} pelo MoE ({moe_check['confidence']:.0%})",
                    "expert_used": moe_check.get("expert_used", ""),
                }
            else:
                log.warning("🛡️ GALE %d ABORTADO — MoE detectou mudança estrutural.", next_gale)
                return None
    finally:
        conn.close()
    return None


# ─────────────────────────── CICLO PRINCIPAL ─────────────────────────────────
_last_notified = None


def run_analysis_cycle() -> bool:
    global _last_notified

    last_db = get_last_round_id()
    last_an = get_last_analyzed_round_id()
    if last_db is None or last_db == last_an: return False

    seq = load_sequence(4000)
    n = len(seq)
    if n < MIN_HISTORY: return False

    with _threshold_lock:
        _threshold_state["total_rounds_seen"] = n

    validate_previous(seq[-1]["color"], seq[-1]["round_id"])
    gale_sig = check_pending_gale(seq)

    if gale_sig:
        colors = [r["color"] for r in seq]
        wh = poisson_white_hazard(colors)
        regime = detect_regime(colors)
        with _cl_lock:
            cl_info = {
                "drift": CL_STATE["drift_detected"],
                "drift_mag": CL_STATE["drift_magnitude"],
                "last_relearn": CL_STATE["last_relearn_ts"],
            }
        er = {
            "n": n,
            "signal": {
                "action": gale_sig["action"], "color": gale_sig["color"],
                "confidence": gale_sig["confidence"], "kelly": gale_sig["kelly"],
                "reason": gale_sig["reason"], "expert_used": gale_sig.get("expert_used", ""),
            },
            "probs": {"red": 0.5, "black": 0.5, "white": 0.0},
            "regime": {"regime": regime["name"], "label": regime["label"], "strength": regime["strength"]},
            "tests": {"white": wh},
            "features": {
                "llm_status": "", "llm_model": "", "kelly_pct": gale_sig["kelly"],
                "gate_red": 0, "gate_black": 0, "gate_white": 0,
                "expert_red_conf": 0, "expert_black_conf": 0, "expert_white_conf": 0,
                "threshold_used": _threshold_state["value"], "regime_name": regime["name"],
                "miner_count": len(GLOBAL_MINED_STRATS), "catalog_count": len(GLOBAL_CATALOG_STRATS),
                "cl_drift": cl_info["drift"], "cl_drift_mag": cl_info["drift_mag"],
                "cl_last_relearn": cl_info["last_relearn"] or "", "votes_json": "[]",
            },
        }
    else:
        er = run_engine(seq)

    snap_id = save_snapshot(seq, er)
    s = er["signal"]
    feat = er["features"]

    log.info("═" * 90)
    log.info(
        "MoE v2 | Rounds=%d | Regime=%s | Expert=%s | Sinal=%s | Cor=%s | Edge=%.1f%% | Kelly=%.2f%%",
        n, feat.get("regime_name", "?").upper(), s.get("expert_used", "?").upper(),
        s["action"].upper(), color_name(s["color"]), s["confidence"] * 100, s["kelly"],
    )
    log.info(
        "Gate R=%.2f B=%.2f W=%.2f | Threshold=%.2f%% | Miner=%d | Catálogo=%d | Drift=%s (%.4f) | CL=%s",
        feat.get("gate_red", 0), feat.get("gate_black", 0), feat.get("gate_white", 0),
        feat.get("threshold_used", 0) * 100,
        feat.get("miner_count", 0), feat.get("catalog_count", 0),
        "⚠️SIM" if feat.get("cl_drift") else "✅NÃO",
        feat.get("cl_drift_mag", 0), feat.get("cl_last_relearn", "-"),
    )
    log.info("Fundamento: %s", s["reason"])
    log.info("═" * 90)

    if TELEGRAM_OK and s["action"] in ("enter", "gale_1", "gale_2"):
        uniq = f"{s['action']}|{s['color']}|{seq[-1]['round_id']}"
        if uniq != _last_notified:
            notificar_sinal(s, er["regime"], [], er["probs"])
            _last_notified = uniq
    elif s["action"] in ("wait", "block"):
        _last_notified = None

    return True


# ─────────────────────────── MAIN ────────────────────────────────────────────
def main():
    log.info("═" * 90)
    log.info("  BLAZE DOUBLE AI — LEVIATHAN ENGINE v2.0 (MIXTURE OF EXPERTS)")
    log.info("  MoE: Red Expert | Black Expert | White Expert | Gating Network")
    log.info("  Continual Learning | Concept Drift Detection | Neural Meta-Learner")
    log.info("  Iniciado: %s", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    log.info("═" * 90)

    init_tables()
    load_threshold_state()

    threading.Thread(target=miner_thread, daemon=True, name="Miner").start()
    threading.Thread(target=refresh_catalog, daemon=True, name="Catalog").start()
    threading.Thread(target=continual_learner_thread, daemon=True, name="ContinualLearner").start()

    log.info("Threads: Miner + Catalog + ContinualLearner + MainLoop")

    try:
        while True:
            try:
                run_analysis_cycle()
            except Exception as e:
                log.error("Erro no ciclo: %s", e, exc_info=True)
            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        log.info("Leviathan MoE v2.0 encerrado.")


if __name__ == "__main__":
    main()