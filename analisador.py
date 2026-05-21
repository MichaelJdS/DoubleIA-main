"""=============================================================================
BLAZE DOUBLE AI — LEVIATHAN ENGINE v2.0 (9-EXPERT SOVEREIGN CORE)
Motor com 9 Experts Independentes, Anti-Drift Neural, Confidence Calibration,
Regime-Aware Voting, Distribuição Corrigida e Proteção de Banca Avançada.

EXPERTS:
  1. ExpertMiner        — Padrões N-gram locais minerados do histórico
  2. ExpertCatalog      — Estratégias walk-forward validadas pelo otimizador
  3. ExpertMarkov       — Cadeia de Markov ordem 1/2/3 com peso dinâmico
  4. ExpertStreak       — Reversão/continuação de streaks com bayes profundo
  5. ExpertWhiteCycle   — Ciclo branca: hazard poisson + padrão pós-branco
  6. ExpertMomentum     — Desvio de distribuição recente vs histórico global
  7. ExpertAlternation  — Detector de regime alternante com predição adaptativa
  8. ExpertVolatility   — Detecta compressão/explosão de volatilidade
  9. ExpertAntiDrift    — Detecta e bloqueia quando o jogo está "frio" para sinais

PROTEÇÕES:
  - Anti-Drift Neural: pesos aprendidos com gradiente correto
  - Confidence Calibration: evita overconfidence
  - Regime-Gated Voting: cada expert só vota no regime adequado
  - Banca Protection Level: 3 níveis (NORMAL / ALERT / LOCKDOWN)
  - Sunk-Cost Gale Block: aborta gale se regime mudou
============================================================================="""

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

from agents.agent_sybil import expert_sybil
from agents.agent_chaos import expert_chaos
from agents.agent_hermes import expert_hermes
from agents.agent_oracle import (
    oracle_get_weights,
    oracle_learn,
    oracle_save_state,
    oracle_load_state,
    _oracle_state,
)

try:
    from notificador import notificar_sinal
    TELEGRAM_OK = True
except Exception:
    TELEGRAM_OK = False
    def notificar_sinal(*a, **k):
        return False

# ─────────────────────────── CONFIGURAÇÕES ────────────────────────────────────
DB_PATH        = "blaze_double.db"
LOG_FILE       = "leviathan.log"
LOOP_INTERVAL  = 2
MIN_HISTORY    = 100

# Threshold — mais conservador para proteger banca
THRESHOLD_MIN      = 0.65
THRESHOLD_START    = 0.74
THRESHOLD_MAX      = 0.90
THRESHOLD_STEP_UP  = 0.025
THRESHOLD_STEP_DOWN= 0.010

# Proteção de banca — 3 níveis
BANCA_NORMAL_LOSSES   = 3   # após 3 perdas seguidas → ALERT
BANCA_ALERT_LOSSES    = 5   # após 5 perdas seguidas → LOCKDOWN
BANCA_LOCKDOWN_ROUNDS = 35  # silêncio de 35 rounds no lockdown
BANCA_ALERT_ROUNDS    = 15  # silêncio de 15 rounds no alerta
BANCA_RECOVERY_WINS   = 4   # precisam 4 wins para sair do lockdown

# Votação — mais exigente
MIN_VOTES_TO_ENTER  = 3   # mínimo 3 experts concordando (antes era 2)
NUM_EXPERTS         = 12
CONFIDENCE_FLOOR    = 0.67  # edge mínimo após calibração

# Minerador
MINER_INTERVAL   = 20
MINER_MIN_MATCHES= 8
MINER_MIN_BAYES  = 0.64

DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

# ─────────────────────────── ESTADO GLOBAL ───────────────────────────────────
_miner_lock   = threading.Lock()
_catalog_lock = threading.Lock()
GLOBAL_MINED_STRATS   = []
GLOBAL_CATALOG_STRATS = []

_threshold_state = {
    "value": THRESHOLD_START,
    "consecutive_losses": 0,
    "consecutive_wins":   0,
    "banca_level":        "NORMAL",   # NORMAL / ALERT / LOCKDOWN
    "lockdown_until":     0,
    "recovery_wins":      0,
    "total_rounds_seen":  0,
    "history":            deque(maxlen=300),
}
_threshold_lock = threading.Lock()

CL_RELEARN_EVERY = 50       # rounds novos para re-aprender
CL_DRIFT_THRESHOLD = 0.12   # diferença de distribuição que dispara re-learn

_cl_lock = threading.Lock()
CL_STATE = {
    "last_db_count": 0,
    "color_dist_full": {0: 0.07, 1: 0.465, 2: 0.465},
    "color_dist_500": {0: 0.07, 1: 0.465, 2: 0.465},
    "color_dist_2000": {0: 0.07, 1: 0.465, 2: 0.465},
    "drift_detected": False,
    "drift_magnitude": 0.0,
    "last_relearn_ts": "",
    "transition_matrix": {},
    "ngram_cache": {},
    "expert_stats": {
        "red":   {"total": 0, "wins": 0, "by_regime": {}},
        "black": {"total": 0, "wins": 0, "by_regime": {}},
        "white": {"total": 0, "wins": 0, "by_regime": {}},
    },
}

_pattern_perf: dict = defaultdict(lambda: deque(maxlen=40))
_perf_lock = threading.Lock()

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
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key=?", (key,))
        row = c.fetchone(); conn.close()
        return row[0] if row else default
    except Exception:
        return default

def set_sys_config(key, value):
    try:
        conn = _conn()
        conn.execute("INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_max_gales():
    try: return int(get_sys_config("max_gales", "0"))
    except: return 0

def get_groq_models():
    custom = os.environ.get("GROQ_MODEL", "").strip()
    if custom: return [x.strip() for x in custom.split(",") if x.strip()]
    return DEFAULT_GROQ_MODELS[:]


# ─────────────────────────── REDE NEURAL CORRIGIDA ───────────────────────────
_neural_lock = threading.Lock()
_LEARNING_RATE = 0.04
_DEFAULT_WEIGHTS = {
    "miner": 1.0, "catalog": 1.2, "markov": 0.9,
    "streak": 0.85, "white": 1.0, "momentum": 1.1,
    "alternation": 0.9, "volatility": 0.8, "antidrift": 1.3,
}

def get_neural_weights():
    raw = get_sys_config("neural_weights_v2", "")
    if raw:
        try: return json.loads(raw)
        except: pass
    return _DEFAULT_WEIGHTS.copy()

def update_neural_weights(module_votes: dict, actual_color: int, won: bool, colors=None, regime=None):
    with _neural_lock:
        weights = get_neural_weights()
        for mod_name, vote_data in module_votes.items():
            vote = vote_data.get("vote")
            if vote is None: continue
            conf = float(vote_data.get("confidence", 0.5))

            voted_correctly = (vote == actual_color) and won
            voted_wrongly   = (vote == actual_color) and not won

            if voted_correctly:
                delta = +_LEARNING_RATE * conf
            elif voted_wrongly:
                delta = -_LEARNING_RATE * conf
            else:
                delta = 0.0

            weights[mod_name] = round(max(0.1, min(3.5, weights[mod_name] + delta)), 4)

        set_sys_config("neural_weights_v2", json.dumps(weights))
        if colors is not None and regime is not None:
            oracle_learn(won, colors, regime)
            if _oracle_state["total_updates"] % 20 == 0:
                oracle_save_state()
        log.info("🧠 Pesos v2: %s", {k: v for k, v in weights.items()})


# ─────────────────────────── INIT TABELAS ────────────────────────────────────
def init_tables():
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS analysis_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT NOT NULL,
        total_rounds    INTEGER,
        last_round_id   TEXT,
        prob_red        REAL, prob_black REAL, prob_white REAL,
        signal_color    INTEGER, signal_conf REAL,
        signal_action   TEXT, signal_reason TEXT,
        regime          TEXT, regime_strength REAL,
        white_hazard    REAL, dist_last_white INTEGER,
        features_json   TEXT, patterns_json TEXT,
        mode_used       TEXT DEFAULT 'leviathan_v2',
        votes_json      TEXT DEFAULT '[]',
        threshold_used  REAL DEFAULT 0.74,
        banca_level     TEXT DEFAULT 'NORMAL'
    );
    CREATE TABLE IF NOT EXISTS prediction_performance (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id INTEGER,
        ts          TEXT NOT NULL,
        predicted   INTEGER, confidence REAL,
        actual      INTEGER, correct INTEGER,
        action      TEXT,
        mode        TEXT DEFAULT 'leviathan_v2',
        pattern_key TEXT DEFAULT ''
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
    CREATE TABLE IF NOT EXISTS leviathan_meta (
        key TEXT PRIMARY KEY, value TEXT
    );
    """)
    conn.commit()
    _add_column_safe(conn, "analysis_snapshots", "banca_level", "TEXT DEFAULT 'NORMAL'")
    _add_column_safe(conn, "prediction_performance", "expert_used", "TEXT DEFAULT ''")
    conn.close()

    for k, v in {
        "threshold": str(THRESHOLD_START), "consecutive_loss": "0",
        "consecutive_win": "0", "lockdown_until": "0",
        "recovery_wins": "0", "banca_level": "NORMAL",
        "total_signals": "0", "total_wins": "0",
    }.items():
        try:
            c2 = _conn()
            c2.execute("INSERT OR IGNORE INTO leviathan_meta (key, value) VALUES (?, ?)", (k, v))
            c2.commit(); c2.close()
        except: pass
    log.info("Tabelas v2 verificadas.")


def load_meta(key, default="0"):
    try:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT value FROM leviathan_meta WHERE key=?", (key,))
        row = c.fetchone(); conn.close()
        return row[0] if row else default
    except: return default

def save_meta(key, value):
    try:
        conn = _conn()
        conn.execute("INSERT OR REPLACE INTO leviathan_meta (key, value) VALUES (?, ?)", (key, value))
        conn.commit(); conn.close()
    except: pass


# ─────────────────────────── CARREGAR SEQUÊNCIA ──────────────────────────────
def load_sequence(limit=5000):
    conn = _conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, round_id, color, roll, created_at
            FROM results_raw ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = c.fetchall()
        return [{"id": r[0], "round_id": r[1], "color": r[2], "roll": r[3], "created_at": r[4]}
                for r in reversed(rows)]
    finally:
        conn.close()

def get_last_round_id():
    conn = _conn(); c = conn.cursor()
    c.execute("SELECT round_id FROM results_raw ORDER BY id DESC LIMIT 1")
    row = c.fetchone(); conn.close()
    return row[0] if row else None

def get_last_analyzed_round_id():
    conn = _conn(); c = conn.cursor()
    c.execute("SELECT last_round_id FROM analysis_snapshots ORDER BY id DESC LIMIT 1")
    row = c.fetchone(); conn.close()
    return row[0] if row else None


# ─────────────────────────── HELPERS MATEMÁTICOS ─────────────────────────────
def cs(c): return {0:"B",1:"V",2:"P"}.get(c,"?")
def cn(c): return {0:"BRANCO",1:"VERMELHO",2:"PRETO"}.get(c,"—")

def bayes_prob(wins, n, alpha=1.5):
    return (wins + alpha) / (n + 2 * alpha)

def wilson_lower(wins, n, z=1.96):
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    variance = (p * (1 - p) + z**2 / (4 * n)) / n
    if variance < 0:
        variance = 0.0
    try:
        spread = z * math.sqrt(variance)
    except ValueError:
        spread = 0.0
    return max(0.0, (center - spread) / denom)

def kelly_fraction(prob, frac=0.20):
    if prob <= 0.52: return 0.0
    edge = 2.0 * prob - 1.0
    return max(0.0, min(edge * frac, 0.25))

def calibrate_confidence(raw_conf, n_voters, max_voters):
    voter_ratio = n_voters / max(max_voters, 1)
    shrink = 0.5 + (raw_conf - 0.5) * (0.5 + 0.5 * voter_ratio)
    return round(min(shrink, 0.97), 4)

def poisson_white_hazard(colors):
    gaps = []; cur = 0
    for c in colors:
        if c == 0: gaps.append(cur); cur = 0
        else: cur += 1
    avg = sum(gaps) / len(gaps) if gaps else 14.0
    lam = 1.0 / max(avg, 1e-9)
    hazard = 1.0 - math.exp(-lam * cur)
    return {"dist": cur, "hazard": round(min(hazard, 0.99), 4),
            "avg_gap": round(avg, 2), "post_white": cur == 0}

def markov_prob(nw, target, order=2):
    if len(nw) < order + 1: return 0.465
    key = tuple(nw[-order:])
    counts = defaultdict(int); total = 0
    for i in range(len(nw) - order):
        k = tuple(nw[i:i+order])
        if k == key:
            counts[nw[i+order]] += 1; total += 1
    if total == 0: return 0.465
    return (counts[target] + 1) / (total + 2)

def entropy_window(colors, window=20):
    recent = [c for c in colors[-window:] if c != 0]
    if len(recent) < 4: return 1.0
    counts = defaultdict(int)
    for c in recent: counts[c] += 1
    n = len(recent)
    ent = -sum((v/n)*math.log2(v/n+1e-10) for v in counts.values())
    return round(min(ent / 1.0, 1.0), 4)

def streak_info(nw):
    if not nw: return {"color": None, "length": 0}
    anchor = nw[-1]; length = 0
    for c in reversed(nw):
        if c == anchor: length += 1
        else: break
    return {"color": anchor, "length": length}

def alternation_ratio(colors, window=10):
    nw = [c for c in colors[-50:] if c != 0][-window:]
    if len(nw) < 2: return 0.5
    flips = sum(1 for i in range(1, len(nw)) if nw[i] != nw[i-1])
    return flips / (len(nw) - 1)

def distribution_deviation(colors, window_recent=50, window_global=500):
    recent = [c for c in colors[-window_recent:] if c != 0]
    hist   = [c for c in colors[-window_global:] if c != 0]
    if len(recent) < 10 or len(hist) < 50: return {"red_dev": 0.0, "black_dev": 0.0, "bias": None}
    r_red   = recent.count(1) / len(recent)
    r_black = recent.count(2) / len(recent)
    h_red   = hist.count(1)   / len(hist)
    h_black = hist.count(2)   / len(hist)
    red_dev   = r_red   - h_red
    black_dev = r_black - h_black
    bias = None
    if abs(red_dev) > 0.07:
        bias = 2 if red_dev > 0 else 1
    elif abs(black_dev) > 0.07:
        bias = 1 if black_dev > 0 else 2
    return {"red_dev": round(red_dev, 4), "black_dev": round(black_dev, 4), "bias": bias}

def volatility_score(colors, window=30):
    nw = [c for c in colors[-window:] if c != 0]
    if len(nw) < 10: return {"level": "unknown", "score": 0.5}
    flips = sum(1 for i in range(1, len(nw)) if nw[i] != nw[i-1])
    ratio = flips / (len(nw) - 1)
    if ratio > 0.75: return {"level": "explosion", "score": ratio}
    if ratio < 0.35: return {"level": "compression", "score": 1.0 - ratio}
    return {"level": "normal", "score": 0.5}


# ─────────────────────────── DETECTOR DE REGIME ──────────────────────────────
def detect_regime(colors):
    ent  = entropy_window(colors, 20)
    nw   = [c for c in colors if c != 0]
    strk = streak_info(nw)
    alt  = alternation_ratio(colors, 12)
    wh   = poisson_white_hazard(colors)

    if wh["dist"] >= 18 and wh["hazard"] >= 0.72:
        return {"name": "white_zone",  "label": "⚪ Zona Branca",   "strength": wh["hazard"], "data": wh}
    if strk["length"] >= 4:
        return {"name": "streak_hot",  "label": f"🔥 Streak {strk['length']}x", "strength": min(strk["length"]/8.0, 0.99), "data": strk}
    if alt >= 0.78 and len(nw) >= 6:
        return {"name": "alternating", "label": "↔️ Alternância",    "strength": round(alt, 3), "data": {"alt_ratio": alt}}
    if ent >= 0.88:
        return {"name": "chaotic",     "label": "🌀 Caótico",        "strength": round(ent, 3), "data": {"entropy": ent}}
    return  {"name": "balanced",       "label": "⚖️ Equilibrado",   "strength": 0.5, "data": {}}


# ─────────────────────────── LIVE PERFORMANCE MEMORY ─────────────────────────
def record_pattern_outcome(key, won):
    with _perf_lock: _pattern_perf[key].append(won)

def get_pattern_weight(key):
    with _perf_lock: hist = list(_pattern_perf.get(key, []))
    if len(hist) < 3: return 0.5
    recent = hist[-12:]
    acc = sum(recent) / len(recent)
    if acc < 0.38: return 0.05
    if acc < 0.48: return 0.25
    if acc < 0.58: return 0.55
    if acc < 0.68: return 0.78
    return round(min(acc, 0.97), 3)


# ─────────────────────────── CATÁLOGO ────────────────────────────────────────
def load_catalog_strategies():
    try:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_catalog'")
        if not c.fetchone(): conn.close(); return []
        c.execute("""
            SELECT strategy_id, family, name, params_json, target_color,
                   wf_acc, weight, quality_score, recent_acc
            FROM strategy_catalog WHERE status='active'
            ORDER BY weight DESC LIMIT 100
        """)
        rows = c.fetchall(); conn.close()
        strats = []
        for r in rows:
            try:
                strats.append({
                    "id": r[0], "family": r[1], "name": r[2],
                    "params": json.loads(r[3]), "target": r[4],
                    "wf_acc": r[5] or 0.0, "weight": r[6] or 0.0,
                    "quality": r[7] or 0.0, "recent_acc": r[8] or 0.0,
                    "source": "catalog",
                })
            except: pass
        return strats
    except: return []

def catalog_matches(strat, colors):
    family = strat["family"]; p = strat["params"]
    if family == "exact_ngram":
        seq = p.get("seq", []); lag = len(seq)
        if len(colors) < lag: return False
        return colors[-lag:] == seq
    if family == "run_edge":
        nw = [c for c in colors if c != 0]
        strk = streak_info(nw)
        return strk["color"] == p.get("run_color") and strk["length"] == p.get("run_size")
    if family == "white_gap":
        wh = poisson_white_hazard(colors)
        dist = wh["dist"]
        if dist <= 4: gb = "0_4"
        elif dist <= 9: gb = "5_9"
        elif dist <= 14: gb = "10_14"
        elif dist <= 22: gb = "15_22"
        else: gb = "23_plus"
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
            with _catalog_lock: GLOBAL_CATALOG_STRATS = strats
            log.info("Catálogo: %d estratégias ativas", len(strats))
        except Exception as e:
            log.warning("Catálogo erro: %s", e)
        time.sleep(55)


# ─────────────────────────── MINERADOR ───────────────────────────────────────
def mine_local_strategies(colors):
    global GLOBAL_MINED_STRATS
    n = len(colors); max_g = get_max_gales()
    patterns = defaultdict(lambda: {1: 0, 2: 0, "m": 0})
    recency_start = max(0, n - 400)

    for length in range(2, 9):
        for i in range(length - 1, n - 1 - max_g):
            seq = tuple(colors[i - length + 1:i + 1])
            d = patterns[seq]; d["m"] += 1
            bonus = 2 if i >= recency_start else 1
            for alvo in (1, 2):
                for step in range(1 + max_g):
                    idx = i + 1 + step
                    if idx < n and colors[idx] == alvo:
                        d[alvo] += bonus; break

    strats = []
    for seq, d in patterns.items():
        m = d["m"]
        if m < MINER_MIN_MATCHES: continue
        for alvo in (1, 2):
            w = d[alvo]
            try:
                wl = wilson_lower(w, m)
            except Exception as e:
                log.debug("Minerador: falha wilson_lower para m=%s, w=%s: %s", m, w, e)
                continue
            if wl >= MINER_MIN_BAYES:
                txt = "-".join(cs(c) for c in seq)
                strats.append({
                    "id": f"mined_{txt}_{alvo}",
                    "family": "exact_ngram_local",
                    "name": f"Local [{txt}]→{'V' if alvo==1 else 'P'}",
                    "seq": seq, "target": alvo,
                    "prob": wl, "matches": m, "wins": w,
                    "source": "miner",
                    "weight": wl * min(m / 40.0, 1.0),
                })

    strats.sort(key=lambda x: x["weight"], reverse=True)
    with _miner_lock:
        GLOBAL_MINED_STRATS = strats[:100]
    log.info("Minerador: %d padrões Wilson-válidos", len(strats[:100]))

def _compute_color_dist(colors):
    n = len(colors)
    if n == 0:
        return {0: 0.07, 1: 0.465, 2: 0.465}
    counts = defaultdict(int)
    for c in colors:
        counts[c] += 1
    return {k: round(counts[k] / n, 4) for k in (0, 1, 2)}


def _compute_transition_matrix(colors):
    matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(colors) - 1):
        matrix[colors[i]][colors[i + 1]] += 1
    result = {}
    for from_c, to_counts in matrix.items():
        total = sum(to_counts.values())
        result[from_c] = {to_c: round(cnt / total, 4) for to_c, cnt in to_counts.items()}
    return result


def _compute_ngram_cache(colors, max_len=6):
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
    magnitude = sum(
        abs(new_dist.get(k, 0) - old_dist.get(k, 0)) for k in (0, 1, 2)
    )
    return magnitude


def continual_learner_thread():
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

            if total_count - last_count >= CL_RELEARN_EVERY or last_count == 0:
                log.info("🧬 Continual Learner: relendo %d rounds do DB...", total_count)

                c.execute("SELECT color FROM results_raw ORDER BY id ASC")
                all_colors = [r[0] for r in c.fetchall()]

                if len(all_colors) < 100:
                    conn.close()
                    time.sleep(15)
                    continue

                dist_full = _compute_color_dist(all_colors)
                dist_2000 = _compute_color_dist(all_colors[-2000:])
                dist_500 = _compute_color_dist(all_colors[-500:])

                transition = _compute_transition_matrix(all_colors)
                ngram = _compute_ngram_cache(all_colors, max_len=6)

                with _cl_lock:
                    old_dist = CL_STATE["color_dist_500"].copy()

                drift_mag = _detect_concept_drift(old_dist, dist_500)
                drift_detected = drift_mag >= CL_DRIFT_THRESHOLD

                if drift_detected:
                    log.warning(
                        "⚠️ CONCEPT DRIFT DETECTADO! Magnitude=%.4f — Re-calibrando experts...",
                        drift_mag
                    )

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

def miner_thread():
    last_count = 0
    while True:
        try:
            conn = _conn(); c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM results_raw")
            n = c.fetchone()[0]
            c.execute("SELECT color FROM results_raw ORDER BY id ASC")
            colors = [r[0] for r in c.fetchall()]; conn.close()
            if n < 120: time.sleep(10); continue
            if n - last_count >= MINER_INTERVAL or last_count == 0:
                mine_local_strategies(colors); last_count = n
        except Exception as e:
            log.error("Minerador erro: %s", e)
        time.sleep(8)


# ═══════════════════════════════════════════════════════════════════════════════
#  9 EXPERTS INDEPENDENTES
# ═══════════════════════════════════════════════════════════════════════════════

def expert_miner(colors, regime):
    best = None
    with _miner_lock:
        for s in GLOBAL_MINED_STRATS:
            seq = s["seq"]
            if len(colors) >= len(seq) and tuple(colors[-len(seq):]) == seq:
                pw = get_pattern_weight(s["id"])
                score = s["weight"] * pw
                if best is None or score > best["score"]:
                    best = {**s, "score": score, "pw": pw}

    if best and best["prob"] >= CONFIDENCE_FLOOR - 0.05:
        regime_bonus = 1.15 if regime["name"] == "balanced" else 1.0
        return {
            "vote": best["target"],
            "confidence": min(best["prob"] * best["pw"] * regime_bonus, 0.97),
            "label": best["name"], "key": best["id"], "source": "miner",
        }
    return {"vote": None, "confidence": 0.0, "label": "miner:sem_match", "key": "", "source": "miner"}


def expert_catalog(colors, regime):
    best = None
    with _catalog_lock:
        for s in GLOBAL_CATALOG_STRATS:
            if catalog_matches(s, colors):
                pw = get_pattern_weight(s["id"])
                score = (s["wf_acc"] * 0.55 + s["weight"] * 0.45) * pw
                if best is None or score > best["score"]:
                    best = {**s, "score": score, "pw": pw}

    if best:
        return {
            "vote": best["target"],
            "confidence": min(best["score"], 0.97),
            "label": best["name"], "key": best["id"], "source": "catalog",
        }
    return {"vote": None, "confidence": 0.0, "label": "catalog:sem_match", "key": "", "source": "catalog"}


def expert_markov(colors, regime):
    nw = [c for c in colors if c != 0]
    if len(nw) < 8:
        return {"vote": None, "confidence": 0.0, "label": "markov:insuf", "key": "markov", "source": "markov"}

    p1r = markov_prob(nw, 1, 1); p1b = markov_prob(nw, 2, 1)
    p2r = markov_prob(nw, 1, 2); p2b = markov_prob(nw, 2, 2)
    p3r = markov_prob(nw, 1, 3); p3b = markov_prob(nw, 2, 3)

    prob_red   = p1r * 0.20 + p2r * 0.40 + p3r * 0.40
    prob_black = p1b * 0.20 + p2b * 0.40 + p3b * 0.40

    margin = abs(prob_red - prob_black)
    if margin < 0.08:
        return {"vote": None, "confidence": 0.0, "label": f"markov:margem_insuf({margin:.2f})", "key": "markov", "source": "markov"}

    vote = 1 if prob_red > prob_black else 2
    conf = max(prob_red, prob_black)

    bonus = 1.12 if regime["name"] in ("alternating", "streak_hot", "balanced") else 0.95
    return {
        "vote": vote,
        "confidence": min(conf * bonus, 0.97),
        "label": f"Markov3 {'V' if vote==1 else 'P'} {conf:.0%} (m={margin:.2f})",
        "key": "markov", "source": "markov",
    }


def expert_streak(colors, regime):
    nw = [c for c in colors if c != 0]
    if len(nw) < 8:
        return {"vote": None, "confidence": 0.0, "label": "streak:insuf", "key": "streak", "source": "streak"}

    strk = streak_info(nw)
    if strk["length"] < 3:
        return {"vote": None, "confidence": 0.0, "label": "streak:curto", "key": "streak", "source": "streak"}

    sc = strk["color"]; sl = strk["length"]; n = len(nw)
    reversals = continuations = 0
    max_g = get_max_gales()

    for i in range(1, n):
        run = 0; j = i - 1
        while j >= 0 and nw[j] == sc: run += 1; j -= 1
        if run == sl:
            for step in range(1 + max_g):
                idx2 = i + step
                if idx2 < n:
                    if nw[idx2] != sc: reversals += 1
                    else: continuations += 1
                    break

    total = reversals + continuations
    if total < 5:
        return {"vote": None, "confidence": 0.0, "label": "streak:dados_insuf", "key": "streak", "source": "streak"}

    rev_rate = bayes_prob(reversals, total)
    opponent = 2 if sc == 1 else 1

    if rev_rate >= 0.62:
        bonus = 1.18 if regime["name"] == "streak_hot" else 1.0
        return {
            "vote": opponent,
            "confidence": min(rev_rate * bonus, 0.97),
            "label": f"Reversão {sl}x{'V' if sc==1 else 'P'}→{'V' if opponent==1 else 'P'} ({rev_rate:.0%})",
            "key": f"streak_{sc}_{sl}_rev", "source": "streak",
        }
    if rev_rate <= 0.38:
        bonus = 1.08 if regime["name"] == "streak_hot" else 1.0
        cont_rate = 1.0 - rev_rate
        return {
            "vote": sc,
            "confidence": min(cont_rate * bonus, 0.97),
            "label": f"Continuação {sl}x ({cont_rate:.0%})",
            "key": f"streak_{sc}_{sl}_cont", "source": "streak",
        }
    return {"vote": None, "confidence": 0.0, "label": "streak:neutro", "key": "streak", "source": "streak"}


def expert_white_cycle(colors, regime):
    wh = poisson_white_hazard(colors)

    if wh["hazard"] >= 0.82:
        return {
            "vote": None, "confidence": 0.0,
            "label": f"⚠️ VETO hazard {wh['hazard']:.0%}",
            "key": "white_veto", "source": "white", "veto": True,
        }

    if wh["post_white"] or wh["dist"] <= 3:
        pw_counts = defaultdict(int)
        for i, c in enumerate(colors):
            if c == 0 and i + 1 < len(colors):
                nc = colors[i+1]
                if nc in (1, 2): pw_counts[nc] += 1

        total_pw = sum(pw_counts.values())
        if total_pw >= 12:
            pr = pw_counts[1] / total_pw
            pb = pw_counts[2] / total_pw
            if abs(pr - pb) < 0.04:
                return {"vote": None, "confidence": 0.0, "label": "white:pós_empate", "key": "post_white", "source": "white"}
            vote = 1 if pr > pb else 2
            conf = max(pr, pb)
            if conf >= 0.60:
                bonus = 1.08 if regime["name"] == "white_zone" else 1.0
                return {
                    "vote": vote,
                    "confidence": min(conf * bonus, 0.97),
                    "label": f"Pós-branco {'V' if vote==1 else 'P'} ({conf:.0%})",
                    "key": "post_white", "source": "white",
                }

    return {"vote": None, "confidence": 0.0, "label": "white:neutro", "key": "white", "source": "white"}


def expert_momentum(colors, regime):
    dev = distribution_deviation(colors, 50, 600)
    bias = dev["bias"]

    if bias is None:
        return {"vote": None, "confidence": 0.0, "label": "momentum:neutro", "key": "momentum", "source": "momentum"}

    red_dev_abs   = abs(dev["red_dev"])
    black_dev_abs = abs(dev["black_dev"])
    strength      = max(red_dev_abs, black_dev_abs)

    if strength < 0.07:
        return {"vote": None, "confidence": 0.0, "label": "momentum:desvio_fraco", "key": "momentum", "source": "momentum"}

    conf = min(0.50 + strength * 2.5, 0.82)

    if regime["name"] == "chaotic":
        return {"vote": None, "confidence": 0.0, "label": "momentum:caótico_bloqueado", "key": "momentum", "source": "momentum"}

    return {
        "vote": bias,
        "confidence": round(conf, 4),
        "label": f"Momentum retorno-à-média → {'V' if bias==1 else 'P'} (dev={strength:.2f})",
        "key": "momentum", "source": "momentum",
    }


def expert_alternation(colors, regime):
    nw = [c for c in colors if c != 0]
    if len(nw) < 10:
        return {"vote": None, "confidence": 0.0, "label": "alt:insuf", "key": "alternation", "source": "alternation"}

    alt6  = alternation_ratio(colors, 6)
    alt10 = alternation_ratio(colors, 10)
    alt_avg = (alt6 + alt10) / 2

    if alt_avg < 0.72:
        return {"vote": None, "confidence": 0.0, "label": f"alt:baixo({alt_avg:.2f})", "key": "alternation", "source": "alternation"}

    last = nw[-1]
    vote = 2 if last == 1 else 1

    conf = min(0.52 + alt_avg * 0.38, 0.88)

    if regime["name"] == "alternating":
        conf = min(conf * 1.12, 0.95)

    return {
        "vote": vote,
        "confidence": round(conf, 4),
        "label": f"Alternância {alt_avg:.0%} → {'V' if vote==1 else 'P'}",
        "key": "alternation", "source": "alternation",
    }


def expert_volatility(colors, regime):
    nw  = [c for c in colors if c != 0]
    vol = volatility_score(colors, 30)

    if vol["level"] == "explosion":
        return {"vote": None, "confidence": 0.0, "label": f"vol:explosão({vol['score']:.2f})", "key": "volatility", "source": "volatility"}

    if vol["level"] == "compression" and vol["score"] >= 0.65:
        if len(nw) < 4:
            return {"vote": None, "confidence": 0.0, "label": "vol:nw_insuf", "key": "volatility", "source": "volatility"}
        strk = streak_info(nw)
        if strk["length"] >= 2:
            conf = min(0.52 + vol["score"] * 0.30, 0.80)
            return {
                "vote": strk["color"],
                "confidence": round(conf, 4),
                "label": f"Vol compressão {vol['score']:.2f} → continua {'V' if strk['color']==1 else 'P'}",
                "key": "volatility", "source": "volatility",
            }

    return {"vote": None, "confidence": 0.0, "label": "vol:normal", "key": "volatility", "source": "volatility"}


def expert_antidrift(colors, regime):
    try:
        conn = _conn(); c = conn.cursor()
        c.execute("""
            SELECT correct FROM prediction_performance
            WHERE mode='leviathan_v2' AND action IN ('win','loss')
            ORDER BY id DESC LIMIT 20
        """)
        rows = c.fetchall(); conn.close()
        if len(rows) >= 10:
            recent_acc = sum(r[0] for r in rows) / len(rows)
            if recent_acc < 0.42:
                return {
                    "vote": None, "confidence": 0.0,
                    "label": f"🛡️ AntiDrift: acc={recent_acc:.0%} < 42% — VETO SISTÊMICO",
                    "key": "antidrift_veto", "source": "antidrift", "veto": True,
                }
            if recent_acc >= 0.60:
                return {"vote": None, "confidence": 0.0, "label": f"antidrift:quente({recent_acc:.0%})", "key": "antidrift", "source": "antidrift", "hot": True, "hot_bonus": 0.05}
    except: pass

    wh = poisson_white_hazard(colors)
    if wh["dist"] >= 22 and wh["hazard"] >= 0.85:
        return {
            "vote": None, "confidence": 0.0,
            "label": f"🛡️ AntiDrift: dist={wh['dist']} hazard={wh['hazard']:.0%} — PROVÁVEL BRANCA",
            "key": "antidrift_white", "source": "antidrift", "veto": True,
        }

    return {"vote": None, "confidence": 0.0, "label": "antidrift:ok", "key": "antidrift", "source": "antidrift"}


# ═══════════════════════════════════════════════════════════════════════════════
#  SOVEREIGN ENSEMBLE — ORQUESTRADOR DOS 9 EXPERTS
# ═══════════════════════════════════════════════════════════════════════════════
def run_ensemble(colors, regime, last_round=None):
    results = [
        expert_miner(colors, regime),
        expert_catalog(colors, regime),
        expert_markov(colors, regime),
        expert_streak(colors, regime),
        expert_white_cycle(colors, regime),
        expert_momentum(colors, regime),
        expert_alternation(colors, regime),
        expert_volatility(colors, regime),
        expert_antidrift(colors, regime),
        expert_sybil(colors, regime),
        expert_chaos(colors, regime),
        expert_hermes(colors, regime),
    ]

    for m in results:
        if m.get("veto"):
            return {
                "action": "block", "color": None, "confidence": 0.0,
                "votes": results, "reason": m["label"],
                "kelly": 0.0, "vote_count": 0,
            }

    hot_bonus = sum(m.get("hot_bonus", 0.0) for m in results)

    neural_weights_base  = get_neural_weights()
    oracle_weights       = oracle_get_weights(colors, regime)

    neural_weights = {
        k: round(neural_weights_base.get(k, 1.0) * oracle_weights.get(k, 1.0), 3)
        for k in set(list(neural_weights_base.keys()) + list(oracle_weights.keys()))
    }
    scores = {1: 0.0, 2: 0.0}
    vote_counts = {1: 0, 2: 0}
    active_votes = []

    for m in results:
        if m["vote"] is None: continue
        w = neural_weights.get(m["source"], 1.0)
        weighted = m["confidence"] * w
        scores[m["vote"]]      += weighted
        vote_counts[m["vote"]] += 1
        active_votes.append(m)

    if not active_votes:
        return {
            "action": "wait", "color": None, "confidence": 0.0,
            "votes": results, "reason": "⏳ Nenhum expert com sinal.",
            "kelly": 0.0, "vote_count": 0,
        }

    if scores[1] == scores[2]:
        return {
            "action": "wait", "color": None, "confidence": 0.0,
            "votes": results, "reason": "⏳ Empate perfeito entre experts.",
            "kelly": 0.0, "vote_count": 0,
        }

    winner = max(scores, key=scores.get)
    loser  = 2 if winner == 1 else 1

    votes_win  = vote_counts[winner]
    votes_lose = vote_counts[loser]

    win_weight_sum = sum(
        neural_weights.get(m["source"], 1.0)
        for m in active_votes if m["vote"] == winner
    )
    raw_conf = scores[winner] / max(win_weight_sum, 1.0)

    if votes_lose >= 2: raw_conf -= 0.10
    if votes_lose >= 3: raw_conf -= 0.08
    if votes_lose == 0 and votes_win >= 3: raw_conf += 0.08
    raw_conf += hot_bonus

    cal_conf = calibrate_confidence(raw_conf, votes_win, NUM_EXPERTS)
    cal_conf = round(max(0.0, min(cal_conf, 0.97)), 4)

    with _threshold_lock:
        threshold    = _threshold_state["value"]
        lockdown_until = _threshold_state["lockdown_until"]
        total_rounds = _threshold_state["total_rounds_seen"]
        banca_level  = _threshold_state["banca_level"]

    if total_rounds < lockdown_until:
        remaining = lockdown_until - total_rounds
        return {
            "action": "wait", "color": None, "confidence": cal_conf,
            "votes": results,
            "reason": f"🔒 LOCKDOWN de banca [{banca_level}] — {remaining} rounds restantes.",
            "kelly": 0.0, "vote_count": votes_win,
        }

    if cal_conf < threshold or votes_win < MIN_VOTES_TO_ENTER:
        parts = []
        if cal_conf < threshold: parts.append(f"edge {cal_conf:.0%} < thr {threshold:.0%}")
        if votes_win < MIN_VOTES_TO_ENTER: parts.append(f"experts {votes_win}/{MIN_VOTES_TO_ENTER}")
        top = [m["label"] for m in active_votes[:3]]
        return {
            "action": "wait", "color": None, "confidence": cal_conf,
            "votes": results,
            "reason": f"⏳ Aguardando: {'; '.join(parts)}. [{', '.join(top)}]",
            "kelly": 0.0, "vote_count": votes_win,
        }

    alvo_label   = "VERMELHO 🔴" if winner == 1 else "PRETO ⚫"
    contributing = [m["label"] for m in active_votes if m["vote"] == winner]
    kelly_val    = kelly_fraction(cal_conf)
    after_str    = f" [após {last_round['roll']} ({cs(last_round['color'])})]" if last_round else ""

    return {
        "action": "enter", "color": winner,
        "confidence": cal_conf,
        "votes": results,
        "reason": f"🎯 {alvo_label}{after_str} | Edge {cal_conf:.0%} | Experts {votes_win}/{NUM_EXPERTS} | [{'; '.join(contributing[:3])}]",
        "kelly": round(kelly_val * 100, 2),
        "vote_count": votes_win,
    }


# ─────────────────────────── GROQ ANALISTA ───────────────────────────────────
def ask_groq_analyst(colors, ensemble_result, regime):
    groq_key    = (get_sys_config("groq_key", "") or "").strip()
    llm_enabled = get_sys_config("llm_enabled", "0")
    if llm_enabled != "1" or not groq_key:
        return {"status": "disabled", "model": "", "reason": "LLM desativado"}

    seq_str  = "-".join(cs(c) for c in colors[-30:])
    alvo_str = "VERMELHO" if ensemble_result["color"] == 1 else "PRETO"

    with _threshold_lock:
        hist = list(_threshold_state["history"])[-20:]
    wr_str = f"{sum(hist)}/{len(hist)}" if hist else "sem histórico"

    prompt = f"""Você é um analista quantitativo de Blaze Double.
Responda APENAS: CONFIRMAR, REDUZIR ou VETAR.

• Últimas 30 cores: {seq_str}
• Regime: {regime['label']}
• Alvo: {alvo_str}
• Edge calibrado: {ensemble_result['confidence']:.1%}
• Experts concordantes: {ensemble_result['vote_count']}/{NUM_EXPERTS}
• Performance recente: {wr_str}
• Fundamento: {ensemble_result['reason'][:120]}

CONFIRMAR = contexto robusto
REDUZIR = sinal existe mas ambíguo (Kelly -50%)
VETAR = ruído, armadilha ou regime desfavorável"""

    for model in get_groq_models():
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0.05, "max_tokens": 8,
                      "messages": [
                          {"role": "system", "content": "Responda CONFIRMAR, REDUZIR ou VETAR."},
                          {"role": "user", "content": prompt},
                      ]},
                timeout=7,
            )
            if resp.status_code != 200: continue
            answer = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            if "VETAR"    in answer: return {"status": "vetar",       "model": model, "reason": answer}
            if "REDUZIR"  in answer: return {"status": "reduzir",     "model": model, "reason": answer}
            if "CONFIRMAR"in answer: return {"status": "confirmar",   "model": model, "reason": answer}
            return {"status": "inconclusivo", "model": model, "reason": answer}
        except Exception as e:
            log.debug("Groq %s falhou: %s", model, e)

    return {"status": "error", "model": "", "reason": "Groq indisponível"}


# ─────────────────────────── ADAPTIVE THRESHOLD + BANCA PROTECTION ───────────
def update_threshold(won):
    with _threshold_lock:
        _threshold_state["history"].append(won)
        _threshold_state["total_rounds_seen"] += 1
        level = _threshold_state["banca_level"]

        if won:
            _threshold_state["consecutive_losses"] = 0
            _threshold_state["consecutive_wins"]   += 1
            _threshold_state["recovery_wins"]      += 1

            if level in ("LOCKDOWN", "ALERT"):
                if _threshold_state["recovery_wins"] >= BANCA_RECOVERY_WINS:
                    _threshold_state["banca_level"]    = "NORMAL"
                    _threshold_state["lockdown_until"] = 0
                    _threshold_state["recovery_wins"]  = 0
                    log.info("✅ Banca recuperada — voltando ao modo NORMAL")

            new_t = max(THRESHOLD_MIN, _threshold_state["value"] - THRESHOLD_STEP_DOWN)

        else:
            _threshold_state["consecutive_wins"]   = 0
            _threshold_state["recovery_wins"]      = 0
            _threshold_state["consecutive_losses"] += 1
            losses = _threshold_state["consecutive_losses"]

            if losses >= BANCA_ALERT_LOSSES and level != "LOCKDOWN":
                mute_end = _threshold_state["total_rounds_seen"] + BANCA_LOCKDOWN_ROUNDS
                _threshold_state["lockdown_until"] = mute_end
                _threshold_state["banca_level"]    = "LOCKDOWN"
                log.warning("🔒 LOCKDOWN ATIVADO! %d derrotas. Silêncio de %d rounds.", losses, BANCA_LOCKDOWN_ROUNDS)

            elif losses >= BANCA_NORMAL_LOSSES and level == "NORMAL":
                mute_end = _threshold_state["total_rounds_seen"] + BANCA_ALERT_ROUNDS
                _threshold_state["lockdown_until"] = mute_end
                _threshold_state["banca_level"]    = "ALERT"
                log.warning("⚠️ BANCA ALERT! %d derrotas. Pausa de %d rounds.", losses, BANCA_ALERT_ROUNDS)

            new_t = min(THRESHOLD_MAX, _threshold_state["value"] + THRESHOLD_STEP_UP)

        _threshold_state["value"] = round(new_t, 4)

    save_meta("threshold",        str(_threshold_state["value"]))
    save_meta("consecutive_loss", str(_threshold_state["consecutive_losses"]))
    save_meta("consecutive_win",  str(_threshold_state["consecutive_wins"]))
    save_meta("lockdown_until",   str(_threshold_state["lockdown_until"]))
    save_meta("recovery_wins",    str(_threshold_state["recovery_wins"]))
    save_meta("banca_level",      _threshold_state["banca_level"])


def load_threshold_state():
    with _threshold_lock:
        _threshold_state["value"]               = float(load_meta("threshold",        str(THRESHOLD_START)))
        _threshold_state["consecutive_losses"]  = int(load_meta("consecutive_loss",   "0"))
        _threshold_state["consecutive_wins"]    = int(load_meta("consecutive_win",    "0"))
        _threshold_state["lockdown_until"]      = int(load_meta("lockdown_until",     "0"))
        _threshold_state["recovery_wins"]       = int(load_meta("recovery_wins",      "0"))
        _threshold_state["banca_level"]         = load_meta("banca_level",            "NORMAL")
    log.info("Estado da banca restaurado: level=%s threshold=%.2f%%",
             _threshold_state["banca_level"], _threshold_state["value"] * 100)


# ─────────────────────────── ENGINE PRINCIPAL ────────────────────────────────
def run_engine(seq):
    colors = [r["color"] for r in seq]
    wh     = poisson_white_hazard(colors)
    regime = detect_regime(colors)
    result = run_ensemble(colors, regime, seq[-1])

    groq_status = ""; groq_model = ""

    if result["action"] == "enter":
        groq = ask_groq_analyst(colors, result, regime)
        groq_status = groq["status"]; groq_model = groq["model"]

        if groq_status == "vetar":
            result["action"] = "block"
            result["confidence"] = min(result["confidence"] * 0.45, 0.44)
            result["reason"]     = f"🤖 VETO LLM ({groq_model}): {result['reason']}"
            result["kelly"]      = 0.0
        elif groq_status == "reduzir":
            result["kelly"]  = round(result["kelly"] * 0.50, 2)
            result["reason"] = f"⚡ KELLY -50% (LLM): {result['reason']}"
        elif groq_status == "confirmar":
            result["kelly"]  = round(min(result["kelly"] * 1.25, 5.0), 2)
            result["reason"] = f"✅ LLM OK: {result['reason']}"

    white_p = 0.07; c = result["color"]
    conf = max(0.0, min(result["confidence"], 1.0 - white_p))
    if   c == 1: probs = {"red": conf, "black": 1.0 - white_p - conf, "white": white_p}
    elif c == 2: probs = {"red": 1.0 - white_p - conf, "black": conf, "white": white_p}
    else:        probs = {"red": 0.465, "black": 0.465, "white": 0.07}

    votes_summary = [
        {"module": m["source"], "vote": m["vote"], "conf": round(m["confidence"], 3), "label": m["label"]}
        for m in result.get("votes", [])
    ]

    with _threshold_lock:
        banca_level = _threshold_state["banca_level"]

    neural_weights_base = get_neural_weights()
    oracle_w = oracle_get_weights(colors, regime)
    neural_weights = {
        k: round(neural_weights_base.get(k, 1.0) * oracle_w.get(k, 1.0), 3)
        for k in set(list(neural_weights_base.keys()) + list(oracle_w.keys()))
    }

    votes_all = result.get("votes", [])
    votes_ativos = [v for v in votes_all if v.get("vote") is not None]
    total_w = sum(neural_weights.get(v.get("source"), 1.0) * float(v.get("confidence", 0) or 0) for v in votes_ativos) or 1.0
    m_red = sum(neural_weights.get(v.get("source"), 1.0) * float(v.get("confidence", 0) or 0) for v in votes_ativos if v.get("vote") == 1) / total_w
    m_black = sum(neural_weights.get(v.get("source"), 1.0) * float(v.get("confidence", 0) or 0) for v in votes_ativos if v.get("vote") == 2) / total_w
    m_unc = round(max(0.0, 1.0 - m_red - m_black), 4)
    m_red = round(m_red, 4)
    m_black = round(m_black, 4)
    ds_conflict = round(1.0 - max(m_red, m_black, m_unc), 4)

    return {
        "n": len(seq),
        "signal": {
            "action": result["action"], "color": result["color"],
            "confidence": result["confidence"], "kelly": result["kelly"],
            "reason": result["reason"],
        },
        "probs": probs,
        "regime": {"regime": regime["name"], "label": regime["label"], "strength": regime["strength"]},
        "tests":  {"white": wh},
        "votes": result.get("votes", []),
        "oracle_weights": oracle_w,
        "oracle_q_states": len(_oracle_state.get("q_table", {})),
        "ds_conflict": ds_conflict,
        "ds_mass_red": m_red,
        "ds_mass_black": m_black,
        "ds_mass_unc": m_unc,
        "features": {
            "llm_status": groq_status, "llm_model": groq_model,
            "kelly_pct": result["kelly"], "vote_count": result.get("vote_count", 0),
            "ensemble_modules": NUM_EXPERTS,
            "threshold_used": _threshold_state["value"],
            "regime_name": regime["name"],
            "micro_regime": regime["name"],
            "miner_count": len(GLOBAL_MINED_STRATS),
            "catalog_count": len(GLOBAL_CATALOG_STRATS),
            "votes_json": json.dumps(votes_summary, ensure_ascii=False),
            "banca_level": banca_level,
            "ds_conflict": ds_conflict,
            "ds_mass_red": m_red,
            "ds_mass_black": m_black,
            "ds_mass_unc": m_unc,
            "oracle_weights": oracle_w,
            "oracle_q_states": len(_oracle_state.get("q_table", {})),
        },
    }


# ─────────────────────────── SALVAR SNAPSHOT ─────────────────────────────────
def save_snapshot(seq, engine_result):
    lid = seq[-1]["round_id"] if seq else None
    s   = engine_result["signal"]; p = engine_result["probs"]
    reg = engine_result["regime"]; wh = engine_result["tests"]["white"]
    feat= engine_result["features"]

    try:
        feat["micro_regime"]    = engine_result.get("micro_regime", feat.get("regime_name", feat.get("micro_regime", "")))
    except Exception:
        feat["micro_regime"] = feat.get("regime_name", "")
    feat["ds_conflict"]     = round(engine_result.get("ds_conflict", feat.get("ds_conflict", 0)), 4)
    feat["ds_mass_red"]     = round(engine_result.get("ds_mass_red", feat.get("ds_mass_red", 0)), 4)
    feat["ds_mass_black"]   = round(engine_result.get("ds_mass_black", feat.get("ds_mass_black", 0)), 4)
    feat["ds_mass_unc"]     = round(engine_result.get("ds_mass_unc", feat.get("ds_mass_unc", 0)), 4)
    feat["oracle_weights"]  = engine_result.get("oracle_weights", feat.get("oracle_weights", {}))
    feat["oracle_q_states"] = engine_result.get("oracle_q_states", feat.get("oracle_q_states", 0))
    feat["banca_level"]     = engine_result.get("banca_level", feat.get("banca_level", "NORMAL"))
    feat["vote_count"]      = engine_result.get("vote_count", feat.get("vote_count", 0))
    feat["votes_json"]      = json.dumps(
        [
            {
                "source": v.get("source", ""),
                "module": v.get("source", ""),
                "vote": v.get("vote"),
                "confidence": round(float(v.get("confidence", 0) or 0), 4),
                "label": v.get("label", ""),
            }
            for v in engine_result.get("votes", [])
        ],
        ensure_ascii=False
    )

    patterns = []
    if s["action"] in ("enter", "block"):
        patterns = [{"name": s["reason"][:80], "strength": s["confidence"]}]

    conn = _conn()
    cur  = conn.execute("""
        INSERT INTO analysis_snapshots
        (ts, total_rounds, last_round_id, prob_red, prob_black, prob_white,
         signal_color, signal_conf, signal_action, signal_reason,
         regime, regime_strength, white_hazard, dist_last_white,
         features_json, patterns_json, mode_used, votes_json, threshold_used, banca_level)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(), len(seq), lid,
        p["red"], p["black"], p["white"],
        s.get("color"), s.get("confidence"), s.get("action"), s.get("reason"),
        reg.get("label"), reg["strength"], wh["hazard"], wh["dist"],
        json.dumps(feat, ensure_ascii=False, default=str),
        json.dumps(patterns, ensure_ascii=False),
        "leviathan_v2", feat.get("votes_json", "[]"),
        feat.get("threshold_used", THRESHOLD_START),
        feat.get("banca_level", "NORMAL"),
    ))
    snap_id = cur.lastrowid
    conn.commit(); conn.close()
    return snap_id


# ─────────────────────────── VALIDAÇÃO DE PERFORMANCE ────────────────────────
def validate_previous(current_color, current_round_id):
    conn = _conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, last_round_id, signal_color, signal_action,
                   COALESCE(votes_json,'[]'), features_json
            FROM analysis_snapshots
            WHERE signal_action IN ('enter','gale_1','gale_2')
            ORDER BY id DESC LIMIT 1
        """)
        row = c.fetchone()
        if not row: return

        snap_id, pred_rid, pred_color, s_action, votes_json_str, _ = row

        c.execute("SELECT action FROM prediction_performance WHERE snapshot_id=?", (snap_id,))
        if c.fetchone(): return

        c.execute("SELECT id FROM results_raw WHERE round_id=?", (pred_rid,))
        pred_db = c.fetchone()
        c.execute("SELECT id FROM results_raw WHERE round_id=?", (current_round_id,))
        curr_db = c.fetchone()
        if not pred_db or not curr_db: return

        gap = curr_db[0] - pred_db[0]
        # ✅ FIX: gap aumentado de max(2,...) para max(15,...) para capturar
        # rounds intermediários (wait/block) entre sinal e resultado real
        if not (1 <= gap <= max(15, get_max_gales() + 2)): return

        if current_color == 0:
            correct = 1; action_res = "empate_branco"; won = True
        else:
            correct = 1 if current_color == pred_color else 0
            won     = bool(correct)
            if correct: action_res = "win"
            else:
                max_g = get_max_gales()
                gale_step = {"enter": 0, "gale_1": 1, "gale_2": 2}.get(s_action, 0)
                action_res = "gale_pending" if gale_step < max_g else "loss"

        c.execute("""
            INSERT INTO prediction_performance
            (snapshot_id, ts, predicted, actual, correct, action, mode, pattern_key)
            VALUES (?,?,?,?,?,?,?,?)
        """, (snap_id, datetime.now(timezone.utc).isoformat(), pred_color, current_color,
              correct, action_res, "leviathan_v2", ""))
        conn.commit()

        if action_res in ("win", "loss", "empate_branco"):
            update_threshold(won)

            try:
                votes = json.loads(votes_json_str or "[]")
                module_votes = {}
                for v in votes:
                    key = v.get("label", "")[:30]
                    if key: record_pattern_outcome(key, won)
                    src = v.get("module")
                    if src:
                        module_votes[src] = {"vote": v.get("vote"), "confidence": v.get("conf", 0.5)}

                seq = load_sequence(5000)
                if seq:
                    colors = [r["color"] for r in seq]
                    regime = detect_regime(colors)
                    update_neural_weights(
                        module_votes,
                        current_color if not won else pred_color,
                        won,
                        colors,
                        regime,
                    )
            except: pass

            with _threshold_lock:
                level = _threshold_state["banca_level"]
                thr   = _threshold_state["value"]

            log.info(
                "📊 %s | pred=%s real=%s | thr=%.2f%% | banca=%s",
                action_res.upper(), cn(pred_color), cn(current_color),
                thr * 100, level,
            )
    finally:
        conn.close()


# ─────────────────────────── GALE INTELIGENTE ────────────────────────────────
def check_pending_gale(seq):
    max_gales = get_max_gales()
    if max_gales == 0: return None

    conn = _conn()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, last_round_id, signal_color, signal_action
            FROM analysis_snapshots
            WHERE signal_action IN ('enter','gale_1','gale_2')
            ORDER BY id DESC LIMIT 1
        """)
        row = c.fetchone()
        if not row: return None

        snap_id, pred_rid, pred_color, action = row

        c.execute("SELECT action FROM prediction_performance WHERE snapshot_id=?", (snap_id,))
        perf = c.fetchone()
        if not perf or perf[0] in ("win", "empate_branco"): return None

        c.execute("SELECT id FROM results_raw WHERE round_id=?", (pred_rid,))
        db_row = c.fetchone()
        if not db_row: return None

        gap = seq[-1]["id"] - db_row[0]
        gale_step = {"enter": 0, "gale_1": 1, "gale_2": 2}.get(action, 0)

        if gap == 1 and gale_step < max_gales:
            colors = [r["color"] for r in seq]
            regime = detect_regime(colors)

            new_regime = regime["name"]
            if new_regime in ("chaotic", "white_zone"):
                log.warning("🛡️ GALE %d ABORTADO — regime mudou para %s", gale_step+1, new_regime)
                return None

            ensemble_check = run_ensemble(colors, regime, seq[-1])
            still_valid = (ensemble_check["action"] == "enter" and ensemble_check["color"] == pred_color)

            if still_valid:
                kelly_g = round(kelly_fraction(ensemble_check["confidence"]) * 2 * 100, 2)
                last    = seq[-1]
                return {
                    "action": f"gale_{gale_step+1}", "color": pred_color,
                    "confidence": ensemble_check["confidence"],
                    "kelly": kelly_g,
                    "reason": f"🔥 Gale {gale_step+1} CONFIRMADO [após {last['roll']} ({cs(last['color'])})] ({ensemble_check['confidence']:.0%})",
                }
            else:
                log.warning("🛡️ GALE %d ABORTADO — ensemble mudou de direção.", gale_step+1)
                return None
    finally:
        conn.close()
    return None


# ─────────────────────────── CICLO DE ANÁLISE ────────────────────────────────
_last_notified = None

def run_analysis_cycle():
    global _last_notified

    last_db = get_last_round_id()
    last_an = get_last_analyzed_round_id()
    if last_db is None or last_db == last_an: return False

    seq = load_sequence(5000)
    n   = len(seq)
    if n < MIN_HISTORY: return False

    with _threshold_lock:
        _threshold_state["total_rounds_seen"] = n

    validate_previous(seq[-1]["color"], seq[-1]["round_id"])

    gale_sig = check_pending_gale(seq)

    if gale_sig:
        colors = [r["color"] for r in seq]
        wh     = poisson_white_hazard(colors)
        regime = detect_regime(colors)
        with _threshold_lock: banca_level = _threshold_state["banca_level"]
        er = {
            "n": n,
            "signal": {"action": gale_sig["action"], "color": gale_sig["color"],
                       "confidence": gale_sig["confidence"], "kelly": gale_sig["kelly"],
                       "reason": gale_sig["reason"]},
            "probs": {"red": 0.5, "black": 0.5, "white": 0.0},
            "regime": {"regime": regime["name"], "label": regime["label"], "strength": regime["strength"]},
            "tests":  {"white": wh},
            "votes": [],
            "oracle_weights": {},
            "oracle_q_states": len(_oracle_state.get("q_table", {})),
            "ds_conflict": 0.0,
            "ds_mass_red": 0.0,
            "ds_mass_black": 0.0,
            "ds_mass_unc": 1.0,
            "features": {
                "llm_status": "", "llm_model": "", "kelly_pct": gale_sig["kelly"],
                "vote_count": 0, "ensemble_modules": NUM_EXPERTS,
                "threshold_used": _threshold_state["value"],
                "regime_name": regime["name"],
                "micro_regime": regime["name"],
                "miner_count": len(GLOBAL_MINED_STRATS),
                "catalog_count": len(GLOBAL_CATALOG_STRATS),
                "votes_json": "[]",
                "banca_level": banca_level,
                "oracle_q_states": len(_oracle_state.get("q_table", {})),
                "oracle_weights": {},
                "ds_conflict": 0.0,
                "ds_mass_red": 0.0,
                "ds_mass_black": 0.0,
                "ds_mass_unc": 1.0,
            },
        }
    else:
        er = run_engine(seq)

    save_snapshot(seq, er)
    s    = er["signal"]
    feat = er["features"]

    log.info("═" * 95)
    log.info(
        "LEVIATHAN v2.0 | Rounds=%d | Regime=%s | Sinal=%s | Cor=%s | Edge=%.1f%% | Kelly=%.2f%%",
        n, feat.get("regime_name","?").upper(), s["action"].upper(),
        cn(s["color"]), s["confidence"]*100, s["kelly"],
    )
    log.info(
        "Banca=%s | Threshold=%.2f%% | Experts=%d/%d | Miner=%d | Catálogo=%d | LLM=%s | Q-States=%d",
        feat.get("banca_level","?"), feat.get("threshold_used",0)*100,
        feat.get("vote_count",0), NUM_EXPERTS,
        feat.get("miner_count",0), feat.get("catalog_count",0),
        feat.get("llm_status","-"),
        feat.get("oracle_q_states", 0),
    )
    log.info("Fundamento: %s", s["reason"])
    log.info("═" * 95)

    if TELEGRAM_OK and s["action"] in ("enter", "gale_1", "gale_2"):
        uniq = f"{s['action']}|{s['color']}|{seq[-1]['round_id']}"
        if uniq != _last_notified:
                notificar_sinal(s, er["regime"], [], er["probs"], feat)
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
    oracle_load_state()

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