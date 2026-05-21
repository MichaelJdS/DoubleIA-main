"""=============================================================================
BLAZE DOUBLE AI — LEVIATHAN ENGINE v1.3 (NEURAL CORE UPGRADE)
Motor Adaptativo de Auto-Evolução com Ensemble Voting, Regime Detection,
Live Performance Memory, Catálogo Integrado, Adaptive Thresholds, Auto-Mute,
e agora com CAMADA NEURAL (Backpropagation em Tempo Real).
=============================================================================

ARQUITETURA:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    LEVIATHAN ENGINE v1.3                        │
  │                                                                 │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
  │  │  REGIME      │  │  ENSEMBLE    │  │  NEURAL META-LEARNER │  │
  │  │  DETECTOR    │→ │  VOTER       │→ │  (Pesos dinâmicos)   │  │
  │  │              │  │  (5 módulos) │  │  Retropropagação     │  │
  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
  │         ↓                ↓                      ↓               │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
  │  │  CATALOG     │  │  LIVE PERF   │  │  GROQ ANALISTA       │  │
  │  │  INTEGRATOR  │  │  MEMORY      │  │  (Context-Rich)      │  │
  │  │  (optimizer) │  │  (per padrão)│  │                      │  │
  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
  │                              ↓                                  │
  │                    ┌──────────────────┐                         │
  │                    │   AUTO-MUTE &    │                         │
  │                    │   KELLY SMART    │                         │
  │                    └──────────────────┘                         │
  └─────────────────────────────────────────────────────────────────┘
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

THRESHOLD_MIN = 0.60
THRESHOLD_START = 0.70
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

DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

# ─────────────────────────── ESTADO GLOBAL ───────────────────────────────────
_miner_lock = threading.Lock()
_catalog_lock = threading.Lock()

GLOBAL_MINED_STRATS = []
GLOBAL_CATALOG_STRATS = []

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


def _add_column_safe(conn, table: str, column: str, definition: str):
    """Migration segura: adiciona coluna se ainda não existir. Nunca falha."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()
        log.info("Migration: coluna '%s' adicionada em '%s'", column, table)
    except Exception:
        pass  # coluna já existe — ignora silenciosamente


def get_sys_config(key: str, default=None):
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except sqlite3.OperationalError:
        return default


def set_sys_config(key: str, value: str):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def get_mode() -> str:
    return get_sys_config("mode", "intelligent")


def get_max_gales() -> int:
    try:
        return int(get_sys_config("max_gales", "0"))
    except Exception:
        return 0


def get_groq_models() -> list:
    custom = os.environ.get("GROQ_MODEL", "").strip()
    if custom:
        return [x.strip() for x in custom.split(",") if x.strip()]
    return DEFAULT_GROQ_MODELS[:]


# ─────────────────────────── REDE NEURAL (META-LEARNER) ────────────────────────
# Os "neurônios" do sistema. Cada módulo possui um peso que é ajustado online
# usando o gradiente descendente a cada validação de round.
_neural_weights_lock = threading.Lock()
_NEURAL_LEARNING_RATE = 0.05
_DEFAULT_NEURAL_WEIGHTS = {
    "miner": 1.0,
    "catalog": 1.2,
    "markov": 0.8,
    "streak": 0.9,
    "white": 1.1
}

def get_neural_weights() -> dict:
    weights_json = get_sys_config("neural_weights_v1", "")
    if weights_json:
        try:
            return json.loads(weights_json)
        except Exception:
            pass
    return _DEFAULT_NEURAL_WEIGHTS.copy()

def update_neural_weights(module_votes: dict, actual_winner: int):
    with _neural_weights_lock:
        weights = get_neural_weights()
        for mod_name, vote_data in module_votes.items():
            if not vote_data.get("vote"):
                continue
            
            # Se o módulo votou no vencedor, o erro é 0 (positivo). Se não, erro é negativo.
            error = 1.0 if vote_data["vote"] == actual_winner else -1.0
            
            # Ajuste de Gradiente Simplificado
            adjustment = _NEURAL_LEARNING_RATE * error * vote_data["confidence"]
            weights[mod_name] = max(0.1, min(3.0, weights[mod_name] + adjustment))
            
        set_sys_config("neural_weights_v1", json.dumps(weights))
        log.info("🧠 Pesos Neurais Atualizados: %s", {k: round(v, 2) for k,v in weights.items()})


# ─────────────────────────── INICIALIZAÇÃO DAS TABELAS ───────────────────────
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
        mode_used       TEXT DEFAULT 'leviathan',
        votes_json      TEXT DEFAULT '[]',
        threshold_used  REAL DEFAULT 0.70
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
        mode        TEXT DEFAULT 'leviathan',
        pattern_key TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS leviathan_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    conn.commit()

    _add_column_safe(conn, "analysis_snapshots", "mode_used", "TEXT DEFAULT 'leviathan'")
    _add_column_safe(conn, "analysis_snapshots", "votes_json", "TEXT DEFAULT '[]'")
    _add_column_safe(conn, "analysis_snapshots", "threshold_used", "REAL DEFAULT 0.70")
    _add_column_safe(conn, "prediction_performance", "snapshot_id", "INTEGER")
    _add_column_safe(conn, "prediction_performance", "predicted", "INTEGER")
    _add_column_safe(conn, "prediction_performance", "confidence", "REAL")
    _add_column_safe(conn, "prediction_performance", "actual", "INTEGER")
    _add_column_safe(conn, "prediction_performance", "correct", "INTEGER")
    _add_column_safe(conn, "prediction_performance", "pattern_key", "TEXT DEFAULT ''")
    
    conn.close()

    for k, v in {
        "threshold": str(THRESHOLD_START),
        "consecutive_loss": "0",
        "consecutive_win": "0",
        "mute_until": "0",
        "mute_wins": "0",
        "total_signals": "0",
        "total_wins": "0",
    }.items():
        try:
            conn2 = _conn()
            conn2.execute(
                "INSERT OR IGNORE INTO leviathan_meta (key, value) VALUES (?, ?)",
                (k, v),
            )
            conn2.commit()
            conn2.close()
        except Exception:
            pass

    log.info("Tabelas Leviathan verificadas / migradas com sucesso.")


def load_meta(key: str, default="0") -> str:
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT value FROM leviathan_meta WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception:
        return default


def save_meta(key: str, value: str):
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


# ─────────────────────────── CARREGAR SEQUÊNCIA ──────────────────────────────
def load_sequence(limit=4000) -> list:
    conn = _conn()
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT id, round_id, color, roll, created_at
            FROM results_raw
            ORDER BY id DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = c.fetchall()
        return [
            {
                "id": r[0],
                "round_id": r[1],
                "color": r[2],
                "roll": r[3],
                "created_at": r[4],
            }
            for r in reversed(rows)
        ]
    finally:
        conn.close()


def get_last_round_id() -> Optional[str]:
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT round_id FROM results_raw ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_last_analyzed_round_id() -> Optional[str]:
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT last_round_id FROM analysis_snapshots ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# ─────────────────────────── HELPERS MATEMÁTICOS ─────────────────────────────
def color_short(c) -> str:
    return {0: "B", 1: "V", 2: "P"}.get(c, "?")


def color_name(c) -> str:
    return {0: "BRANCO", 1: "VERMELHO", 2: "PRETO"}.get(c, "—")


def bayes_prob(wins: int, matches: int, alpha=1.5) -> float:
    return (wins + alpha) / (matches + 2 * alpha)


def kelly_fraction(prob: float, frac: float = 0.25) -> float:
    if prob <= 0.50:
        return 0.0
    edge = 2.0 * prob - 1.0
    return max(0.0, min(edge * frac, 0.30))


def poisson_white_hazard(colors: list) -> dict:
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


def markov_prob(colors: list, target: int, order: int = 2) -> float:
    nw = [c for c in colors if c != 0]
    if len(nw) < order + 1:
        return 0.465
    key = tuple(nw[-(order):])
    counts = defaultdict(int)
    total = 0
    for i in range(len(nw) - order):
        k = tuple(nw[i : i + order])
        if k == key:
            counts[nw[i + order]] += 1
            total += 1
    if total == 0:
        return 0.465
    return (counts[target] + 1) / (total + 2)


def entropy_regime(colors: list, window: int = 20) -> float:
    recent = [c for c in colors[-window:] if c != 0]
    if len(recent) < 4:
        return 1.0
    counts = defaultdict(int)
    for c in recent:
        counts[c] += 1
    n = len(recent)
    ent = -sum((v / n) * math.log2(v / n + 1e-10) for v in counts.values())
    return round(min(ent / 1.0, 1.0), 4)


def streak_info(colors: list) -> dict:
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


def alternation_ratio(colors: list, window: int = 10) -> float:
    nw = [c for c in colors[-50:] if c != 0][-window:]
    if len(nw) < 2:
        return 0.5
    flips = sum(1 for i in range(1, len(nw)) if nw[i] != nw[i - 1])
    return flips / (len(nw) - 1)


# ─────────────────────────── DETECTOR DE REGIME ──────────────────────────────
def detect_regime(colors: list) -> dict:
    ent = entropy_regime(colors, 20)
    strk = streak_info(colors)
    alt = alternation_ratio(colors, 12)
    wh = poisson_white_hazard(colors)
    nw = [c for c in colors if c != 0]

    if wh["dist"] >= 18 and wh["hazard"] >= 0.72:
        return {
            "name": "white_zone",
            "label": "⚪ Zona Branca",
            "strength": wh["hazard"],
            "data": wh,
        }
    if strk["length"] >= 4:
        return {
            "name": "streak_hot",
            "label": f"🔥 Streak {strk['length']}x",
            "strength": min(strk["length"] / 8.0, 0.99),
            "data": strk,
        }
    if alt >= 0.78 and len(nw) >= 6:
        return {
            "name": "alternating",
            "label": "↔️ Alternância",
            "strength": round(alt, 3),
            "data": {"alt_ratio": alt},
        }
    if ent >= 0.88:
        return {
            "name": "chaotic",
            "label": "🌀 Caótico",
            "strength": round(ent, 3),
            "data": {"entropy": ent},
        }
    return {"name": "balanced", "label": "⚖️ Equilibrado", "strength": 0.5, "data": {}}


# ─────────────────────────── LIVE PERFORMANCE MEMORY ─────────────────────────
def record_pattern_outcome(pattern_key: str, won: bool):
    with _perf_lock:
        _pattern_perf_memory[pattern_key].append(won)


def get_pattern_weight(pattern_key: str) -> float:
    with _perf_lock:
        hist = list(_pattern_perf_memory.get(pattern_key, []))
    if len(hist) < 3:
        return 0.5
    recent = hist[-10:]
    acc = sum(recent) / len(recent)
    if acc < 0.40:
        return 0.1
    if acc < 0.50:
        return 0.3
    if acc < 0.60:
        return 0.55
    if acc < 0.70:
        return 0.75
    return round(min(acc, 0.99), 3)


# ─────────────────────────── CATÁLOGO DO OTIMIZADOR ──────────────────────────
def load_catalog_strategies() -> list:
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_catalog'"
        )
        if not c.fetchone():
            conn.close()
            return []
        c.execute("""
            SELECT strategy_id, family, name, params_json, target_color,
                   wf_acc, weight, quality_score, recent_acc
            FROM strategy_catalog
            WHERE status='active'
            ORDER BY weight DESC
            LIMIT 80
        """)
        rows = c.fetchall()
        conn.close()
        strats = []
        for r in rows:
            try:
                params = json.loads(r[3])
                strats.append(
                    {
                        "id": r[0],
                        "family": r[1],
                        "name": r[2],
                        "params": params,
                        "target": r[4],
                        "wf_acc": r[5] or 0.0,
                        "weight": r[6] or 0.0,
                        "quality": r[7] or 0.0,
                        "recent_acc": r[8] or 0.0,
                        "source": "catalog",
                    }
                )
            except Exception:
                pass
        return strats
    except Exception:
        return []


def catalog_strategy_matches(strat: dict, colors: list) -> bool:
    family = strat["family"]
    p = strat["params"]

    if family == "exact_ngram":
        seq = p.get("seq", [])
        lag = len(seq)
        if len(colors) - lag + 1 < 0:
            return False
        return colors[len(colors) - lag :] == seq

    if family == "run_edge":
        strk = streak_info(colors)
        return strk["color"] == p.get("run_color") and strk["length"] == p.get("run_size")

    if family == "white_gap":
        wh = poisson_white_hazard(colors)
        gap = wh["dist"]
        gb = (
            "0_4"
            if gap <= 4
            else (
                "5_9"
                if gap <= 9
                else "10_14" if gap <= 14 else "15_22" if gap <= 22 else "23_plus"
            )
        )
        nw = [c for c in colors if c != 0]
        last_nw = nw[-1] if nw else None
        return gb == p.get("gap_bucket") and last_nw == p.get("last_color")

    if family == "alternation_edge":
        nw = [c for c in colors if c != 0]
        last_nw = nw[-1] if nw else None
        fr = alternation_ratio(colors, int(p.get("window", 6)))
        return last_nw == p.get("last_color") and fr >= float(
            p.get("min_flip_ratio", 0.75)
        )

    return False


def refresh_catalog():
    global GLOBAL_CATALOG_STRATS
    while True:
        try:
            strats = load_catalog_strategies()
            with _catalog_lock:
                GLOBAL_CATALOG_STRATS = strats
            log.info("Catálogo atualizado: %d estratégias ativas", len(strats))
        except Exception as e:
            log.warning("Erro ao carregar catálogo: %s", e)
        time.sleep(60)


# ─────────────────────────── MINERADOR LOCAL ─────────────────────────────────
def mine_local_strategies(colors: list):
    global GLOBAL_MINED_STRATS
    n = len(colors)
    max_g = get_max_gales()
    patterns: dict = defaultdict(lambda: {1: 0, 2: 0, "m": 0})

    recency_bonus_start = max(0, n - 300)

    for length in range(2, 8):
        for i in range(length - 1, n - 1 - max_g):
            seq = tuple(colors[i - length + 1 : i + 1])
            d = patterns[seq]
            d["m"] += 1
            bonus = 2 if i >= recency_bonus_start else 1
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
                strats.append(
                    {
                        "id": f"mined_{txt}_{alvo}",
                        "family": "exact_ngram_local",
                        "name": f"Local [{txt}]→{'V' if alvo==1 else 'P'}",
                        "seq": seq,
                        "target": alvo,
                        "prob": b,
                        "matches": m,
                        "wins": w,
                        "source": "miner",
                        "wf_acc": 0.0,
                        "weight": b * min(m / 30.0, 1.0),
                    }
                )

    strats.sort(key=lambda x: x["weight"], reverse=True)
    with _miner_lock:
        GLOBAL_MINED_STRATS = strats[:80]
    log.info("Minerador local: %d padrões válidos", len(strats[:80]))


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

            if n < 100:
                time.sleep(10)
                continue

            if n - last_count >= MINER_INTERVAL or last_count == 0:
                mine_local_strategies(colors)
                last_count = n
        except Exception as e:
            log.error("Erro no minerador: %s", e)
        time.sleep(8)


# ─────────────────────────── MÓDULOS DO ENSEMBLE ─────────────────────────────
def module_local_miner(colors: list, regime: dict) -> dict:
    best = None
    with _miner_lock:
        for s in GLOBAL_MINED_STRATS:
            seq = s["seq"]
            if len(colors) >= len(seq) and tuple(colors[-len(seq) :]) == seq:
                pw = get_pattern_weight(s["id"])
                score = s["weight"] * pw
                if best is None or score > best["score"]:
                    best = {**s, "score": score, "pw": pw}

    if best and best["prob"] >= THRESHOLD_MIN:
        regime_bonus = 1.2 if regime["name"] == "balanced" else 1.0
        return {
            "vote": best["target"],
            "confidence": min(best["prob"] * best["pw"] * regime_bonus, 0.99),
            "label": best["name"],
            "key": best["id"],
            "source": "miner",
        }
    return {"vote": None, "confidence": 0.0, "label": "miner:sem_match", "key": "", "source": "miner"}


def module_catalog(colors: list, regime: dict) -> dict:
    best = None
    with _catalog_lock:
        for s in GLOBAL_CATALOG_STRATS:
            if catalog_strategy_matches(s, colors):
                pw = get_pattern_weight(s["id"])
                score = (s["wf_acc"] * 0.6 + s["weight"] * 0.4) * pw * 1.15
                if best is None or score > best["score"]:
                    best = {**s, "score": score, "pw": pw}

    if best:
        return {
            "vote": best["target"],
            "confidence": min(best["score"], 0.99),
            "label": best["name"],
            "key": best["id"],
            "source": "catalog",
        }
    return {"vote": None, "confidence": 0.0, "label": "catalog:sem_match", "key": "", "source": "catalog"}


def module_markov(colors: list, regime: dict) -> dict:
    nw = [c for c in colors if c != 0]
    if len(nw) < 5:
        return {"vote": None, "confidence": 0.0, "label": "markov:insuf", "key": "markov", "source": "markov"}

    p1r = markov_prob(nw, 1, order=1)
    p2r = markov_prob(nw, 1, order=2)
    p1b = markov_prob(nw, 2, order=1)
    p2b = markov_prob(nw, 2, order=2)

    prob_red = p1r * 0.4 + p2r * 0.6
    prob_black = p1b * 0.4 + p2b * 0.6
    margin = abs(prob_red - prob_black)

    if margin < 0.06:
        return {"vote": None, "confidence": 0.0, "label": "markov:indeciso", "key": "markov", "source": "markov"}

    if abs(prob_red - prob_black) < 0.02:
        return {"vote": None, "confidence": 0.0, "label": "markov:empate", "key": "markov", "source": "markov"}

    vote = 1 if prob_red > prob_black else 2
    conf = max(prob_red, prob_black)
    regime_bonus = 1.15 if regime["name"] in ("alternating", "streak_hot") else 1.0
    return {
        "vote": vote,
        "confidence": min(conf * regime_bonus, 0.99),
        "label": f"Markov ord2 {'V' if vote==1 else 'P'} {conf:.0%}",
        "key": "markov",
        "source": "markov",
    }


def module_streak_reversal(colors: list, regime: dict) -> dict:
    nw = [c for c in colors if c != 0]
    if len(nw) < 6:
        return {"vote": None, "confidence": 0.0, "label": "streak:insuf", "key": "streak", "source": "streak"}

    strk = streak_info(nw)
    if strk["length"] < 3:
        return {"vote": None, "confidence": 0.0, "label": "streak:curto", "key": "streak", "source": "streak"}

    sc = strk["color"]
    sl = strk["length"]
    n = len(nw)
    reversals = continuations = 0
    max_g = get_max_gales()

    for i in range(1, n):
        run = 0
        j = i - 1
        while j >= 0 and nw[j] == sc:
            run += 1
            j -= 1
        if run == sl:
            for step in range(1 + max_g):
                idx2 = i + step
                if idx2 < n:
                    if nw[idx2] != sc:
                        reversals += 1
                    else:
                        continuations += 1
                    break

    total = reversals + continuations
    if total < 4:
        return {"vote": None, "confidence": 0.0, "label": "streak:sem_dados", "key": "streak", "source": "streak"}

    reversal_rate = bayes_prob(reversals, total)
    opponent = 2 if sc == 1 else 1

    if reversal_rate >= 0.60:
        regime_bonus = 1.2 if regime["name"] == "streak_hot" else 1.0
        return {
            "vote": opponent,
            "confidence": min(reversal_rate * regime_bonus, 0.99),
            "label": f"Reversão streak {sl}x {'V' if sc==1 else 'P'} → {'V' if opponent==1 else 'P'} ({reversal_rate:.0%})",
            "key": f"streak_{sc}_{sl}",
            "source": "streak",
        }

    if reversal_rate <= 0.40:
        regime_bonus = 1.1 if regime["name"] == "streak_hot" else 1.0
        return {
            "vote": sc,
            "confidence": min((1.0 - reversal_rate) * regime_bonus, 0.99),
            "label": f"Continuação streak {sl}x ({(1-reversal_rate):.0%})",
            "key": f"streak_{sc}_{sl}_cont",
            "source": "streak",
        }

    return {"vote": None, "confidence": 0.0, "label": "streak:neutro", "key": "streak", "source": "streak"}


def module_white_cycle(colors: list, regime: dict) -> dict:
    wh = poisson_white_hazard(colors)

    if wh["hazard"] >= 0.80:
        return {
            "vote": None,
            "confidence": 0.0,
            "label": f"⚠️ Hazard branca {wh['hazard']:.0%} — VETO",
            "key": "white_hazard_veto",
            "source": "white",
            "veto": True,
        }

    if wh["post_white"] or wh["dist"] <= 3:
        post_white_counts = defaultdict(int)
        for i, c in enumerate(colors):
            if c == 0 and i + 1 < len(colors):
                next_c = colors[i + 1]
                if next_c in (1, 2):
                    post_white_counts[next_c] += 1

        total_pw = sum(post_white_counts.values())
        if total_pw >= 10:
            pr = post_white_counts[1] / total_pw
            pb = post_white_counts[2] / total_pw
            if abs(pr - pb) < 0.03:
                return {"vote": None, "confidence": 0.0, "label": "white:post_empate", "key": "post_white_cycle", "source": "white"}

            vote = 1 if pr > pb else 2
            conf = max(pr, pb)
            if conf >= 0.58:
                regime_bonus = 1.1 if regime["name"] == "white_zone" else 1.0
                return {
                    "vote": vote,
                    "confidence": min(conf * regime_bonus, 0.99),
                    "label": f"Pós-branco: {'V' if vote==1 else 'P'} ({conf:.0%})",
                    "key": "post_white_cycle",
                    "source": "white",
                }

    return {"vote": None, "confidence": 0.0, "label": "white:neutro", "key": "white", "source": "white"}


# ─────────────────────────── ENSEMBLE VOTER (COM REDE NEURAL) ─────────────────
def run_ensemble(colors: list, regime: dict, last_round: dict = None) -> dict:
    modules_results = [
        module_local_miner(colors, regime),
        module_catalog(colors, regime),
        module_markov(colors, regime),
        module_streak_reversal(colors, regime),
        module_white_cycle(colors, regime),
    ]

    for m in modules_results:
        if m.get("veto"):
            return {
                "action": "block",
                "color": None,
                "confidence": 0.0,
                "votes": modules_results,
                "reason": m["label"],
                "kelly": 0.0,
                "vote_count": 0,
            }

    scores = {1: 0.0, 2: 0.0}
    vote_counts = {1: 0, 2: 0}
    active_votes = []
    
    # Busca os pesos da camada neural (Meta-Learner)
    neural_weights = get_neural_weights()

    for m in modules_results:
        if m["vote"] is None:
            continue
        
        # Aplicação da Camada Sináptica
        weight = neural_weights.get(m["source"], 1.0)
        weighted_conf = m["confidence"] * weight
        
        scores[m["vote"]]      += weighted_conf
        vote_counts[m["vote"]] += 1
        active_votes.append(m)

    if not active_votes:
        return {
            "action": "wait", "color": None, "confidence": 0.0,
            "votes": modules_results, "reason": "⏳ Nenhum módulo com sinal. Aguardando convergência.",
            "kelly": 0.0, "vote_count": 0,
        }

    if scores[1] == scores[2]:
        return {
            "action": "wait", "color": None, "confidence": 0.0,
            "votes": modules_results, "reason": "⏳ Empate no ensemble — aguardando convergência.",
            "kelly": 0.0, "vote_count": max(vote_counts.values()),
        }

    winner = max(scores, key=scores.get)
    loser = 2 if winner == 1 else 1

    votes_for_winner = vote_counts[winner]
    votes_against = vote_counts[loser]
    
    total_weights = sum(neural_weights.get(m["source"], 1.0) for m in active_votes if m["vote"] == winner)

    # Sigmoid / Normalização pela soma dos pesos
    avg_conf = scores[winner] / max(total_weights, 1.0)
    
    consensus_bonus = 0.15 if votes_against == 0 and votes_for_winner > 1 else 0.0
    penalty = 0.12 if votes_against >= 2 else 0.0

    norm_conf = min(max(avg_conf + consensus_bonus - penalty, 0.0), 0.99)

    with _threshold_lock:
        threshold = _threshold_state["value"]
        mute_until = _threshold_state["mute_until_round"]
        total_rounds = _threshold_state["total_rounds_seen"]

    if total_rounds < mute_until:
        remaining = mute_until - total_rounds
        return {
            "action": "wait", "color": None, "confidence": norm_conf,
            "votes": modules_results, "reason": f"🔇 Auto-Mute ativo. Recuperando performance ({remaining} rounds).",
            "kelly": 0.0, "vote_count": votes_for_winner,
        }

    if norm_conf < threshold or votes_for_winner < MIN_VOTES_TO_ENTER:
        reason_parts = []
        if norm_conf < threshold:
            reason_parts.append(f"edge {norm_conf:.0%} < threshold {threshold:.0%}")
        if votes_for_winner < MIN_VOTES_TO_ENTER:
            reason_parts.append(f"votos {votes_for_winner}/{MIN_VOTES_TO_ENTER}")
        top_labels = [m["label"] for m in active_votes[:2]]
        return {
            "action": "wait", "color": None, "confidence": norm_conf,
            "votes": modules_results, "reason": f"⏳ Aguardando: {'; '.join(reason_parts)}. Sinais: {', '.join(top_labels)}",
            "kelly": 0.0, "vote_count": votes_for_winner,
        }

    alvo_label = "VERMELHO 🔴" if winner == 1 else "PRETO ⚫"
    contributing = [m["label"] for m in active_votes if m["vote"] == winner]
    kelly_val = kelly_fraction(norm_conf)

    after_str = f" [Após {last_round['roll']} ({color_short(last_round['color'])})]" if last_round else ""

    return {
        "action": "enter", "color": winner, "confidence": round(norm_conf, 4),
        "votes": modules_results,
        "reason": f"🎯 {alvo_label}{after_str} | Edge {norm_conf:.0%} | [{'; '.join(contributing[:2])}]",
        "kelly": round(kelly_val * 100, 2), "vote_count": votes_for_winner,
    }


# ─────────────────────────── GROQ ANALISTA RICH ──────────────────────────────
def ask_groq_analyst(colors: list, ensemble_result: dict, regime: dict) -> dict:
    groq_key = (get_sys_config("groq_key", "") or "").strip()
    llm_enabled = get_sys_config("llm_enabled", "0")

    if llm_enabled != "1" or not groq_key:
        return {"status": "disabled", "model": "", "reason": "LLM desativado"}

    seq_str = "-".join(color_short(c) for c in colors[-25:])
    alvo_str = "VERMELHO" if ensemble_result["color"] == 1 else "PRETO"
    regime_str = regime["label"]
    votes_str = f"{ensemble_result['vote_count']}/{ENSEMBLE_MODULES} módulos"
    edge_str = f"{ensemble_result['confidence']:.1%}"

    with _threshold_lock:
        hist = list(_threshold_state["history"])[-15:]
    wins_recent = sum(hist)
    total_recent = len(hist)
    wr_str = f"{wins_recent}/{total_recent}" if total_recent else "sem histórico"

    prompt = f"""Você é um analista quantitativo especialista em Blaze Double.
Analise este sinal e responda com APENAS UMA PALAVRA: CONFIRMAR, REDUZIR ou VETAR.

CONTEXTO DO SISTEMA:
• Últimas 25 cores: {seq_str}
• Regime atual: {regime_str}
• Alvo do sinal: {alvo_str}
• Edge do ensemble: {edge_str}
• Módulos concordantes: {votes_str}
• Performance recente: {wr_str} acertos

CRITÉRIO:
• CONFIRMAR: contexto é favorável, padrão é robusto
• REDUZIR: sinal existe mas contexto é ambíguo (entrada com Kelly reduzido)
• VETAR: padrão parece ruído, armadilha ou regime desfavorável"""

    for model in get_groq_models():
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model, "temperature": 0.05, "max_tokens": 6,
                    "messages": [
                        {"role": "system", "content": "Responda CONFIRMAR, REDUZIR ou VETAR."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=6,
            )
            if resp.status_code != 200: continue

            answer = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()

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
                log.info("🔊 Auto-Mute encerrado — performance recuperada")

            new_t = max(THRESHOLD_MIN, _threshold_state["value"] - THRESHOLD_STEP_DOWN)
        else:
            _threshold_state["consecutive_wins"] = 0
            _threshold_state["consecutive_losses"] += 1

            if _threshold_state["consecutive_losses"] >= AUTO_MUTE_LOSSES:
                mute_end = _threshold_state["total_rounds_seen"] + AUTO_MUTE_ROUNDS
                _threshold_state["mute_until_round"] = mute_end
                _threshold_state["mute_win_counter"] = 0
                log.warning(
                    "🔇 AUTO-MUTE ATIVADO — %d derrotas consecutivas. Silêncio por %d rounds.",
                    _threshold_state["consecutive_losses"], AUTO_MUTE_ROUNDS,
                )

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


# ─────────────────────────── ENGINE PRINCIPAL ────────────────────────────────
def run_engine(seq: list) -> dict:
    colors = [r["color"] for r in seq]
    n = len(colors)
    wh = poisson_white_hazard(colors)
    regime = detect_regime(colors)
    result = run_ensemble(colors, regime, seq[-1])

    groq_status = ""
    groq_model = ""

    if result["action"] == "enter":
        groq = ask_groq_analyst(colors, result, regime)
        groq_status = groq["status"]
        groq_model = groq["model"]

        if groq_status == "vetar":
            result["action"] = "block"
            result["confidence"] = min(result["confidence"] * 0.50, 0.45) 
            result["reason"] = f"🤖 VETO LLM ({groq_model}): {result['reason']}"
            result["kelly"] = 0.0

        elif groq_status == "reduzir":
            result["kelly"] = round(result["kelly"] * 0.50, 2)
            result["reason"] = f"⚡ KELLY REDUZIDO (LLM): {result['reason']}"

        elif groq_status == "confirmar":
            result["kelly"] = round(min(result["kelly"] * 1.30, 5.0), 2)
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

    votes_summary = [
        {"module": m["source"], "vote": m["vote"], "conf": round(m["confidence"], 3), "label": m["label"]}
        for m in result.get("votes", [])
    ]

    return {
        "n": n,
        "signal": {
            "action": result["action"], "color": result["color"],
            "confidence": result["confidence"], "kelly": result["kelly"],
            "reason": result["reason"],
        },
        "probs": probs,
        "regime": {"regime": regime["name"], "label": regime["label"], "strength": regime["strength"]},
        "tests": {"white": wh},
        "features": {
            "llm_status": groq_status, "llm_model": groq_model,
            "kelly_pct": result["kelly"], "vote_count": result.get("vote_count", 0),
            "ensemble_modules": ENSEMBLE_MODULES, "threshold_used": _threshold_state["value"],
            "regime_name": regime["name"], "miner_count": len(GLOBAL_MINED_STRATS),
            "catalog_count": len(GLOBAL_CATALOG_STRATS),
            "votes_json": json.dumps(votes_summary, ensure_ascii=False),
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

    patterns = []
    if s["action"] in ("enter", "block"):
        patterns = [{"name": s["reason"][:80], "strength": s["confidence"]}]

    conn = _conn()
    cur = conn.execute(
        """
        INSERT INTO analysis_snapshots
        (ts, total_rounds, last_round_id, prob_red, prob_black, prob_white,
         signal_color, signal_conf, signal_action, signal_reason,
         regime, regime_strength, white_hazard, dist_last_white,
         features_json, patterns_json, mode_used, votes_json, threshold_used)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            datetime.now(timezone.utc).isoformat(), len(seq), lid, p["red"], p["black"], p["white"],
            s.get("color"), s.get("confidence"), s.get("action"), s.get("reason"),
            reg.get("label"), reg["strength"], wh["hazard"], wh["dist"],
            json.dumps(feat, ensure_ascii=False, default=str), json.dumps(patterns, ensure_ascii=False),
            "leviathan", feat.get("votes_json", "[]"), feat.get("threshold_used", THRESHOLD_START),
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
            SELECT id, last_round_id, signal_color, signal_action,
                   COALESCE(votes_json, '[]'), features_json
            FROM analysis_snapshots
            WHERE signal_action IN ('enter', 'gale_1', 'gale_2')
            ORDER BY id DESC LIMIT 1
        """)
        row = c.fetchone()
        if not row: return

        snap_id, pred_last_rid, pred_color, s_action, votes_json_str, feat_json_str = row

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
            if correct: action_res = "win"
            else:
                max_g = get_max_gales()
                gale_step = {"enter": 0, "gale_1": 1, "gale_2": 2}.get(s_action, 0)
                action_res = "gale_pending" if gale_step < max_g else "loss"

        c.execute(
            """
            INSERT INTO prediction_performance
            (snapshot_id, ts, predicted, actual, correct, action, mode, pattern_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (snap_id, datetime.now(timezone.utc).isoformat(), pred_color, current_color,
             correct, action_res, "leviathan", ""),
        )
        conn.commit()

        if action_res in ("win", "loss", "empate_branco"):
            update_threshold(won)

            try:
                votes = json.loads(votes_json_str or "[]")
                # Converter JSON para dict nativo para a atualização neural
                module_votes = {}
                for v in votes:
                    key = v.get("label", "")[:30]
                    if key: record_pattern_outcome(key, won)
                    module_votes[v.get("module")] = {"vote": v.get("vote"), "confidence": v.get("conf")}
                
                # Executa o Backpropagation do Meta-Learner
                actual_winner = pred_color if won else (2 if pred_color == 1 else 1)
                update_neural_weights(module_votes, actual_winner)

            except Exception:
                pass

            log.info(
                "📊 Performance: %s | predicted=%s actual=%s | threshold=%.2f%%",
                action_res.upper(), color_name(pred_color), color_name(current_color),
                _threshold_state["value"] * 100,
            )

    finally:
        conn.close()


# ─────────────────────────── GALE INTELIGENTE (COM ABORTO NEURAL) ─────────────
def check_pending_gale(seq: list) -> Optional[dict]:
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
            ensemble_check = run_ensemble(colors, regime, seq[-1])
            
            # BLOQUEIO DE CUSTO IRRECUPERÁVEL (Sunk Cost Fallacy Block)
            still_valid = (ensemble_check["action"] == "enter" and ensemble_check["color"] == pred_color)

            next_gale = gale_step + 1
            last_round = seq[-1]
            after_gale_str = f" [Após {last_round['roll']} ({color_short(last_round['color'])})]"

            if still_valid:
                kelly_g = round(kelly_fraction(ensemble_check["confidence"]) * 2 * 100, 2)
                conf_g = ensemble_check["confidence"]
                label = f"🔥 Gale {next_gale} CONFIRMADO{after_gale_str} pelo Ensemble ({conf_g:.0%})"
                
                return {
                    "action": f"gale_{next_gale}", "color": pred_color, "confidence": conf_g,
                    "kelly": kelly_g, "reason": label,
                }
            else:
                # SE O MOTOR NEURAL DISCORDAR AGORA, ABORTA O GALE PARA PROTEGER BANCA
                log.warning(f"🛡️ GALE {next_gale} ABORTADO! A Rede Neural detectou mudança estrutural.")
                return None
    finally:
        conn.close()
    return None


# ─────────────────────────── CICLO DE ANÁLISE ────────────────────────────────
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
        er = {
            "n": n,
            "signal": {
                "action": gale_sig["action"], "color": gale_sig["color"],
                "confidence": gale_sig["confidence"], "kelly": gale_sig["kelly"],
                "reason": gale_sig["reason"],
            },
            "probs": {"red": 0.5, "black": 0.5, "white": 0.0},
            "regime": {"regime": regime["name"], "label": regime["label"], "strength": regime["strength"]},
            "tests": {"white": wh},
            "features": {
                "llm_status": "", "llm_model": "", "kelly_pct": gale_sig["kelly"], "vote_count": 0,
                "ensemble_modules": ENSEMBLE_MODULES, "threshold_used": _threshold_state["value"],
                "regime_name": regime["name"], "miner_count": len(GLOBAL_MINED_STRATS),
                "catalog_count": len(GLOBAL_CATALOG_STRATS), "votes_json": "[]",
            },
        }
    else:
        er = run_engine(seq)

    snap_id = save_snapshot(seq, er)
    s = er["signal"]
    feat = er["features"]

    log.info("═" * 90)
    log.info(
        "LEVIATHAN | Rounds=%d | Regime=%s | Sinal=%s | Cor=%s | Edge=%.1f%% | Kelly=%.2f%%",
        n, feat.get("regime_name", "?").upper(), s["action"].upper(), color_name(s["color"]),
        s["confidence"] * 100, s["kelly"],
    )
    log.info(
        "Threshold=%.2f%% | Votos=%d/%d | Miner=%d | Catálogo=%d | LLM=%s",
        feat.get("threshold_used", 0) * 100, feat.get("vote_count", 0), ENSEMBLE_MODULES,
        feat.get("miner_count", 0), feat.get("catalog_count", 0), feat.get("llm_status", "-"),
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
    log.info("  BLAZE DOUBLE AI — LEVIATHAN ENGINE v1.3 (NEURAL CORE)")
    log.info("  Ensemble Voting | Backpropagation | Adaptive Threshold | Live Memory")
    log.info("  Iniciado: %s", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    log.info("═" * 90)

    init_tables() 
    load_threshold_state() 

    threading.Thread(target=miner_thread, daemon=True, name="Miner").start()
    threading.Thread(target=refresh_catalog, daemon=True, name="Catalog").start()

    log.info("Threads iniciadas: Minerador + Catálogo + Loop Principal")

    try:
        while True:
            try:
                run_analysis_cycle()
            except Exception as e:
                log.error("Erro no ciclo: %s", e, exc_info=True)
            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        log.info("Leviathan encerrado.")

if __name__ == "__main__":
    main()