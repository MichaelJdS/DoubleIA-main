"""
=============================================================================
BLAZE DOUBLE AI — OTIMIZADOR DE ESTRATÉGIAS v1.2 (WILSON NEURAL + 100% DB)
Estuda TODO o banco SQLite (cobertura dinâmica de 100%), cria/reformula/exclui 
estratégias automaticamente e publica apenas as que sobrevivem ao 
walk-forward validation e ao crivo estatístico de Wilson.

Famílias mineradas:
  exact_ngram      → sequência exata N rodadas (N=2..7, com ou sem branco)
  run_edge         → quebra/continuação de streak (tamanho 3..8)
  white_gap        → contexto pela distância da última branca
  alternation_edge → regime de alternância controlada

Saídas:
  strategy_catalog      → catálogo vivo de estratégias
  strategy_backtests    → histórico de folds walk-forward
  optimizer_runs        → log de cada execução
=============================================================================
"""
import hashlib
import json
import logging
import math
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

DB_PATH = "blaze_double.db"
LOG_FILE = "otimizador_estrategias.log"

MIN_HISTORY = 450
MIN_TRAIN_MATCHES = 12
MIN_WF_MATCHES = 6
TRAIN_ACC_FLOOR = 0.58
WF_ACC_FLOOR = 0.56
RECENT_ACC_FLOOR = 0.52
DRIFT_THRESHOLD = 0.14
MAX_ACTIVE = 60
FOLDS = 5
WATCH_INTERVAL = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("optimizer")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def db_get(key, default=""):
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception:
        return default


def db_set(key, value):
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        conn.commit()
    finally:
        conn.close()


def get_max_gales():
    try:
        return int(db_get("max_gales", "0"))
    except Exception:
        return 0


def ensure_tables():
    conn = _conn()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy_catalog (
            strategy_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            name TEXT NOT NULL,
            params_json TEXT NOT NULL,
            target_color INTEGER NOT NULL,
            train_matches INTEGER DEFAULT 0,
            train_wins INTEGER DEFAULT 0,
            train_acc REAL DEFAULT 0,
            wf_matches INTEGER DEFAULT 0,
            wf_wins INTEGER DEFAULT 0,
            wf_acc REAL DEFAULT 0,
            recent_matches INTEGER DEFAULT 0,
            recent_wins INTEGER DEFAULT 0,
            recent_acc REAL DEFAULT 0,
            folds_seen INTEGER DEFAULT 0,
            support_score REAL DEFAULT 0,
            quality_score REAL DEFAULT 0,
            weight REAL DEFAULT 0,
            status TEXT DEFAULT 'inactive',
            activation_rank INTEGER,
            last_reason TEXT,
            last_seen_rounds INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            fold_idx INTEGER NOT NULL,
            train_end INTEGER NOT NULL,
            test_start INTEGER NOT NULL,
            test_end INTEGER NOT NULL,
            matches INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            acc REAL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS optimizer_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            total_rounds INTEGER NOT NULL,
            candidates_mined INTEGER NOT NULL,
            approved_count INTEGER NOT NULL,
            active_count INTEGER NOT NULL,
            max_gales INTEGER NOT NULL,
            notes TEXT
        );
        """)
        conn.commit()
    finally:
        conn.close()

    for k, v in {
        "auto_optimize": "1",
        "optimizer_every_rounds": "250",
        "optimizer_min_gap_minutes": "30",
        "optimizer_last_rounds": "0",
        "optimizer_last_ts": "",
    }.items():
        if db_get(k, "") == "":
            db_set(k, v)


def load_colors():
    conn = _conn()
    try:
        c = conn.cursor()
        # Carrega estritamente 100% das linhas inseridas no banco
        c.execute("SELECT color FROM results_raw ORDER BY id ASC")
        return [int(r[0]) for r in c.fetchall()]
    finally:
        conn.close()


def seq_to_text(seq):
    m = {0: "B", 1: "V", 2: "P"}
    return "-".join(m.get(x, "?") for x in seq)


def stable_id(family, params, target_color):
    raw = json.dumps(
        {"family": family, "params": params, "target_color": target_color},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def last_nonwhite_color(colors, idx):
    for j in range(idx, -1, -1):
        if colors[j] in (1, 2):
            return colors[j]
    return None


def nonwhite_tail(colors, idx, window):
    out = []
    for j in range(idx, -1, -1):
        if colors[j] in (1, 2):
            out.append(colors[j])
            if len(out) == window:
                break
    return list(reversed(out))


def flip_ratio_nonwhite(colors, idx, window):
    tail = nonwhite_tail(colors, idx, window)
    if len(tail) < window:
        return 0.0
    flips = sum(1 for i in range(1, len(tail)) if tail[i] != tail[i - 1])
    return flips / max(len(tail) - 1, 1)


def white_gap_before(colors, idx):
    gap = 0
    for j in range(idx, -1, -1):
        if colors[j] == 0:
            return gap
        gap += 1
    return None


def gap_bucket(gap):
    if gap is None:
        return None
    if gap <= 4:   return "0_4"
    if gap <= 9:   return "5_9"
    if gap <= 14:  return "10_14"
    if gap <= 22:  return "15_22"
    return "23_plus"


def nonwhite_run_ending_at(colors, idx):
    anchor = last_nonwhite_color(colors, idx)
    if anchor not in (1, 2):
        return {"color": None, "size": 0}
    size = 0
    for j in range(idx, -1, -1):
        c = colors[j]
        if c == 0:
            continue
        if c == anchor:
            size += 1
        else:
            break
    return {"color": anchor, "size": size}


def hit_target(colors, idx, target, max_gales):
    stop = min(len(colors), idx + 2 + max_gales)
    for j in range(idx + 1, stop):
        if colors[j] == target:
            return True
    return False


# ==============================================================================
# WILSON SCORE INTERVAL (O NÚCLEO ESTATÍSTICO)
# Protege contra pequenas amostras (Ex: 2/2 vitórias é menos confiável que 90/100)
# ==============================================================================
def wilson_score(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denominator = 1 + z**2/n
    centre_adjusted_prob = p + z**2 / (2*n)
    adjusted_standard_deviation = math.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    return (centre_adjusted_prob - z * adjusted_standard_deviation) / denominator


def make_candidate(family, name, params, target, matches, wins):
    acc_wilson = wilson_score(wins, matches)
    return {
        "strategy_id": stable_id(family, params, target),
        "family": family,
        "name": name,
        "params": params,
        "target_color": target,
        "train_matches": matches,
        "train_wins": wins,
        "train_acc": acc_wilson, 
    }


def mine_exact_ngrams(train, max_gales):
    upper = len(train) - 1 - max_gales
    counts = defaultdict(lambda: {"m": 0, 1: 0, 2: 0})
    for lag in range(2, 8):
        for i in range(lag - 1, upper):
            seq = tuple(train[i - lag + 1:i + 1])
            if lag >= 6 and seq.count(0) >= 2:
                continue
            row = counts[(lag, seq)]
            row["m"] += 1
            for t in (1, 2):
                if hit_target(train, i, t, max_gales):
                    row[t] += 1

    out = []
    for (lag, seq), row in counts.items():
        m = row["m"]
        if m < MIN_TRAIN_MATCHES:
            continue
        for t in (1, 2):
            acc_wilson = wilson_score(row[t], m)
            if acc_wilson >= TRAIN_ACC_FLOOR:
                params = {"seq": list(seq), "lag": lag}
                out.append(make_candidate(
                    "exact_ngram",
                    f"NGRAM [{seq_to_text(seq)}]→{'V' if t==1 else 'P'}",
                    params, t, m, row[t],
                ))
    return out


def mine_run_edges(train, max_gales):
    upper = len(train) - 1 - max_gales
    counts = defaultdict(lambda: {"m": 0, 1: 0, 2: 0})
    for i in range(upper):
        run = nonwhite_run_ending_at(train, i)
        if run["color"] not in (1, 2) or run["size"] < 3 or run["size"] > 8:
            continue
        key = (run["color"], run["size"])
        counts[key]["m"] += 1
        for t in (1, 2):
            if hit_target(train, i, t, max_gales):
                counts[key][t] += 1

    out = []
    for (rc, rs), row in counts.items():
        m = row["m"]
        if m < MIN_TRAIN_MATCHES:
            continue
        for t in (1, 2):
            acc_wilson = wilson_score(row[t], m)
            if acc_wilson >= TRAIN_ACC_FLOOR:
                mode = "cont" if t == rc else "quebra"
                name = f"RUN {rs}x {'V' if rc==1 else 'P'} {mode}→{'V' if t==1 else 'P'}"
                params = {"run_color": rc, "run_size": rs}
                out.append(make_candidate("run_edge", name, params, t, m, row[t]))
    return out


def mine_white_gap(train, max_gales):
    upper = len(train) - 1 - max_gales
    counts = defaultdict(lambda: {"m": 0, 1: 0, 2: 0})
    for i in range(upper):
        bucket = gap_bucket(white_gap_before(train, i))
        lastc = last_nonwhite_color(train, i)
        if bucket is None or lastc not in (1, 2):
            continue
        key = (bucket, lastc)
        counts[key]["m"] += 1
        for t in (1, 2):
            if hit_target(train, i, t, max_gales):
                counts[key][t] += 1

    out = []
    for (bucket, lastc), row in counts.items():
        m = row["m"]
        if m < MIN_TRAIN_MATCHES:
            continue
        for t in (1, 2):
            acc_wilson = wilson_score(row[t], m)
            if acc_wilson >= TRAIN_ACC_FLOOR:
                params = {"gap_bucket": bucket, "last_color": lastc}
                name = f"WGAP {bucket} após {'V' if lastc==1 else 'P'}→{'V' if t==1 else 'P'}"
                out.append(make_candidate("white_gap", name, params, t, m, row[t]))
    return out


def mine_alternation_edges(train, max_gales):
    upper = len(train) - 1 - max_gales
    counts = defaultdict(lambda: {"m": 0, 1: 0, 2: 0})
    for window in (6, 8, 10):
        for i in range(upper):
            fr = flip_ratio_nonwhite(train, i, window)
            if fr < 0.75:
                continue
            lastc = last_nonwhite_color(train, i)
            if lastc not in (1, 2):
                continue
            key = (window, lastc)
            counts[key]["m"] += 1
            for t in (1, 2):
                if hit_target(train, i, t, max_gales):
                    counts[key][t] += 1

    out = []
    for (window, lastc), row in counts.items():
        m = row["m"]
        if m < MIN_TRAIN_MATCHES:
            continue
        for t in (1, 2):
            acc_wilson = wilson_score(row[t], m)
            if acc_wilson >= TRAIN_ACC_FLOOR:
                params = {"window": window, "min_flip_ratio": 0.75, "last_color": lastc}
                name = f"ALT w{window} após {'V' if lastc==1 else 'P'}→{'V' if t==1 else 'P'}"
                out.append(make_candidate("alternation_edge", name, params, t, m, row[t]))
    return out


def mine_candidates(train, max_gales):
    merged = {}
    for c in (
        mine_exact_ngrams(train, max_gales)
        + mine_run_edges(train, max_gales)
        + mine_white_gap(train, max_gales)
        + mine_alternation_edges(train, max_gales)
    ):
        sid = c["strategy_id"]
        if sid not in merged or c["train_acc"] > merged[sid]["train_acc"]:
            merged[sid] = c
    return list(merged.values())


def strategy_matches(s, colors, idx):
    family = s["family"]
    p = s["params"]

    if family == "exact_ngram":
        seq = p.get("seq", [])
        lag = len(seq)
        if idx - lag + 1 < 0:
            return False
        return colors[idx - lag + 1:idx + 1] == seq

    if family == "run_edge":
        run = nonwhite_run_ending_at(colors, idx)
        return run["color"] == p.get("run_color") and run["size"] == p.get("run_size")

    if family == "white_gap":
        return (
            gap_bucket(white_gap_before(colors, idx)) == p.get("gap_bucket")
            and last_nonwhite_color(colors, idx) == p.get("last_color")
        )

    if family == "alternation_edge":
        lastc = last_nonwhite_color(colors, idx)
        fr = flip_ratio_nonwhite(colors, idx, int(p.get("window", 6)))
        return lastc == p.get("last_color") and fr >= float(p.get("min_flip_ratio", 0.75))

    return False


def evaluate(colors, strategy, start_idx, end_idx, max_gales):
    matches = wins = 0
    stop = min(end_idx, len(colors) - 1 - max_gales)
    for i in range(start_idx, stop):
        if strategy_matches(strategy, colors, i):
            matches += 1
            if hit_target(colors, i, strategy["target_color"], max_gales):
                wins += 1
    return matches, wins


# ==============================================================================
# LEITURA DE 100% DO BANCO DE DADOS DINAMICAMENTE
# ==============================================================================
def build_folds(n, num_folds=FOLDS):
    """
    Fatia o banco de dados dinamicamente, varrendo de 0 até o total N.
    O treino inicial começa usando os primeiros 30% dos dados.
    Os restantes 70% são divididos matematicamente para cobrir 100% do banco.
    """
    slices = []
    base_train_size = int(n * 0.30)
    
    # +1 no divisor garante que o último bloco de dados absoluto seja 
    # reservado exclusivamente para a validação "recente"
    test_block_size = (n - base_train_size) // (num_folds + 1) 

    for i in range(num_folds):
        train_end = base_train_size + (i * test_block_size)
        test_start = train_end
        test_end = test_start + test_block_size
        
        if train_end >= n - 20: 
            break
            
        slices.append((i + 1, train_end, test_start, test_end))
        
    return slices


def support_score(wf_matches):
    if wf_matches <= 0:
        return 0.0
    return min(1.0, math.log1p(wf_matches) / math.log(60))


def quality(train_acc, wf_acc, recent_acc, folds_seen, wf_matches):
    rec = recent_acc if recent_acc > 0 else wf_acc
    fold_c = min(1.0, folds_seen / max(FOLDS, 1))
    sup_c = support_score(wf_matches)
    return round(0.42 * wf_acc + 0.24 * rec + 0.18 * train_acc + 0.08 * fold_c + 0.08 * sup_c, 6)


def weight(q, wf_matches, recent_matches):
    sup = support_score(wf_matches)
    rb = 0.03 if recent_matches >= 6 else 0.0
    return round(min(q * (0.55 + 0.45 * sup) + rb, 0.99), 6)


def optimize_once():
    ensure_tables()
    colors = load_colors()
    n = len(colors)

    if n < MIN_HISTORY:
        log.warning("Histórico insuficiente: %d/%d rodadas", n, MIN_HISTORY)
        return

    max_gales = get_max_gales()
    run_ts = utc_now()
    folds = build_folds(n)

    if not folds:
        log.warning("Não foi possível montar folds walk-forward")
        return

    log.info("=" * 64)
    log.info(" OTIMIZADOR 100%% DB | rounds=%d | gales=%d | folds=%d", n, max_gales, len(folds))
    log.info("=" * 64)

    aggregate = {}
    backtests = []

    for fold_idx, train_end, test_start, test_end in folds:
        train = colors[:train_end]
        candidates = mine_candidates(train, max_gales)
        log.info("Fold %d | treino=%d | teste=[%d,%d) | candidatos=%d",
                 fold_idx, train_end, test_start, test_end, len(candidates))

        for cand in candidates:
            sid = cand["strategy_id"]
            if sid not in aggregate:
                aggregate[sid] = {
                    "strategy_id": sid,
                    "family": cand["family"],
                    "name": cand["name"],
                    "params": cand["params"],
                    "target_color": cand["target_color"],
                    "train_matches": 0,
                    "train_wins": 0,
                    "wf_matches": 0,
                    "wf_wins": 0,
                    "folds_seen": 0,
                }
            agg = aggregate[sid]
            agg["train_matches"] += cand["train_matches"]
            agg["train_wins"] += cand["train_wins"]

            m, w = evaluate(colors, cand, test_start, test_end, max_gales)
            if m > 0:
                agg["wf_matches"] += m
                agg["wf_wins"] += w
                agg["folds_seen"] += 1
                backtests.append({
                    "run_ts": run_ts,
                    "strategy_id": sid,
                    "fold_idx": fold_idx,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "matches": m,
                    "wins": w,
                    "acc": round(w / m, 6) if m else 0.0,
                })

    # Aqui o `recent_start` consome exatamente o que sobrou até o final do banco absoluto (100%)
    recent_start = folds[-1][3] if folds else int(n * 0.90)
    rows = []

    for sid, agg in aggregate.items():
        tm = agg["train_matches"]
        wm = agg["wf_matches"]
        if tm <= 0 or wm < MIN_WF_MATCHES:
            continue

        strat = {
            "strategy_id": sid,
            "family": agg["family"],
            "name": agg["name"],
            "params": agg["params"],
            "target_color": agg["target_color"],
        }

        # Lê estritamente o bloco final do banco
        rm, rw = evaluate(colors, strat, recent_start, n - 1, max_gales)
        
        train_acc = wilson_score(agg["train_wins"], tm)
        wf_acc = wilson_score(agg["wf_wins"], wm)
        recent_acc = wilson_score(rw, rm) if rm > 0 else 0.0

        q = quality(train_acc, wf_acc, recent_acc, agg["folds_seen"], wm)
        w = weight(q, wm, rm)
        ss = support_score(wm)

        active = True
        reasons = []
        if train_acc < TRAIN_ACC_FLOOR:
            active = False; reasons.append("train fraco (wilson)")
        if wf_acc < WF_ACC_FLOOR:
            active = False; reasons.append("walk-forward fraco (wilson)")
        if rm >= 6 and recent_acc < RECENT_ACC_FLOOR:
            active = False; reasons.append("recente degradado (wilson)")
        if rm >= 8 and recent_acc + DRIFT_THRESHOLD < wf_acc:
            active = False; reasons.append("drift estatístico detectado")
        if agg["folds_seen"] < 2:
            active = False; reasons.append("cobertura temporal insuficiente")

        rows.append({
            "strategy_id": sid,
            "family": agg["family"],
            "name": agg["name"],
            "params_json": json.dumps(agg["params"], ensure_ascii=False, sort_keys=True),
            "target_color": agg["target_color"],
            "train_matches": tm,
            "train_wins": agg["train_wins"],
            "train_acc": round(train_acc, 6),
            "wf_matches": wm,
            "wf_wins": agg["wf_wins"],
            "wf_acc": round(wf_acc, 6),
            "recent_matches": rm,
            "recent_wins": rw,
            "recent_acc": round(recent_acc, 6) if rm > 0 else 0.0,
            "folds_seen": agg["folds_seen"],
            "support_score": round(ss, 6),
            "quality_score": round(q, 6),
            "weight": round(w, 6),
            "_active": active,
            "last_reason": "ok" if active else "; ".join(reasons),
            "last_seen_rounds": n,
        })

    rows.sort(
        key=lambda x: (x["_active"], x["weight"], x["wf_acc"], x["wf_matches"]),
        reverse=True,
    )

    rank = 1
    active_count = approved_count = 0
    for row in rows:
        if row["_active"]:
            approved_count += 1
            if active_count < MAX_ACTIVE:
                row["status"] = "active"
                row["activation_rank"] = rank
                rank += 1
                active_count += 1
            else:
                row["status"] = "standby"
                row["activation_rank"] = None
                row["last_reason"] = "aprovado mas fora do top ativo"
        else:
            row["status"] = "inactive"
            row["activation_rank"] = None

    conn = _conn()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM strategy_backtests WHERE run_ts=?", (run_ts,))
        for bt in backtests:
            c.execute("""
            INSERT INTO strategy_backtests
            (run_ts,strategy_id,fold_idx,train_end,test_start,test_end,matches,wins,acc,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (bt["run_ts"], bt["strategy_id"], bt["fold_idx"], bt["train_end"],
                  bt["test_start"], bt["test_end"], bt["matches"], bt["wins"],
                  bt["acc"], run_ts))

        current_ids = set()
        for row in rows:
            sid = row["strategy_id"]
            current_ids.add(sid)
            c.execute("""
            INSERT INTO strategy_catalog (
                strategy_id, family, name, params_json, target_color,
                train_matches, train_wins, train_acc,
                wf_matches, wf_wins, wf_acc,
                recent_matches, recent_wins, recent_acc,
                folds_seen, support_score, quality_score, weight,
                status, activation_rank, last_reason, last_seen_rounds,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                family=excluded.family, name=excluded.name,
                params_json=excluded.params_json, target_color=excluded.target_color,
                train_matches=excluded.train_matches, train_wins=excluded.train_wins,
                train_acc=excluded.train_acc, wf_matches=excluded.wf_matches,
                wf_wins=excluded.wf_wins, wf_acc=excluded.wf_acc,
                recent_matches=excluded.recent_matches, recent_wins=excluded.recent_wins,
                recent_acc=excluded.recent_acc, folds_seen=excluded.folds_seen,
                support_score=excluded.support_score, quality_score=excluded.quality_score,
                weight=excluded.weight, status=excluded.status,
                activation_rank=excluded.activation_rank, last_reason=excluded.last_reason,
                last_seen_rounds=excluded.last_seen_rounds, updated_at=excluded.updated_at
            """, (
                sid, row["family"], row["name"], row["params_json"], row["target_color"],
                row["train_matches"], row["train_wins"], row["train_acc"],
                row["wf_matches"], row["wf_wins"], row["wf_acc"],
                row["recent_matches"], row["recent_wins"], row["recent_acc"],
                row["folds_seen"], row["support_score"], row["quality_score"], row["weight"],
                row["status"], row["activation_rank"], row["last_reason"], row["last_seen_rounds"],
                run_ts, run_ts,
            ))

        if current_ids:
            ph = ",".join("?" for _ in current_ids)
            c.execute(
                f"""UPDATE strategy_catalog
                SET status='inactive', activation_rank=NULL,
                    last_reason='não revalidada nesta rodada', updated_at=?
                WHERE strategy_id NOT IN ({ph})""",
                [run_ts, *current_ids],
            )

        c.execute("""
        INSERT INTO optimizer_runs
        (run_ts, total_rounds, candidates_mined, approved_count, active_count, max_gales, notes)
        VALUES(?,?,?,?,?,?,?)
        """, (run_ts, n, len(rows), approved_count, active_count, max_gales, f"folds={len(folds)}"))

        conn.commit()
    finally:
        conn.close()

    db_set("optimizer_last_rounds", n)
    db_set("optimizer_last_ts", run_ts)

    log.info("─" * 64)
    log.info("Concluído | catalogadas=%d | aprovadas=%d | ativas=%d | gales=%d",
             len(rows), approved_count, active_count, max_gales)

    active_rows = [r for r in rows if r["status"] == "active"]
    if active_rows:
        log.info("TOP 5 estratégias ativas:")
        for r in active_rows[:5]:
            log.info("  [%s] %-40s wf=%.2f weight=%.3f",
                     r["family"], r["name"][:40], r["wf_acc"], r["weight"])
    log.info("─" * 64)


def main():
    ensure_tables()
    args = sys.argv[1:]

    if "--watch" in args:
        log.info("Modo watch | intervalo=%ds", WATCH_INTERVAL)
        while True:
            try:
                optimize_once()
            except Exception as e:
                log.exception("Falha no ciclo: %s", e)
            time.sleep(WATCH_INTERVAL)
    else:
        optimize_once()


if __name__ == "__main__":
    main()