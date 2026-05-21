"""
=============================================================================
BLAZE DOUBLE AI — COLETOR v4.0 (Leviathan Backend)
WebSocket nativo + REST fallback + HTTP API + SSE
Compatível com Leviathan Engine v1.0
Novos endpoints: /leviathan_meta, /votes, threshold_used em /analysis
=============================================================================
"""

import asyncio
import json
import logging
import os
import queue
import signal as sig_mod
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer as _HTTPServer
from socketserver import ThreadingMixIn

import requests

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False

class ThreadingHTTPServer(ThreadingMixIn, _HTTPServer):
    daemon_threads = True

def load_jwt() -> str:
    if os.path.exists("blaze_token.json"):
        try:
            with open("blaze_token.json", "r", encoding="utf-8") as f:
                return json.load(f).get("jwt", "")
        except Exception:
            pass
    return os.environ.get("BLAZE_JWT", "")

DB_PATH   = "blaze_double.db"
LOG_FILE  = "coletor.log"
HTTP_PORT = 8765
WS_URL    = "wss://api-gaming.blaze.bet.br/replication/?EIO=3&transport=websocket"
POLL_SEC  = 15

RECENT_URLS = [
    "https://api-gaming.blaze.bet.br/api/roulette_games/recent",
    "https://blaze.bet.br/api/roulette_games/recent",
]

REST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
    "Referer":    "https://blaze.bet.br/pt/games/double",
    "Origin":     "https://blaze.bet.br",
    "Cache-Control": "no-cache",
    "Pragma":     "no-cache",
}

running    = True
ws_status  = {"connected": False, "last_event": None, "total_ws": 0}
_sse_clients = []
_sse_lock    = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("coletor")
_db_lock = threading.RLock()

def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c

def _sse_push(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            if q in _sse_clients:
                _sse_clients.remove(q)

def init_db():
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS results_raw (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id     TEXT UNIQUE,
        color        INTEGER NOT NULL,
        roll         INTEGER NOT NULL,
        created_at   TEXT NOT NULL,
        collected_at TEXT NOT NULL,
        source       TEXT DEFAULT 'ws'
    );
    CREATE TABLE IF NOT EXISTS system_config (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    for k, v in [
        ("mode",        "intelligent"),
        ("max_gales",   "0"),
        ("groq_key",    ""),
        ("llm_enabled", "0"),
    ]:
        conn.execute("INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()
    log.info("Banco inicializado: %s", DB_PATH)

def insert_result(round_id, color, roll, created_at, source="ws"):
    with _db_lock:
        conn = _conn()
        try:
            conn.execute("""
                INSERT INTO results_raw (round_id, color, roll, created_at, collected_at, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (round_id, color, roll, created_at,
                  datetime.now(timezone.utc).isoformat(), source))
            conn.commit()
            _sse_push("new_result", {"round_id": round_id, "color": color,
                                     "roll": roll, "source": source})
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

def get_all_ids():
    with _db_lock:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT round_id FROM results_raw")
        ids = {row[0] for row in c.fetchall()}
        conn.close()
        return ids

def get_total():
    with _db_lock:
        conn = _conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM results_raw")
        n = c.fetchone()[0]
        conn.close()
        return n

def get_mode():
    with _db_lock:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key='mode'")
        row = c.fetchone(); conn.close()
        return row[0] if row else "intelligent"

def set_mode(mode: str):
    if mode not in ("intelligent", "standard"): return False
    with _db_lock:
        conn = _conn()
        conn.execute("INSERT OR REPLACE INTO system_config (key,value) VALUES ('mode',?)", (mode,))
        conn.commit(); conn.close()
    return True

def get_gales():
    with _db_lock:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key='max_gales'")
        row = c.fetchone(); conn.close()
        try: return int(row[0]) if row else 0
        except: return 0

def set_gales(gales: int):
    if gales not in (0, 1, 2): return False
    with _db_lock:
        conn = _conn()
        conn.execute("INSERT OR REPLACE INTO system_config (key,value) VALUES ('max_gales',?)", (str(gales),))
        conn.commit(); conn.close()
    return True

def get_groq_config():
    with _db_lock:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key='groq_key'"); k = c.fetchone()
        c.execute("SELECT value FROM system_config WHERE key='llm_enabled'"); e = c.fetchone()
        conn.close()
        return {"key": k[0] if k else "", "enabled": bool(e and e[0] == "1")}

def set_groq_config(key: str, enabled: bool):
    with _db_lock:
        conn = _conn()
        conn.execute("INSERT OR REPLACE INTO system_config (key,value) VALUES ('groq_key',?)", (key,))
        conn.execute("INSERT OR REPLACE INTO system_config (key,value) VALUES ('llm_enabled',?)",
                     ("1" if enabled else "0",))
        conn.commit(); conn.close()
    return True

def reset_stats():
    with _db_lock:
        ts = datetime.now(timezone.utc).isoformat()
        conn = _conn()
        conn.execute("INSERT OR REPLACE INTO system_config (key,value) VALUES ('stats_reset_at',?)", (ts,))
        conn.commit(); conn.close()
        return ts

def get_scoreboard():
    with _db_lock:
        conn = _conn()
        try:
            c = conn.cursor()
            scores = {
                "leviathan": {"wins": 0, "losses": 0, "ties": 0},
                "intelligent": {"wins": 0, "losses": 0, "ties": 0},
                "standard":    {"wins": 0, "losses": 0, "ties": 0},
            }
            c.execute("SELECT value FROM system_config WHERE key='stats_reset_at'")
            row = c.fetchone(); reset_ts = row[0] if row else None

            q = """
                SELECT mode, action, COUNT(*)
                FROM prediction_performance
                {} GROUP BY mode, action
            """
            if reset_ts:
                c.execute(q.format("WHERE ts > ?"), (reset_ts,))
            else:
                c.execute(q.format(""))

            for mode, action, count in c.fetchall():
                m = mode if mode in scores else "leviathan"
                if action == "win":          scores[m]["wins"]   = count
                elif action == "loss":       scores[m]["losses"] = count
                elif action == "empate_branco": scores[m]["ties"] = count

            for m in scores:
                total = scores[m]["wins"] + scores[m]["losses"]
                scores[m]["win_rate"] = round(scores[m]["wins"] / total * 100, 1) if total else 0.0
            return scores
        except Exception:
            return {"leviathan": {"wins":0,"losses":0,"ties":0,"win_rate":0.0}}
        finally:
            conn.close()

def get_leviathan_meta():
    with _db_lock:
        conn = _conn()
        try:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='leviathan_meta'")
            if not c.fetchone():
                return {}
            c.execute("SELECT key, value FROM leviathan_meta")
            return {row[0]: row[1] for row in c.fetchall()}
        except Exception:
            return {}
        finally:
            conn.close()

def get_analysis():
    with _db_lock:
        try:
            conn = _conn(); c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_snapshots'")
            if not c.fetchone():
                conn.close()
                return {"status": "no_analysis", "message": "Analisador nao rodou ainda"}

            c.execute("""
                SELECT ts, total_rounds, last_round_id,
                       prob_red, prob_black, prob_white,
                       signal_color, signal_conf, signal_action, signal_reason,
                       regime, regime_strength,
                       white_hazard, dist_last_white,
                       features_json, patterns_json,
                       votes_json, threshold_used
                FROM analysis_snapshots
                ORDER BY id DESC LIMIT 1
            """)
            row = c.fetchone(); conn.close()
            if not row:
                return {"status": "no_analysis", "message": "Sem analises ainda"}

            try: patterns = json.loads(row[15]) if row[15] else []
            except: patterns = []
            try: votes = json.loads(row[16]) if row[16] else []
            except: votes = []
            try: feat = json.loads(row[14]) if row[14] else {}
            except: feat = {}

            # ── Extrai campos Pantheon do features_json ────────────────────
            micro_regime  = feat.get("micro_regime", feat.get("regime_name", ""))
            ds_conflict   = float(feat.get("ds_conflict", 0))
            ds_mass_red   = float(feat.get("ds_mass_red", 0))
            ds_mass_black = float(feat.get("ds_mass_black", 0))
            ds_mass_unc   = float(feat.get("ds_mass_unc", 0))
            oracle_w      = feat.get("oracle_weights", {})
            oracle_qs     = feat.get("oracle_q_states", 0)
            banca_level   = feat.get("banca_level", "NORMAL")
            vote_count    = feat.get("vote_count", 0)

            return {
                "status":       "ok",
                "ts":           row[0],
                "total_rounds": row[1],
                "last_round_id": row[2],
                "probabilities": {
                    "red":   round(row[3] or 0, 4),
                    "black": round(row[4] or 0, 4),
                    "white": round(row[5] or 0, 4),
                },
                "signal": {
                    "color":          row[6],
                    "confidence":     round(row[7] or 0, 4),
                    "action":         row[8] or "wait",
                    "reason":         row[9] or "",
                    "votes_json":     row[16] or "[]",
                    "threshold_used": round(float(row[17] or 0.74), 4),
                    "kelly":          float(feat.get("kelly_pct", 0) or 0),
                    "vote_count":     vote_count,
                },
                "regime": {
                    "name":         row[10] or "balanced",
                    "label":        row[10] or "balanced",
                    "strength":     round(row[11] or 0, 4),
                    "micro_regime": micro_regime,
                },
                "white_hazard":    round(row[12] or 0, 4),
                "dist_last_white": row[13] or 0,
                "features_json":   row[14] or "{}",
                "features": {
                    **feat,
                    # ── Campos Pantheon explícitos para o dashboard ──────
                    "micro_regime":   micro_regime,
                    "ds_conflict":    ds_conflict,
                    "ds_mass_red":    ds_mass_red,
                    "ds_mass_black":  ds_mass_black,
                    "ds_mass_unc":    ds_mass_unc,
                    "oracle_weights": oracle_w,
                    "oracle_q_states":oracle_qs,
                    "banca_level":    banca_level,
                    "votes_json":     row[16] or "[]",
                },
                "patterns": patterns,
                "votes":    votes,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

def get_signal_history(limit=30):
    with _db_lock:
        try:
            conn = _conn(); c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_snapshots'")
            if not c.fetchone():
                conn.close(); return []
            c.execute("""
                SELECT ts, total_rounds, prob_red, prob_black, prob_white,
                       signal_color, signal_conf, signal_action, signal_reason,
                       regime, regime_strength, threshold_used, votes_json
                FROM analysis_snapshots
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = c.fetchall(); conn.close()
            return [{
                "ts":           r[0], "rounds":    r[1],
                "prob_red":     round(r[2] or 0, 3),
                "prob_black":   round(r[3] or 0, 3),
                "prob_white":   round(r[4] or 0, 3),
                "signal_color": r[5],
                "confidence":   round(r[6] or 0, 3),
                "action":       r[7] or "wait",
                "reason":       r[8] or "",
                "regime":       r[9] or "balanced",
                "regime_str":   round(r[10] or 0, 3),
                "threshold":    round(float(r[11] or 0.70), 3),
                "votes_json":   r[12] or "[]",
            } for r in rows]
        except Exception:
            return []

def get_strategy_ranking():
    from collections import defaultdict
    max_g = get_gales()
    with _db_lock:
        conn = _conn(); c = conn.cursor()
        c.execute("SELECT color FROM results_raw ORDER BY id ASC")
        colors = [r[0] for r in c.fetchall()]; conn.close()

    patterns = defaultdict(lambda: {1: 0, 2: 0, "matches": 0})
    n = len(colors)
    for length in range(3, 6):
        for i in range(n - length - max_g):
            seq = tuple(colors[i:i + length])
            if 0 in seq: continue
            patterns[seq]["matches"] += 1
            for alvo in (1, 2):
                for step in range(1 + max_g):
                    idx = i + length + step
                    if idx < n and colors[idx] == alvo:
                        patterns[seq][alvo] += 1; break

    results = []
    for seq, data in patterns.items():
        if data["matches"] >= 5:
            for alvo in (1, 2):
                wins = data[alvo]; acc = wins / data["matches"] * 100
                if acc >= 60.0:
                    nome = "-".join("V" if c == 1 else "P" for c in seq)
                    results.append({"nome": f"Padrão {nome}", "seq": list(seq),
                                    "alvo": alvo, "matches": data["matches"],
                                    "wins": wins, "acc": round(acc, 1)})
    results.sort(key=lambda x: (x["acc"], x["matches"]), reverse=True)
    return results[:20]

def get_stats():
    with _db_lock:
        try:
            conn = _conn(); c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM results_raw"); total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM results_raw WHERE color=0"); w = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM results_raw WHERE color=1"); r = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM results_raw WHERE color=2"); b = c.fetchone()[0]
            c.execute("SELECT color,roll,created_at,round_id FROM results_raw ORDER BY id DESC LIMIT 50")
            rows = c.fetchall(); conn.close()
            return {
                "total": total, "whites": w, "reds": r, "blacks": b,
                "pct_red":   round(r / max(total,1)*100,1),
                "pct_black": round(b / max(total,1)*100,1),
                "pct_white": round(w / max(total,1)*100,1),
                "recent": [{"color":x[0],"roll":x[1],"created_at":x[2],"round_id":x[3]} for x in rows],
                "updated_at":   datetime.now(timezone.utc).isoformat(),
                "ws_connected": ws_status["connected"],
                "ws_events":    ws_status["total_ws"],
                "mode":         get_mode(),
                "max_gales":    get_gales(),
                "groq":         get_groq_config(),
                "scoreboard":   get_scoreboard(),
                "leviathan":    get_leviathan_meta(),
            }
        except Exception as e:
            return {"error": str(e)}

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if   path == "/stats":            self._json(get_stats())
        elif path == "/recent":
            s = get_stats()
            self._json({"results": s.get("recent",[]), "total": s.get("total",0)})
        elif path == "/analysis":         self._json(get_analysis())
        elif path == "/signal":
            a = get_analysis()
            self._json(a.get("signal", {"action":"wait","confidence":0}))
        elif path == "/history":          self._json({"history": get_signal_history(30)})
        elif path == "/strategy_ranking": self._json({"ranking": get_strategy_ranking()})
        elif path == "/leviathan_meta":   self._json(get_leviathan_meta())
        elif path == "/health":           self._json({
            "status": "online", "total": get_total(),
            "ws_connected": ws_status["connected"],
            "ws_events": ws_status["total_ws"],
            "ts": datetime.now(timezone.utc).isoformat(),
            "version": "5.0-pantheon",
        })
        elif path == "/events": self._sse_stream()
        else: self._json({"endpoints": [
            "/stats","/recent","/analysis","/signal",
            "/history","/health","/events",
            "/strategy_ranking","/leviathan_meta",
        ]})

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode("utf-8") if length > 0 else ""

        if path == "/api/mode":
            try:
                data = json.loads(body) if body else {}
                new_mode = data.get("mode")
                if set_mode(new_mode): self._json({"status":"ok","mode":new_mode})
                else: self._json({"status":"error","message":"Invalid mode"},400)
            except: self._json({"status":"error","message":"Invalid JSON"},400)

        elif path == "/api/reset_stats":
            self._json({"status":"ok","reset_at":reset_stats()})

        elif path == "/api/gales":
            try:
                data = json.loads(body) if body else {}
                new_gales = int(data.get("gales",0))
                if set_gales(new_gales): self._json({"status":"ok","max_gales":new_gales})
                else: self._json({"status":"error","message":"Invalid gales"},400)
            except: self._json({"status":"error","message":"Invalid JSON/Value"},400)

        elif path == "/api/groq_config":
            try:
                data = json.loads(body) if body else {}
                set_groq_config(data.get("key",""), data.get("enabled",False))
                self._json({"status":"ok"})
            except: self._json({"status":"error","message":"Invalid JSON"},400)

        else: self._json({"status":"not_found"},404)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self._cors(); self.end_headers(); self.wfile.write(body)

    def _sse_stream(self):
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream; charset=utf-8")
        self.send_header("Cache-Control","no-cache")
        self.send_header("X-Accel-Buffering","no")
        self._cors(); self.end_headers()
        q = queue.Queue(maxsize=50)
        with _sse_lock: _sse_clients.append(q)
        try:
            self.wfile.write(b": connected\n\n"); self.wfile.flush()
            while running:
                try:
                    payload = q.get(timeout=20)
                    self.wfile.write(payload.encode("utf-8")); self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n"); self.wfile.flush()
        except Exception:
            pass
        finally:
            with _sse_lock:
                if q in _sse_clients: _sse_clients.remove(q)

    def log_message(self, *args): pass

def start_http():
    ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler).serve_forever()

COR   = {0:"BRANCO",1:"VERMELHO",2:"PRETO"}
EMOJI = {0:"⚪",1:"🔴",2:"⚫"}

async def ws_collector():
    saved_ids = get_all_ids(); reconnect_delay = 5
    while running:
        try:
            log.info("WebSocket: conectando a %s", WS_URL)
            extra = {"additional_headers": [
                ("User-Agent","Mozilla/5.0"),
                ("Origin","https://blaze.bet.br"),
                ("Referer","https://blaze.bet.br/pt/games/double"),
            ]}
            async with websockets.connect(WS_URL, ping_interval=None,
                                          open_timeout=20, **extra) as ws:
                ws_status["connected"] = True; reconnect_delay = 5; handshook = False

                async def ping():
                    while True:
                        await asyncio.sleep(15)
                        try: await ws.send("2")
                        except: break

                ping_task = asyncio.create_task(ping())
                try:
                    async for raw in ws:
                        msg = raw if isinstance(raw, str) else raw.decode()
                        if msg.startswith("0") and not handshook:
                            jwt = load_jwt()
                            if jwt: await ws.send(f'40{{"jwt":"{jwt}"}}')
                            else:   await ws.send("40")
                            await ws.send('420["cmd",{"id":"subscribe","payload":{"room":"double_room_1"}}]')
                            await ws.send('421["cmd",{"id":"subscribe","payload":{"room":"doubles"}}]')
                            handshook = True; continue
                        if msg == "3": continue
                        if msg == "2": await ws.send("3"); continue
                        if msg.startswith("42"):
                            try:
                                data   = json.loads(msg[2:])
                                evento = data[0]; payload = {}
                                if len(data) > 1 and isinstance(data[1], dict):
                                    inner_id = data[1].get("id","")
                                    payload  = data[1].get("payload", data[1])
                                    evento   = inner_id if inner_id else evento
                                status = payload.get("status","")
                                color  = payload.get("color")
                                roll   = payload.get("roll")
                                rid    = str(payload.get("id") or payload.get("uuid") or "")
                                if status=="complete" and color is not None and roll is not None and rid:
                                    if int(color) in (0,1,2) and rid not in saved_ids:
                                        cat = payload.get("created_at", datetime.now(timezone.utc).isoformat())
                                        if insert_result(rid, int(color), int(roll), cat, "websocket"):
                                            saved_ids.add(rid); ws_status["total_ws"] += 1
                                            ws_status["last_event"] = datetime.now().isoformat()
                                            log.info("WS LIVE: %s %s | roll=%-2d | banco=%d",
                                                     EMOJI[int(color)], COR[int(color)],
                                                     int(roll), get_total())
                            except: pass
                finally: ping_task.cancel()
        except Exception as e:
            ws_status["connected"] = False
            log.warning("WebSocket desconectado: %s", e)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)

def fetch_recent():
    for url in RECENT_URLS:
        try:
            r = requests.get(url, headers=REST_HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data: return data
                for k in ("records","items","data","results"):
                    if isinstance(data.get(k), list) and data[k]: return data[k]
        except: pass
    return []

def rest_fallback_loop():
    saved_ids = get_all_ids()
    while running:
        time.sleep(POLL_SEC)
        last = ws_status.get("last_event")
        if last:
            try:
                last_dt = datetime.fromisoformat(last.replace("Z","+00:00"))
                if last_dt.tzinfo: last_dt = last_dt.replace(tzinfo=None)
                if (datetime.now() - last_dt).total_seconds() < 120: continue
            except: pass
        for item in fetch_recent():
            rid   = str(item.get("id") or "")
            color = item.get("color"); roll = item.get("roll")
            cat   = item.get("created_at", datetime.now(timezone.utc).isoformat())
            if not rid or color is None or roll is None: continue
            if int(color) not in (0,1,2) or rid in saved_ids: continue
            if insert_result(rid, int(color), int(roll), cat, "rest_fallback"):
                saved_ids.add(rid)
                log.info("REST fallback: %s %s | roll=%-2d | banco=%d",
                         EMOJI[int(color)], COR[int(color)], int(roll), get_total())

def carga_inicial():
    log.info("Carga inicial via REST...")
    n = 0
    for item in fetch_recent():
        rid   = str(item.get("id") or "")
        color = item.get("color"); roll = item.get("roll")
        cat   = item.get("created_at", datetime.now(timezone.utc).isoformat())
        if not rid or color is None or roll is None: continue
        if int(color) not in (0,1,2): continue
        if insert_result(rid, int(color), int(roll), cat, "init"): n += 1
    log.info("Carga inicial: %d novos | total banco=%d", n, get_total())

def shutdown(s=None, f=None):
    global running; running = False
    log.info("Encerrado. total banco=%d", get_total()); sys.exit(0)

def main():
    sig_mod.signal(sig_mod.SIGINT,  shutdown)
    sig_mod.signal(sig_mod.SIGTERM, shutdown)
    log.info("=" * 64)
    log.info("  BLAZE DOUBLE AI — COLETOR v4.0 (Leviathan Backend)")
    log.info("  Iniciado: %s", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    log.info("  WS  : %s", WS_URL)
    log.info("  HTTP: localhost:%d", HTTP_PORT)
    log.info("=" * 64)
    if not HAS_WS:
        log.error("ERRO: websockets nao instalado. Rode: pip install websockets")
        sys.exit(1)
    init_db()
    threading.Thread(target=start_http, daemon=True).start()
    log.info("HTTP online em http://localhost:%d", HTTP_PORT)
    carga_inicial()
    threading.Thread(target=rest_fallback_loop, daemon=True).start()
    log.info("Fallback REST iniciado.")
    asyncio.run(ws_collector())

if __name__ == "__main__":
    main()