"""
=============================================================================
BLAZE DOUBLE AI — CONFIGURAR TOKEN JWT / GROQ
Exemplos:
  python set_token.py --jwt SEU_JWT
  python set_token.py --groq gsk_xxxxx
  python set_token.py --groq-on
  python set_token.py --groq-off
  python set_token.py --status
=============================================================================
"""

import sys
import json
import os
import sqlite3

TOKEN_FILE = "blaze_token.json"
DB_PATH = "blaze_double.db"


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_jwt(token: str):
    token = token.strip().replace('"', "").replace("'", "")
    if token.startswith("Bearer "):
        token = token[7:]

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"jwt": token}, f, ensure_ascii=False)

    print(f"JWT salvo em {TOKEN_FILE}")
    print(f"JWT: {token[:40]}...")


def load_jwt() -> str:
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("jwt", "")
        except Exception:
            pass
    return os.environ.get("BLAZE_JWT", "")


def set_config(key: str, value: str):
    init_db()
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()
    conn.close()


def get_config(key: str, default: str = "") -> str:
    init_db()
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT value FROM system_config WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default


def save_groq_key(key: str):
    key = key.strip().replace('"', "").replace("'", "")
    set_config("groq_key", key)
    print(f"Groq key salva: {key[:18]}...")


def set_groq_enabled(enabled: bool):
    set_config("llm_enabled", "1" if enabled else "0")
    print(f"Groq {'ATIVADO' if enabled else 'DESATIVADO'}")


def show_status():
    jwt = load_jwt()
    groq_key = get_config("groq_key", "")
    groq_enabled = get_config("llm_enabled", "0") == "1"

    print("=" * 55)
    print(" STATUS ATUAL")
    print("=" * 55)
    print(f"JWT Blaze : {'OK' if jwt else 'NÃO CONFIGURADO'}")
    if jwt:
        print(f"  {jwt[:50]}...")
    print(f"Groq Key  : {'OK' if groq_key else 'NÃO CONFIGURADA'}")
    if groq_key:
        print(f"  {groq_key[:24]}...")
    print(f"Groq LLM  : {'ATIVADO' if groq_enabled else 'DESATIVADO'}")
    print("=" * 55)


def usage():
    print(__doc__)


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        usage()
        sys.exit(0)

    if args[0] == "--jwt" and len(args) > 1:
        save_jwt(args[1])
        sys.exit(0)

    if args[0] == "--groq" and len(args) > 1:
        save_groq_key(args[1])
        set_groq_enabled(True)
        sys.exit(0)

    if args[0] == "--groq-on":
        set_groq_enabled(True)
        sys.exit(0)

    if args[0] == "--groq-off":
        set_groq_enabled(False)
        sys.exit(0)

    if args[0] == "--status":
        show_status()
        sys.exit(0)

    usage()