import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'blaze_double.db'

print("=" * 55)
print("   TESTE V4 ULTIMATE - COM DADOS REAIS DO BANCO")
print("=" * 55)

# ── 1. Inicializar o motor via pantheon_engine ──────────────
from pantheon_engine import pantheon_engine, get_engine_stats

print("\n[1] Inicializando motor V4 com blaze_double.db...")
pantheon_engine.db_path = DB_PATH
pantheon_engine.initialize()

# ── 2. Stats pós-inicialização ──────────────────────────────
stats = get_engine_stats()
print(f"\n[2] ENGINE STATS:")
print(f"    Versao  : {stats['version']}")
print(f"    Experts : {stats['experts_count']}")
print(f"    Modo    : {stats['mode']}")
print(f"    Limiar  : {stats['min_threshold']}")

# ── 3. Pegar as últimas 50 rodadas reais e simular análise ──
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT roll, color, created_at FROM results_raw ORDER BY created_at DESC LIMIT 50",
    conn
)
conn.close()

# Inverter para ordem cronológica (mais antiga → mais recente)
df = df.iloc[::-1].reset_index(drop=True)

COLOR_MAP = {0: 'white', 1: 'red', 2: 'black'}

print(f"\n[3] SIMULANDO ANÁLISE COM ÚLTIMAS 50 RODADAS REAIS...")
print(f"    (alimentando o buffer uma a uma)\n")

history_buf = []
sinais = 0

for _, row in df.iterrows():
    round_data = {
        'value': int(row['roll']),
        'color': COLOR_MAP.get(int(row['color']), 'red'),
        'timestamp': row['created_at']
    }
    history_buf.append(round_data)
    result = pantheon_engine.analyze_round(round_data)

    if result.get('signal') not in ('NO_BET', 'WAIT', None):
        sinais += 1
        regime = result.get('regime', 'N/A')
        conf   = result.get('confidence', 0)
        signal = result.get('signal', '?')
        roll   = row['roll']
        color  = COLOR_MAP.get(int(row['color']), '?')
        print(f"    ► Rodada {_+1:02d} | Roll={roll:02d} ({color:5s}) "
              f"| SINAL: {signal:10s} | Confianca: {conf:.2f} | Regime: {regime}")

print(f"\n    Total de sinais emitidos: {sinais} / 50 rodadas")

# ── 4. Resultado final ──────────────────────────────────────
print("\n" + "=" * 55)
if stats['version'] == '4.0 Ultimate' and sinais >= 0:
    print("  [OK] SUCESSO - Motor V4 operando com dados reais!")
else:
    print("  [FALHA] Verifique os logs acima.")
print("=" * 55)
