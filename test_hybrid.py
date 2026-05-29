"""
Teste integrado do Sistema Hibrido Pantheon + V4
Simula 100 rodadas reais e conta sinais emitidos
"""
import sys, sqlite3
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 65)
print("   TESTE INTEGRADO — SISTEMA HIBRIDO PANTHEON + V4")
print("=" * 65)

# ── Inicializa o motor hibrido ─────────────────────────────────
from pantheon_engine import pantheon_engine, get_engine_stats

print("\n[1] Inicializando motor hibrido com blaze_double.db...")
pantheon_engine.db_path = 'blaze_double.db'
pantheon_engine.initialize()

# ── Stats ──────────────────────────────────────────────────────
stats = get_engine_stats()
print(f"\n[2] ENGINE STATS:")
print(f"    Versao        : {stats['version']}")
print(f"    Agentes orig  : {stats['agents_count']}")
print(f"    V4 Experts    : {stats['v4_experts']}")
print(f"    Total fontes  : {stats['total_sources']}")
print(f"    Modo          : {stats['mode']}")
print(f"    Peso V4       : {stats['v4_weight']:.0%}")
print(f"    Agentes OK    : {stats['agents_ok']}")
print(f"    V4 OK         : {stats['v4_ok']}")

# ── Carrega ultimas 150 rodadas reais ─────────────────────────
conn = sqlite3.connect('blaze_double.db')
df   = pd.read_sql_query(
    "SELECT roll as value, color, created_at as timestamp "
    "FROM results_raw ORDER BY created_at DESC LIMIT 150",
    conn); conn.close()
df   = df.iloc[::-1].reset_index(drop=True)

COLOR_MAP = {0: 'white', 1: 'red', 2: 'black'}

print(f"\n[3] SIMULANDO 100 RODADAS REAIS...")
print(f"    {'#':>3}  {'Roll':>4}  {'Cor':6}  {'Sinal':8}  {'Conf':6}  {'Agentes':8}  {'V4':8}  {'Regime'}")
print(f"    {'-'*70}")

sinais = 0
reds = 0; blacks = 0

for idx, row in df.iterrows():
    data = {
        'value':     int(row['value']),
        'color':     COLOR_MAP.get(int(row['color']), 'red'),
        'timestamp': row['timestamp'],
    }
    result = pantheon_engine.analyze_round(data)

    signal = result.get('signal', 'NO_BET')
    conf   = result.get('confidence', 0.)
    agents = result.get('agent_votes', 0)
    v4s    = result.get('v4_signal') or '-'
    regime = result.get('regime', '?')

    if signal not in ('NO_BET', 'WAIT'):
        sinais += 1
        if signal == 'RED':   reds   += 1
        if signal == 'BLACK': blacks += 1
        cor = COLOR_MAP.get(int(row['color']), '?')
        acerto = 'ACERTO' if (signal == 'RED' and cor == 'red') or \
                             (signal == 'BLACK' and cor == 'black') else 'ERRO'
        print(f"    {idx+1:>3}  {row['value']:>4}  {cor:6}  "
              f"{signal:8}  {conf:.3f}  {agents:>8}  {v4s:8}  {regime}  <- {acerto}")

# ── Resumo ────────────────────────────────────────────────────
print(f"\n    Total de rodadas simuladas : {len(df)}")
print(f"    Sinais emitidos            : {sinais}")
print(f"      -> RED  : {reds}")
print(f"      -> BLACK: {blacks}")
print(f"    Taxa de sinal              : {sinais/len(df):.1%}")

print("\n" + "=" * 65)
if sinais > 0:
    print("  [OK] SISTEMA HIBRIDO OPERACIONAL E EMITINDO SINAIS!")
else:
    print("  [WARN] Nenhum sinal emitido — historico insuficiente no buffer")
print("=" * 65)
