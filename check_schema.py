import sqlite3
db = sqlite3.connect('blaze_double.db')
c = db.cursor()

print("📋 SCHEMA DE TABELAS:\n")

for table in ['analysis_snapshots', 'results_raw', 'strategy_catalog', 'strategy_backtests']:
    try:
        c.execute(f"PRAGMA table_info({table})")
        print(f"\n{table}:")
        for row in c.fetchall():
            print(f"  {row[1]:25} {row[2]}")
    except:
        print(f"\n{table}: (não existe)")

db.close()
