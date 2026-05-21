"""
Diagnóstico de viés de votação do ensemble.
Monitora a distribuição de votos dos 5 módulos.
"""

import sqlite3
import json
from collections import defaultdict

DB_PATH = "blaze_double.db"


def analyze_votes():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Busca últimos 100 snapshots com votos
    c.execute("""
        SELECT id, votes_json, signal_color, signal_action
        FROM analysis_snapshots
        WHERE votes_json IS NOT NULL
        ORDER BY id DESC
        LIMIT 100
    """)

    snapshots = c.fetchall()
    conn.close()

    if not snapshots:
        print("❌ Nenhum snapshot com votos_json encontrado.")
        return

    module_stats = defaultdict(lambda: {"red": 0, "black": 0, "none": 0})
    signal_stats = defaultdict(int)
    color_dist = {1: 0, 2: 0, None: 0}

    for snap_id, votes_json_str, signal_color, signal_action in snapshots:
        try:
            votes = json.loads(votes_json_str) if isinstance(votes_json_str, str) else votes_json_str
        except:
            continue

        signal_stats[signal_action] += 1
        color_dist[signal_color] += 1

        for vote in votes:
            if not isinstance(vote, dict):
                continue
            module = vote.get("module") or vote.get("source", "unknown")
            vote_val = vote.get("vote")

            if vote_val == 1:
                module_stats[module]["red"] += 1
            elif vote_val == 2:
                module_stats[module]["black"] += 1
            else:
                module_stats[module]["none"] += 1

    print("\n" + "=" * 70)
    print(" DIAGNÓSTICO DE VIÉS DE VOTAÇÃO DO ENSEMBLE")
    print("=" * 70)

    print("\n📊 Distribuição de sinais:")
    for action, count in signal_stats.items():
        print(f"  {action.upper():12} : {count:3} occorrências")

    print("\n🎲 Cores nos sinais:")
    for color, count in color_dist.items():
        color_name = {1: "VERMELHO", 2: "PRETO", None: "NULO"}.get(color, "?")
        print(f"  {color_name:12} : {count:3} occorrências")

    print("\n🔴 Votos por módulo:")
    print(f"{'Módulo':<20} {'VERMELHO':<12} {'PRETO':<12} {'NENHUM':<12} {'TOTAL':<8}")
    print("-" * 70)

    for module in sorted(module_stats.keys()):
        stats = module_stats[module]
        total = stats["red"] + stats["black"] + stats["none"]
        red_pct = f"{stats['red']}/{total}" if total else "0/0"
        black_pct = f"{stats['black']}/{total}" if total else "0/0"
        none_pct = f"{stats['none']}/{total}" if total else "0/0"

        print(
            f"{module:<20} {red_pct:<12} {black_pct:<12} {none_pct:<12} {total:<8}"
        )

    print("\n" + "=" * 70)
    print("🔍 Análise:")

    # Verifica viés global
    total_red = sum(s["red"] for s in module_stats.values())
    total_black = sum(s["black"] for s in module_stats.values())
    total_votes = total_red + total_black

    if total_votes > 0:
        red_ratio = total_red / total_votes
        black_ratio = total_black / total_votes
        print(
            f"  Proporção VERMELHO/PRETO: {red_ratio:.2%} / {black_ratio:.2%}"
        )

        if black_ratio > 0.65:
            print("  ⚠️  VIÉS DETECTADO: Sistema vota muito em PRETO")
            print("     Investigar: markov_prob, module_white_cycle, empates")
        elif red_ratio > 0.65:
            print("  ⚠️  VIÉS DETECTADO: Sistema vota muito em VERMELHO")
        else:
            print("  ✅ Distribuição equilibrada")
    else:
        print("  ❌ Nenhum voto registrado")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    analyze_votes()
