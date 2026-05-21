"""
=============================================================================
REORGANIZADOR DE ESTRATÉGIAS - BASEADO EM QUALIDADE REAL
Remove viés de cor, elimina ineficazes, reconstrói catálogo com dados REAIS

Processo:
  1. Lê TODO o banco de dados (results_raw) para cores reais
  2. Reavalia cada estratégia por histório de resultados reais
  3. Remove/inativa estratégias ineficazes sem viés de cor
  4. Ordena APENAS por qualidade e performance real
  5. Seleciona TOP 60 para catálogo ativo

Uso:
  python reorganizador_qualidade.py
=============================================================================
"""
import sqlite3
import json
from collections import defaultdict
from datetime import datetime, timezone

DB_PATH = "blaze_double.db"

# Parâmetros de qualidade
MIN_QUALITY_ACTIVE = 0.56   # Mínimo para estar ativo
MIN_QUALITY_STANDBY = 0.50  # Mínimo para standby
MAX_ACTIVE = 60


def _conn():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def get_all_colors():
    """Retorna lista de todas as cores do banco (results_raw)."""
    try:
        conn = _conn()
        c = conn.cursor()
        c.execute(
            "SELECT color FROM results_raw WHERE color IN (0,1,2) ORDER BY id DESC"
        )
        colors = [row[0] for row in c.fetchall() if row[0] in (0, 1, 2)]
        conn.close()
        return list(reversed(colors))
    except Exception as e:
        print(f"❌ Erro ao ler cores: {e}")
        return []


def evaluate_strategy_on_history(strategy, colors):
    """
    Avalia estratégia contra histórico de cores reais.
    Retorna: (matches, wins, accuracy)
    """
    family = strategy["family"]
    params = strategy["params"]
    target = strategy["target_color"]
    matches = wins = 0

    # Importar helpers do otimizador
    from otimizador_estrategias import (
        strategy_matches,
        hit_target,
    )

    upper = len(colors) - 3  # Deixar margem para lookahead
    for i in range(upper):
        if strategy_matches(strategy, colors, i):
            matches += 1
            if hit_target(colors, i, target, 2):
                wins += 1

    acc = wins / max(matches, 1) if matches > 0 else 0.0
    return matches, wins, acc


def recalculate_quality(acc, matches, recent_matches, recent_wins):
    """
    Calcula score de qualidade puro baseado em performance.
    Sem viés de cor.
    """
    if matches < 6:
        return 0.0
    if recent_matches < 3:
        recent_acc = 0.5
    else:
        recent_acc = recent_wins / recent_matches

    # Combinar walk-forward accuracy com recente (favor recente)
    quality = acc * 0.4 + recent_acc * 0.6
    return round(max(quality, 0.0), 6)


def main():
    print("\n" + "=" * 80)
    print(" 🔄 REORGANIZADOR DE ESTRATÉGIAS - BASEADO EM QUALIDADE REAL")
    print("=" * 80)

    conn = _conn()
    c = conn.cursor()

    # 1. Verificar se tabelas existem
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_catalog'"
    )
    if not c.fetchone():
        print("\n❌ Tabela strategy_catalog não existe. Abortando.")
        conn.close()
        return

    # 2. Carregar todas as cores do histórico
    print("\n📊 Lendo histórico de cores do banco...")
    colors = get_all_colors()
    print(f"   ✅ Carregadas {len(colors)} rodadas de histórico")

    if len(colors) < 100:
        print("   ⚠️  Histórico muito curto (<100 rodadas). Abortando.")
        conn.close()
        return

    # 3. Carregar estratégias atuais
    print("\n📚 Carregando estratégias atuais...")
    c.execute(
        """
        SELECT strategy_id, family, name, params_json, target_color, 
               wf_acc, recent_acc, recent_matches, recent_wins, status
        FROM strategy_catalog
        ORDER BY strategy_id
    """
    )

    strategies = []
    for row in c.fetchall():
        try:
            params = json.loads(row[3]) if row[3] else {}
            strategies.append(
                {
                    "id": row[0],
                    "family": row[1],
                    "name": row[2],
                    "params": params,
                    "target_color": row[4],
                    "wf_acc": row[5] or 0.0,
                    "recent_acc": row[6] or 0.0,
                    "recent_matches": row[7] or 0,
                    "recent_wins": row[8] or 0,
                    "status": row[9],
                }
            )
        except Exception:
            continue

    print(f"   ✅ Carregadas {len(strategies)} estratégias")

    if not strategies:
        print("   ❌ Nenhuma estratégia para processar.")
        conn.close()
        return

    # 4. Reavaliar todas contra histórico
    print("\n🧪 Reavaliando todas as estratégias contra histórico real...")
    stats = defaultdict(int)

    for i, s in enumerate(strategies):
        try:
            matches, wins, acc = evaluate_strategy_on_history(s, colors)
            s["quality"] = recalculate_quality(
                acc, matches, s["recent_matches"], s["recent_wins"]
            )
            s["matches_hist"] = matches
            s["wins_hist"] = wins
            s["acc_hist"] = acc

            # Contar por qualidade
            if s["quality"] >= MIN_QUALITY_ACTIVE:
                stats["good"] += 1
            elif s["quality"] >= MIN_QUALITY_STANDBY:
                stats["standby"] += 1
            else:
                stats["bad"] += 1

            if (i + 1) % 20 == 0:
                print(f"   Processadas {i+1}/{len(strategies)}...")

        except Exception as e:
            print(f"   ⚠️  Erro ao avaliar {s['id']}: {e}")
            s["quality"] = 0.0
            stats["error"] += 1

    print(f"   ✅ Reavaliação completa")

    # 5. Estatísticas
    print("\n📈 Estatísticas de qualidade:")
    print(f"   Excelentes (≥{MIN_QUALITY_ACTIVE:.0%}): {stats['good']}")
    print(f"   Standby ({MIN_QUALITY_STANDBY:.0%}-{MIN_QUALITY_ACTIVE:.0%}): {stats['standby']}")
    print(f"   Ineficazes (<{MIN_QUALITY_STANDBY:.0%}): {stats['bad']}")
    print(f"   Com erro: {stats['error']}")

    # 6. Ordenar por qualidade (SEM VIÉS DE COR)
    print("\n🎯 Selecionando TOP estratégias por qualidade pura...")
    strategies.sort(key=lambda x: (x["quality"], x["matches_hist"]), reverse=True)

    # Mostrar TOP 10
    print("\n   TOP 10 por qualidade:")
    print("   Rank | ID | Cor | Qualidade | Matches | Wins | Acurácia")
    print("   " + "-" * 65)
    for idx, s in enumerate(strategies[:10], 1):
        color_name = "VERM" if s["target_color"] == 1 else "PRET"
        print(
            f"   {idx:2}   | {s['id'][:8]}... | {color_name} | {s['quality']:.4f} | {s['matches_hist']:3} | {s['wins_hist']:3} | {s['acc_hist']:.1%}"
        )

    # 7. Determinar novo status SEM VIÉS
    print("\n🔧 Aplicando novo status baseado em qualidade...")

    active_count = 0
    for s in strategies:
        if s["quality"] >= MIN_QUALITY_ACTIVE and active_count < MAX_ACTIVE:
            s["new_status"] = "active"
            active_count += 1
        elif s["quality"] >= MIN_QUALITY_STANDBY:
            s["new_status"] = "standby"
        else:
            s["new_status"] = "inactive"

    active_by_color = defaultdict(int)
    for s in strategies:
        if s["new_status"] == "active":
            active_by_color[s["target_color"]] += 1

    print(f"\n   Distribuição final no ATIVO:")
    print(f"   VERMELHO (1): {active_by_color[1]} estratégias")
    print(f"   PRETO (2):    {active_by_color[2]} estratégias")
    print(f"   TOTAL ATIVO:  {active_by_color[1] + active_by_color[2]} / {MAX_ACTIVE}")

    # 8. Aplicar mudanças ao banco
    print("\n💾 Gravando mudanças no banco...")
    run_ts = datetime.now(timezone.utc).isoformat()

    updated = 0
    for s in strategies:
        if s["new_status"] != s["status"]:
            c.execute(
                """
                UPDATE strategy_catalog
                SET status=?, updated_at=?
                WHERE strategy_id=?
            """,
                (s["new_status"], run_ts, s["id"]),
            )
            updated += 1

    conn.commit()
    conn.close()

    print(f"   ✅ {updated} estratégias atualizadas")

    # 9. Relatório final
    print("\n" + "=" * 80)
    print(" ✅ REORGANIZAÇÃO COMPLETA")
    print("=" * 80)
    print(f"\nCatálogo agora contém:")
    print(f"  • {active_by_color[1]} estratégias VERMELHO (selecionadas por qualidade)")
    print(f"  • {active_by_color[2]} estratégias PRETO (selecionadas por qualidade)")
    print(f"  • Total ATIVO: {sum(active_by_color.values())} / {MAX_ACTIVE}")
    print(f"\n🎯 SEM VIÉS: Distribuição reflete EFETIVIDADE REAL, não cores")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
