"""
Diagnostico completo dos agentes Pantheon
Testa importacao, execucao basica e retorno de cada agente
"""
import sys
import traceback
import io

# Forca saida UTF-8 para suportar emojis dos labels dos agentes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Cores ANSI (desabilitadas no Windows sem suporte)
OK   = "[OK]   "
FAIL = "[FAIL] "
WARN = "[WARN] "

resultados = {}

print("=" * 60)
print("   DIAGNOSTICO DOS AGENTES PANTHEON")
print("=" * 60)

# ──────────────────────────────────────────────────────────
# Dados de entrada simulados (realistas)
# ──────────────────────────────────────────────────────────
COLORS = [1,2,1,1,2,0,1,2,2,1,2,1,0,2,1,1,2,1,2,2,
          1,2,1,2,0,1,2,1,1,2,2,1,0,1,2,1,2,1,2,1,
          1,2,1,2,1,0,1,2,1,2]  # 50 valores: 0=branco,1=vermelho,2=preto

# Regime como DICT (formato real esperado pelos agentes)
REGIME = {"name": "balanced", "strength": 0.6}

# ──────────────────────────────────────────────────────────
# 1. SYBIL
# ──────────────────────────────────────────────────────────
print("\n[1/6] agent_sybil")
try:
    from agents.agent_sybil import expert_sybil
    res = expert_sybil(COLORS, REGIME)
    assert isinstance(res, dict), "Retorno nao e dict"
    assert 'signal' in res or 'color' in res or 'vote' in res or len(res) > 0, "Dict vazio"
    print(f"  {OK} Import OK | Retorno: {res}")
    resultados['sybil'] = True
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    resultados['sybil'] = False

# ──────────────────────────────────────────────────────────
# 2. CHAOS
# ──────────────────────────────────────────────────────────
print("\n[2/6] agent_chaos")
try:
    from agents.agent_chaos import expert_chaos
    res = expert_chaos(COLORS, REGIME)
    assert isinstance(res, dict), "Retorno nao e dict"
    print(f"  {OK} Import OK | Retorno: {res}")
    resultados['chaos'] = True
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    resultados['chaos'] = False

# ──────────────────────────────────────────────────────────
# 3. HERMES
# ──────────────────────────────────────────────────────────
print("\n[3/6] agent_hermes")
try:
    from agents.agent_hermes import expert_hermes, _hermes_state
    res = expert_hermes(COLORS, REGIME)
    assert isinstance(res, dict), "Retorno nao e dict"
    print(f"  {OK} Import OK | Retorno: {res}")
    resultados['hermes'] = True
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    resultados['hermes'] = False

# ──────────────────────────────────────────────────────────
# 4. ORACLE
# ──────────────────────────────────────────────────────────
print("\n[4/6] agent_oracle")
try:
    from agents.agent_oracle import (
        oracle_get_weights, oracle_learn,
        oracle_save_state, oracle_load_state
    )
    weights = oracle_get_weights(COLORS, REGIME)
    assert isinstance(weights, dict), "oracle_get_weights nao e dict"
    print(f"  {OK} oracle_get_weights() OK | Pesos: {dict(list(weights.items())[:3])} ...")
    # Testa oracle_learn com dado simulado (assinatura: won, colors, regime)
    oracle_learn(won=True, colors=COLORS, regime=REGIME)
    print(f"  {OK} oracle_learn() executou sem erros")
    resultados['oracle'] = True
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    resultados['oracle'] = False

# ──────────────────────────────────────────────────────────
# 5. ATLAS
# ──────────────────────────────────────────────────────────
print("\n[5/6] agent_atlas")
try:
    from agents.agent_atlas import expert_atlas, atlas_rebuild_tree
    res = expert_atlas(COLORS, REGIME)
    assert isinstance(res, dict), "Retorno nao e dict"
    # Remove emojis do label para exibir no terminal Windows
    label = res.get('label', '').encode('ascii', 'replace').decode('ascii')
    print(f"  {OK} Import OK | vote={res.get('vote')} conf={res.get('confidence')} label={label}")
    resultados['atlas'] = True
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    resultados['atlas'] = False

# ──────────────────────────────────────────────────────────
# 6. TITAN
# ──────────────────────────────────────────────────────────
print("\n[6/6] agent_titan")
try:
    from agents.agent_titan import expert_titan, titan_rebuild
    res = expert_titan(COLORS, REGIME)
    assert isinstance(res, dict), "Retorno nao e dict"
    print(f"  {OK} Import OK | Retorno: {res}")
    resultados['titan'] = True
except Exception as e:
    print(f"  {FAIL} {e}")
    traceback.print_exc()
    resultados['titan'] = False

# ──────────────────────────────────────────────────────────
# RESUMO FINAL
# ──────────────────────────────────────────────────────────
total  = len(resultados)
ok     = sum(resultados.values())
falhos = [k for k, v in resultados.items() if not v]

print("\n" + "=" * 60)
print(f"   RESULTADO: {ok}/{total} agentes funcionando")
print("=" * 60)
for agente, status in resultados.items():
    s = OK if status else FAIL
    print(f"  {s} {agente.upper()}")

if falhos:
    print(f"\n  Agentes com problema: {', '.join(falhos).upper()}")
else:
    print("\n  Todos os agentes operacionais!")
print("=" * 60)
