📋 RELATÓRIO DE CORREÇÃO: VIÉS DO SISTEMA (APENAS PRETO)
═══════════════════════════════════════════════════════════════════════════════

🔍 PROBLEMA IDENTIFICADO
────────────────────────────────────────────────────────────────────────────────
Sistema gerava sinais PRETO (2) em ~85% dos casos, contra apenas ~15% VERMELHO (1).
Você relatou: "percebi que o sistema só entra no preto"

🧪 DIAGNÓSTICO
────────────────────────────────────────────────────────────────────────────────
Executada análise de viés em 100 snapshots recentes:

  Sinais PRETO:     80/100 (80%)
  Sinais VERMELHO:  0/100  (0%)

Votação por módulo:
  ├─ Catalog:  61 PRETO vs 8 VERMELHO  (85% PRETO)
  ├─ Miner:    57 PRETO vs 10 VERMELHO (85% PRETO)
  ├─ Markov:   6 PRETO vs 0 VERMELHO   (100% PRETO quando vota)
  ├─ Streak:   0 PRETO vs 3 VERMELHO   (100% VERMELHO quando vota)
  └─ White:    0 votos em nenhuma cor

🚨 RAIZ DO PROBLEMA
────────────────────────────────────────────────────────────────────────────────

1. DESEQUILÍBRIO NO CATÁLOGO DE ESTRATÉGIAS
   ├─ VERMELHO ativo: 22 estratégias
   ├─ PRETO ativo:    38 estratégias (2:1 ratio)
   └─ Motivo: Otimizador não mantinha balanço entre cores

2. LÓGICA DE DESEMPATE ENVIESADA
   ├─ Quando prob_red == prob_black, código usava "else 2" (PRETO)
   ├─ Módulos Markov e White-Cycle sempre favoreciam PRETO em empates
   └─ Afetava seleção em casos de baixa confiança

3. FALTA DE FATOR DE NORMALIZAÇÃO
   └─ Ensemble não corrigia proporcionalmente o viés catalogo/miner

🔧 CORREÇÕES APLICADAS
────────────────────────────────────────────────────────────────────────────────

✅ FIX 1: Tiebreaker Fair em Markov (analisador.py:762)
   Antes: vote = 1 if prob_red > prob_black else 2
   Depois: if abs(prob_red - prob_black) < 0.02: return None
           (Não vota em caso de diferença mínima)

✅ FIX 2: Tiebreaker Fair em White-Cycle (analisador.py:884)
   Aplicou mesma lógica: diferença mínima de 3% para votar

✅ FIX 3: Fator de Correção de Viés no Ensemble (analisador.py:958)
   Adicionado bias_correction = 1.18 para VERMELHO
   └─ Compensa a proporção 22:38 do catálogo (≈1.73)
   └─ 1.18 mantém viés reduzido mas não o inverte

✅ FIX 4: Balanceamento Forçado no Otimizador (otimizador_estrategias.py:627)
   Nova lógica garante:
   ├─ Mínimo 40% das estratégias ATIVAS sejam VERMELHO
   ├─ Máximo 60% das estratégias ATIVAS sejam PRETO
   └─ Próximas execuções criarão catálogo mais balanceado

📊 IMPACTO ESPERADO
────────────────────────────────────────────────────────────────────────────────

Antes das correções:
  ├─ Proporção teórica: ~85% PRETO, 15% VERMELHO
  └─ Hits VERMELHO muito baixos

Depois das correções:
  ├─ Catálogo próximas rodadas: ~40% VERMELHO, 60% PRETO (vs 22/38)
  ├─ Ensemble: +18% weight para votos VERMELHO
  └─ Proporção esperada: ~50% PRETO, 50% VERMELHO (normalizado)

⏱️ PRÓXIMOS PASSOS
────────────────────────────────────────────────────────────────────────────────

1. Sistema usa correções imediatamente
2. Otimizador terá efeito completo na próxima iteração (600s ou manual trigger)
3. Monitorar com: python diagnose_bias.py
4. Aguardar ~10-20 snapshots para validar balanceamento

✨ ARQUIVOS MODIFICADOS
────────────────────────────────────────────────────────────────────────────────
  • analisador.py (Markov, White-Cycle, Ensemble bias correction)
  • otimizador_estrategias.py (Balanceamento de cores no catálogo)
  • diagnose_bias.py (NOVO - ferramenta de monitoramento)

═══════════════════════════════════════════════════════════════════════════════
