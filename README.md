# 🔥 Blaze Double AI v2.0 — Sistema Completo

Sistema modular e integrado de coleta contínua, análise estatística, IA multi-agente e dashboard em tempo real para o Double da Blaze.

---

## ⚡ Início rápido (tudo em 1 comando)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Iniciar tudo (coletor + analisador + otimizador + dashboard)
python start.py

# 3. Abrir dashboard local
# O navegador será aberto automaticamente pelo start.py
```

---

## 📁 Estrutura

```
superIA/
├── start.py                  ← Launcher que inicia coletor, analisador e otimizador
├── coletor.py                ← Coleta em tempo real + servidor HTTP + SSE
├── analisador.py             ← Motor Leviathan de análise, sinais e auto-adaptação
├── otimizador_estrategias.py ← Minerador/otimizador de estratégias e catálogo
├── notificador.py            ← Envio de alertas via Telegram
├── set_token.py              ← Configuração de JWT Blaze e chave Groq
├── blaze-dashboard.html      ← Interface dashboard local
├── simulador.html            ← Interface de simulação local
├── blaze_double.db           ← Banco SQLite gerado automaticamente
├── blaze_token.json          ← Arquivo de token JWT gerado pelo set_token.py
├── requirements.txt          ← Dependências Python
└── README.md                 ← Documentação do projeto
```

---

## 🌐 API HTTP (localhost:8765)

| Endpoint | Retorna |
|---|---|
| `GET /stats` | Estatísticas gerais, contagem e resultados recentes |
| `GET /analysis` | Último snapshot de análise do motor |
| `GET /signal` | Último sinal gerado pelo analisador |
| `GET /history` | Histórico recente de sinais |
| `GET /strategy_ranking` | Ranking de padrões mining local |
| `GET /leviathan_meta` | Metadados do Leviathan |
| `GET /health` | Status do serviço |

---

## 🤖 Motor de IA (analisador.py)

Combina **5 módulos** de evidência no ensemble:

| Módulo | Origem | Descrição |
|---|---|---|
| Minerador local | analisador.py | Padrões exatos `ngram` extraídos do histórico recente |
| Catálogo | database | Estratégias ativas aprovadas pelo otimizador |
| Markov | analisador.py | Probabilidade de transição com ordens 1 e 2 |
| Streak | analisador.py | Reversão/continuação de streaks de cores |
| White cycle | analisador.py | Comportamento pós-branca e hazard branco |

---

## 🔧 Configuração de tokens

Use `set_token.py` para gravar o JWT Blaze e a chave Groq:

```bash
python set_token.py --jwt SEU_JWT
python set_token.py --groq SUA_CHAVE_GROQ
python set_token.py --groq-on
python set_token.py --groq-off
python set_token.py --status
```

---

## 📱 Telegram

Configurado em `notificador.py`. Envia alertas automáticos quando:
- Sinal de **ENTRAR** com alta confiança
- Sinal de **GALE** ou auto-mute
- Encerramento do sistema

---

## 🗃️ Banco de dados (SQLite)

| Tabela | Conteúdo |
|---|---|
| `results_raw` | Resultados coletados (cor, número, timestamp, fonte) |
| `analysis_snapshots` | Snapshots de análise, sinal e features |
| `prediction_performance` | Avaliação de acertos/erros dos sinais |
| `strategy_catalog` | Catálogo vivo de estratégias aprovadas pelo otimizador |
| `strategy_backtests` | Histórico de backtests do otimizador |
| `optimizer_runs` | Registros de execuções do otimizador |
