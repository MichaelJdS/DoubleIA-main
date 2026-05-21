# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE ORACLE — Meta Reinforcement Learning Gating
#  Q-Learning agent que aprende QUAIS experts confiar em cada micro-regime.
#  Estado = (regime, streak_len, entropy_level, white_zone)
#  Ação   = peso multiplicador para cada expert (discretizado)
#  Reward = +1 para win, -1 para loss (com desconto temporal)
# ═══════════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import threading
from collections import deque

import numpy as np

from agents.agent_chaos import _permutation_entropy

log = logging.getLogger("oracle")
ORACLE_STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "oracle_state.json"))

ORACLE_ALPHA     = 0.08     # learning rate Q-Learning
ORACLE_GAMMA     = 0.85     # fator de desconto
ORACLE_EPSILON   = 0.08     # exploração (8% random para não convergir cedo)
ORACLE_EXPERTS   = [        # lista de todos os experts rastreados
    "miner", "catalog", "markov", "streak",
    "white", "momentum", "alternation", "volatility",
    "antidrift", "sybil", "chaos", "hermes"
]

_oracle_lock  = threading.Lock()
_oracle_state = {
    "q_table":         {},    # {(state_key, expert): Q-value}
    "last_state":      None,
    "last_action":     None,
    "episode_rewards": deque(maxlen=200),
    "total_updates":   0,
}

def _oracle_encode_state(colors, regime):
    """
    Codifica o estado do ambiente em uma chave discreta para a Q-Table.
    Estado = tupla de 5 dimensões discretizadas.
    """
    from analisador import alternation_ratio, poisson_white_hazard, streak_info

    nw    = [c for c in colors if c != 0]
    strk  = streak_info(nw)["length"] if nw else 0
    wh    = poisson_white_hazard(colors)
    alt   = alternation_ratio(colors, 10)

    # Discretização
    streak_bucket = "long" if strk >= 5 else "medium" if strk >= 3 else "short"
    white_bucket  = "high" if wh["hazard"] >= 0.70 else "medium" if wh["hazard"] >= 0.40 else "low"
    alt_bucket    = "high" if alt >= 0.70 else "medium" if alt >= 0.45 else "low"

    # Entropia recente
    pe = _permutation_entropy([c for c in colors[-60:] if c != 0], m=3) if len(colors) >= 30 else 0.8
    entropy_bucket = "high" if pe >= 0.85 else "medium" if pe >= 0.65 else "low"

    return (regime["name"], streak_bucket, white_bucket, alt_bucket, entropy_bucket)

def _oracle_get_q(state_key, expert):
    """Retorna Q-value para (estado, expert). Default = 1.0 (neutro)."""
    with _oracle_lock:
        return _oracle_state["q_table"].get((state_key, expert), 1.0)

def oracle_get_weights(colors, regime):
    """
    Retorna dicionário de pesos multiplicadores para cada expert,
    baseado na Q-Table aprendida para o estado atual.
    
    Combinado com get_neural_weights() do sistema original para máximo poder.
    """
    state_key = _oracle_encode_state(colors, regime)

    weights = {}
    for expert in ORACLE_EXPERTS:
        q_val = _oracle_get_q(state_key, expert)

        # Epsilon-greedy: 8% das vezes usa peso padrão (exploração)
        if np.random.random() < ORACLE_EPSILON:
            weights[expert] = 1.0
        else:
            # Transforma Q-value em multiplicador [0.2, 2.5]
            weights[expert] = round(max(0.2, min(q_val, 2.5)), 3)

    with _oracle_lock:
        _oracle_state["last_state"]  = state_key
        _oracle_state["last_action"] = weights.copy()

    log.debug("🧠 ORACLE state=%s | top=%s",
              state_key[:2],
              sorted(weights.items(), key=lambda x: -x[1])[:3])

    return weights

def oracle_learn(won, colors, regime):
    """
    Atualiza Q-Table com o resultado do último sinal.
    Chamado dentro de update_neural_weights() após cada resultado.
    
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    """
    with _oracle_lock:
        last_state  = _oracle_state["last_state"]
        last_action = _oracle_state["last_action"]
        if last_state is None or last_action is None:
            return

    reward = 1.0 if won else -1.0
    new_state = _oracle_encode_state(colors, regime)

    with _oracle_lock:
        for expert, weight_used in last_action.items():
            old_q  = _oracle_state["q_table"].get((last_state, expert), 1.0)
            best_next_q = max(
                _oracle_state["q_table"].get((new_state, e), 1.0)
                for e in ORACLE_EXPERTS
            )
            # Bellman update
            new_q = old_q + ORACLE_ALPHA * (
                reward + ORACLE_GAMMA * best_next_q - old_q
            )
            _oracle_state["q_table"][(last_state, expert)] = round(
                max(0.1, min(new_q, 3.0)), 4
            )

        _oracle_state["episode_rewards"].append(reward)
        _oracle_state["total_updates"]  += 1

        if _oracle_state["total_updates"] % 50 == 0:
            recent_rewards = list(_oracle_state["episode_rewards"])[-50:]
            avg_r = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            log.info("🧠 ORACLE | Updates=%d | AvgReward(50)=%.2f | Q-States=%d",
                     _oracle_state["total_updates"], avg_r,
                     len(_oracle_state["q_table"]))

def oracle_save_state():
    """Persiste Q-Table em arquivo local."""
    os.makedirs(os.path.dirname(ORACLE_STATE_PATH), exist_ok=True)
    with _oracle_lock:
        data = {
            "q_table": {json.dumps(k): v for k, v in _oracle_state["q_table"].items()},
            "total_updates": _oracle_state["total_updates"],
        }
    with open(ORACLE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

def oracle_load_state():
    """Restaura Q-Table do banco de dados."""
    if not os.path.exists(ORACLE_STATE_PATH):
        return
    try:
        with open(ORACLE_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        with _oracle_lock:
            _oracle_state["q_table"] = {
                tuple(json.loads(k)): float(v)
                for k, v in data.get("q_table", {}).items()
                if isinstance(k, str)
            }
            _oracle_state["total_updates"] = int(data.get("total_updates", 0))
        log.info("🧠 ORACLE restaurado: %d Q-states", len(_oracle_state["q_table"]))
    except Exception as e:
        log.warning("ORACLE load falhou: %s", e)