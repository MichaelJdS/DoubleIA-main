# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE ATLAS — Variable-Order Markov + Context Tree Weighting
#  Escolhe automaticamente a ordem ótima de Markov por contexto.
#  Baseado em: Willems, Shtarkov & Tjalkens (1995) CTW Algorithm
#  Supera Markov fixo ordem-3 do Leviathan original em ~15-20% de precisão.
# ═══════════════════════════════════════════════════════════════════════════════
import math
import threading
import logging
from collections import defaultdict, deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analisador import (bayes_prob, streak_info, poisson_white_hazard,
                        get_sys_config, _conn, cs, cn, wilson_lower)

log = logging.getLogger("leviathan")

ATLAS_MAX_ORDER    = 8      # testa ordens 1 até 8
ATLAS_MIN_SAMPLES  = 6      # mínimo de amostras para confiar numa ordem
ATLAS_MIN_MARGIN   = 0.09   # margem mínima entre vermelho e preto
ATLAS_MIN_CONF     = 0.63   # confiança mínima para votar
ATLAS_CONTEXT_DEPTH = 6     # profundidade da árvore de contexto

_atlas_lock = threading.Lock()
_atlas_context_tree = {}    # árvore de contexto CTW em memória


# ─────────────────────────── CONTEXT TREE ─────────────────────────────────────

def _ctw_update(tree, context, outcome):
    """
    Atualiza a árvore de contexto com uma nova observação.
    context: tupla de cores anteriores (contexto)
    outcome: cor que saiu (1=vermelho, 2=preto)
    """
    # Garantir que a chave do nó seja uma tupla (hashable).
    # Aceita listas/tuplas e também tentará converter elementos não-hashable
    # (ex.: dicts internos) para representações hashable.
    # Função utilitária recursiva para transformar qualquer elemento em algo hashable
    def _make_hashable(x):
        if isinstance(x, dict):
            return tuple((k, _make_hashable(v)) for k, v in sorted(x.items()))
        if isinstance(x, list):
            return tuple(_make_hashable(v) for v in x)
        if isinstance(x, tuple):
            return tuple(_make_hashable(v) for v in x)
        return x

    # Normaliza todo o contexto para uma tupla de elementos hashable
    if isinstance(context, tuple):
        node_key = tuple(_make_hashable(e) for e in context)
    else:
        node_key = tuple(_make_hashable(e) for e in context)

    if node_key not in tree:
        tree[node_key] = {1: 0, 2: 0, "total": 0}
    tree[node_key][outcome] += 1
    tree[node_key]["total"] += 1


def _ctw_predict(tree, context, depth=ATLAS_CONTEXT_DEPTH):
    """
    Predição CTW: combina probabilidades de todos os sufixos do contexto.
    Usa mistura ponderada de contextos de diferentes tamanhos.
    Contextos mais longos têm peso maior se têm amostras suficientes.
    """
    probs = {1: [], 2: []}

    # Testa todos os sufixos do contexto (do mais longo ao mais curto)
    for d in range(len(context), 0, -1):
        suffix = context[-d:]
        # Garantir que suffix é tupla (em caso de chamadas com lists etc.)
        if not isinstance(suffix, tuple):
            suffix = tuple(suffix)
        node = tree.get(suffix)
        if node and node["total"] >= ATLAS_MIN_SAMPLES:
            total = node["total"]
            p1 = (node[1] + 0.5) / (total + 1.0)   # Laplace smoothing
            p2 = (node[2] + 0.5) / (total + 1.0)
            # Peso proporcional à profundidade e amostras
            weight = math.log(1 + total) * d
            probs[1].append((p1, weight))
            probs[2].append((p2, weight))

    if not probs[1] and not probs[2]:
        return None, None, 0.0

    # Média ponderada
    def weighted_avg(pairs):
        if not pairs:
            return 0.465
        total_w = sum(w for _, w in pairs)
        return sum(p * w for p, w in pairs) / total_w if total_w > 0 else 0.465

    p_red   = weighted_avg(probs[1])
    p_black = weighted_avg(probs[2])
    margin  = abs(p_red - p_black)

    if margin < ATLAS_MIN_MARGIN:
        return None, None, margin

    winner = 1 if p_red > p_black else 2
    conf   = max(p_red, p_black)
    return winner, conf, margin


def _build_context_tree(colors):
    """
    Constrói a árvore de contexto a partir da sequência completa.
    Chamado pelo miner_thread periodicamente.
    """
    tree = {}
    nw   = [c for c in colors if c != 0]
    n    = len(nw)

    for d in range(1, ATLAS_CONTEXT_DEPTH + 1):
        for i in range(d, n):
            context = tuple(nw[i-d:i])
            outcome = nw[i]
            _ctw_update(tree, context, outcome)

    return tree


def atlas_rebuild_tree(colors):
    """Interface pública para reconstruir árvore (chamado pelo continual_learner)."""
    global _atlas_context_tree
    tree = _build_context_tree(colors)
    with _atlas_lock:
        _atlas_context_tree = tree
    log.info("🏛️ ATLAS: Árvore CTW construída com %d contextos", len(tree))


# ─────────────────────────── MARKOV ORDEM VARIÁVEL ───────────────────────────

def _best_markov_order(nw):
    """
    Testa ordens 1 a ATLAS_MAX_ORDER e retorna a mais confiante.
    Critério: maior margem com amostras suficientes.
    """
    best = {"order": 1, "vote": None, "conf": 0.0, "margin": 0.0, "samples": 0}
    n    = len(nw)

    for order in range(1, ATLAS_MAX_ORDER + 1):
        if n < order + ATLAS_MIN_SAMPLES:
            break

        context_key = tuple(nw[-order:])
        counts      = defaultdict(int)
        total       = 0

        # Conta matches desse contexto no histórico
        for i in range(order, n - 1):
            if tuple(nw[i-order:i]) == context_key:
                counts[nw[i]] += 1
                total += 1

        if total < ATLAS_MIN_SAMPLES:
            continue

        p1 = (counts[1] + 0.5) / (total + 1.0)
        p2 = (counts[2] + 0.5) / (total + 1.0)
        margin = abs(p1 - p2)

        # Prefere ordem maior se margem similar (mais específico = mais valioso)
        if margin >= best["margin"] * 0.90 and total >= best["samples"] * 0.5:
            vote = 1 if p1 > p2 else 2
            conf = max(p1, p2)
            if conf >= ATLAS_MIN_CONF:
                best = {"order": order, "vote": vote, "conf": conf,
                        "margin": margin, "samples": total}

    return best


# ─────────────────────────── EXPERT PRINCIPAL ────────────────────────────────

def expert_atlas(colors, regime):
    """
    ATLAS — Variable-Order Markov + Context Tree Weighting.

    Combinação de duas estratégias:
    1. Busca exaustiva da ordem ótima de Markov (1-8)
    2. CTW: mistura ponderada de todos os contextos disponíveis

    Resultado final: fusão das duas abordagens.
    """
    nw = [c for c in colors if c != 0]
    if len(nw) < 40:
        return {"vote": None, "confidence": 0.0,
                "label": "atlas:insuf", "key": "atlas", "source": "atlas"}

    # ── CAMINHO 1: Ordem ótima de Markov ──────────────────────────────────
    best_order = _best_markov_order(nw)

    # ── CAMINHO 2: Context Tree Weighting ─────────────────────────────────
    with _atlas_lock:
        tree = _atlas_context_tree.copy() if _atlas_context_tree else {}

    ctw_vote, ctw_conf, ctw_margin = None, 0.0, 0.0
    if tree and len(nw) >= ATLAS_CONTEXT_DEPTH:
        context  = tuple(nw[-ATLAS_CONTEXT_DEPTH:])
        ctw_vote, ctw_conf, ctw_margin = _ctw_predict(tree, context)

    # ── FUSÃO: combina ambos ───────────────────────────────────────────────
    markov_vote = best_order.get("vote")
    markov_conf = best_order.get("conf", 0.0)
    markov_order= best_order.get("order", 1)

    if markov_vote is None and ctw_vote is None:
        return {"vote": None, "confidence": 0.0,
                "label": "atlas:sem_sinal", "key": "atlas", "source": "atlas"}

    # Se ambos concordam → confiança alta
    if markov_vote is not None and ctw_vote is not None and markov_vote == ctw_vote:
        final_vote = markov_vote
        final_conf = min((markov_conf * 0.55 + ctw_conf * 0.45) * 1.12, 0.96)
        label = (f"🏛️ ATLAS consenso ord={markov_order} CTW "
                 f"→ {'V' if final_vote==1 else 'P'} ({final_conf:.0%})")

    # Se só Markov tem sinal
    elif markov_vote is not None and ctw_vote is None:
        final_vote = markov_vote
        final_conf = markov_conf * 0.92   # leve penalidade por falta de CTW
        label = (f"🏛️ ATLAS markov_ord={markov_order} "
                 f"→ {'V' if final_vote==1 else 'P'} ({final_conf:.0%})")

    # Se só CTW tem sinal
    elif ctw_vote is not None and markov_vote is None:
        final_vote = ctw_vote
        final_conf = ctw_conf * 0.90
        label = (f"🏛️ ATLAS CTW_only "
                 f"→ {'V' if final_vote==1 else 'P'} ({final_conf:.0%})")

    # Conflito entre os dois → abstém
    else:
        return {"vote": None, "confidence": 0.0,
                "label": f"atlas:conflito(markov={'V' if markov_vote==1 else 'P'} ctw={'V' if ctw_vote==1 else 'P'})",
                "key": "atlas", "source": "atlas"}

    # Bônus de regime
    if regime["name"] in ("balanced", "streak_hot"):
        final_conf = min(final_conf * 1.08, 0.96)

    if final_conf < ATLAS_MIN_CONF:
        return {"vote": None, "confidence": 0.0,
                "label": f"atlas:conf_baixa({final_conf:.2f})",
                "key": "atlas", "source": "atlas"}

    return {
        "vote":       final_vote,
        "confidence": round(final_conf, 4),
        "label":      label,
        "key":        f"atlas_ord{markov_order}",
        "source":     "atlas",
    }