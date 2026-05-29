# ═══════════════════════════════════════════════════════════════════
#  PANTHEON ENGINE — Sistema Híbrido
#  Combina os 6 agentes originais (SYBIL/CHAOS/HERMES/ORACLE/ATLAS/TITAN)
#  com o Leviathan V4 (14 experts + Meta-Learner + Isotonic Calibration)
#  Fusão final via Dempster-Shafer + Oracle Q-Learning
# ═══════════════════════════════════════════════════════════════════

import sqlite3
import threading
import logging
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger("pantheon_hybrid")

# ── Importa os 6 agentes originais ─────────────────────────────
try:
    from agents.agent_sybil  import expert_sybil
    from agents.agent_chaos  import expert_chaos
    from agents.agent_hermes import expert_hermes, _hermes_state
    from agents.agent_oracle import (oracle_get_weights, oracle_learn,
                                      oracle_save_state, oracle_load_state)
    from agents.agent_atlas  import expert_atlas, atlas_rebuild_tree
    from agents.agent_titan  import expert_titan, titan_rebuild
    _AGENTS_OK = True
except ImportError as e:
    log.warning("Agentes originais nao carregados: %s", e)
    _AGENTS_OK = False

# ── Importa Leviathan V4 ────────────────────────────────────────
try:
    from leviathan_v4_ultimate import LeviathanV4Ultimate, COLOR_STR, COLOR_INT
    _V4_OK = True
except ImportError as e:
    log.warning("Leviathan V4 nao carregado: %s", e)
    _V4_OK = False

# Peso relativo do V4 no ensemble (0.0 = só agentes, 1.0 = só V4)
V4_WEIGHT = 0.40   # 40% V4, 60% agentes originais

# Mapeamento regime string <-> dict (agentes originais exigem dict)
def _regime_dict(regime_str):
    return {"name": regime_str, "strength": 0.6}

def _detect_micro_regime(colors):
    """
    Detecta micro-regime a partir da lista de cores (int).
    Retorna dict compatível com os agentes originais.
    """
    if len(colors) < 10:
        return {"name": "unknown", "strength": 0.5}
    nw = [c for c in colors if c != 0]
    if not nw:
        return {"name": "white_storm", "strength": 0.9}

    # Calcula autocorrelação
    arr = np.array(nw[-40:], dtype=float)
    if len(arr) > 2:
        ret = np.diff(arr)
        if len(ret) > 1:
            ac = float(np.corrcoef(ret[:-1], ret[1:])[0, 1])
            if   ac >  0.3: name = "momentum"
            elif ac < -0.3: name = "mean_rev"
            else:           name = "balanced"
        else:
            name = "balanced"
    else:
        name = "balanced"

    # Streak
    streak = 0
    last   = nw[-1]
    for c in reversed(nw):
        if c == last: streak += 1
        else: break
    if streak >= 5:
        name = "streak_hot"

    return {"name": name, "strength": 0.6}


class PantheonHybridEngine:
    """
    Motor Híbrido: 6 Agentes Pantheon + Leviathan V4.
    
    Fluxo:
    1. Carrega dados históricos do banco → inicializa V4
    2. Para cada rodada:
       a. Consulta 6 agentes (votos + confiança)
       b. Consulta V4 (color_probs + confidence)
       c. Normaliza todos os votos para (cor, confiança)
       d. Pondera via Oracle Q-Learning
       e. Fusão Dempster-Shafer
       f. Decisão final
    """

    def __init__(self, db_path='blaze_double.db'):
        self.db_path        = db_path
        self.v4_engine      = LeviathanV4Ultimate() if _V4_OK else None
        self.is_initialized = False
        self.history_buffer = []   # lista de dicts {value, color, timestamp}
        self._lock          = threading.Lock()

    # ────────────────────────────────────────────────────────────
    #  INICIALIZACAO
    # ────────────────────────────────────────────────────────────
    def initialize(self):
        """Carrega histórico do banco e inicializa V4 + agentes."""
        print("[PANTHEON HYBRID] Inicializando sistema hibrido...")

        # ── V4 ──────────────────────────────────────────────────
        if _V4_OK and self.v4_engine:
            conn = sqlite3.connect(self.db_path)
            try:
                df = pd.read_sql_query(
                    "SELECT roll as value, color, created_at as timestamp "
                    "FROM results_raw ORDER BY created_at ASC LIMIT 5000",
                    conn
                )
                df['result'] = df['value']
                if len(df) >= 60:
                    self.v4_engine.initialize(df)
                    print(f"[PANTHEON HYBRID] V4 inicializado com {len(df)} rodadas.")
                else:
                    print("[PANTHEON HYBRID] Poucos dados para V4.")
            except Exception as e:
                print(f"[PANTHEON HYBRID] Erro ao inicializar V4: {e}")
            finally:
                conn.close()

        # ── Agentes originais: carrega Oracle + reconstrói árvores ──
        if _AGENTS_OK:
            try:
                oracle_load_state()
                print("[PANTHEON HYBRID] Oracle Q-Table carregado.")
            except Exception as e:
                print(f"[PANTHEON HYBRID] Oracle load erro: {e}")

        self.is_initialized = True
        print("[PANTHEON HYBRID] Sistema pronto.")

    # ────────────────────────────────────────────────────────────
    #  ANALISE DE RODADA
    # ────────────────────────────────────────────────────────────
    def analyze_round(self, round_data):
        """
        Analisa uma nova rodada.
        round_data: dict com 'value' (int), 'color' (str ou int),
                    'timestamp' (str, opcional)
        Retorna: dict com signal, confidence, regime, votos detalhados, etc.
        """
        # Normaliza cor para string e int
        color_raw = round_data.get('color', 1)
        if isinstance(color_raw, int):
            color_str = COLOR_STR.get(color_raw, 'red')
            color_int = color_raw
        else:
            color_str = str(color_raw)
            color_int = COLOR_INT.get(color_str, 1)

        entry = {
            'value':     int(round_data.get('value', round_data.get('roll', 0))),
            'color':     color_str,
            'color_int': color_int,
            'timestamp': round_data.get('timestamp', datetime.now().isoformat()),
        }

        with self._lock:
            self.history_buffer.append(entry)
            if len(self.history_buffer) > 300:
                self.history_buffer.pop(0)

        if not self.is_initialized:
            return self._no_signal('LOADING')

        colors_int = [h['color_int'] for h in self.history_buffer]
        regime     = _detect_micro_regime(colors_int)

        # ── Coleta votos dos 6 agentes ──────────────────────────
        agent_votes = self._collect_agent_votes(colors_int, regime)

        # ── Coleta análise do V4 ────────────────────────────────
        v4_result = self._collect_v4_vote()

        # ── Pesos Oracle Q-Learning ─────────────────────────────
        oracle_weights = {}
        if _AGENTS_OK:
            try:
                oracle_weights = oracle_get_weights(colors_int, regime)
            except Exception:
                pass

        # ── Fusão final ─────────────────────────────────────────
        return self._fuse(agent_votes, v4_result, oracle_weights, regime)

    # ────────────────────────────────────────────────────────────
    #  COLETA DE VOTOS DOS 6 AGENTES
    # ────────────────────────────────────────────────────────────
    def _collect_agent_votes(self, colors_int, regime):
        """Chama cada agente e retorna lista de {key, vote, confidence, label}."""
        votes = []
        if not _AGENTS_OK or len(colors_int) < 10:
            return votes

        agents = [
            ('sybil',  lambda: expert_sybil(colors_int, regime)),
            ('chaos',  lambda: expert_chaos(colors_int, regime)),
            ('hermes', lambda: expert_hermes(colors_int, regime)),
            ('atlas',  lambda: expert_atlas(colors_int, regime)),
            ('titan',  lambda: expert_titan(colors_int, regime)),
        ]

        for name, fn in agents:
            try:
                res = fn()
                if res and res.get('vote') is not None:
                    votes.append({
                        'key':        res.get('key', name),
                        'vote':       int(res['vote']),   # 1=vermelho, 2=preto
                        'confidence': float(res.get('confidence', 0.5)),
                        'label':      res.get('label', name),
                        'source':     'agent',
                    })
            except Exception as e:
                log.debug("Agente %s erro: %s", name, e)

        return votes

    # ────────────────────────────────────────────────────────────
    #  COLETA DO V4
    # ────────────────────────────────────────────────────────────
    def _collect_v4_vote(self):
        """Chama o V4 e retorna resultado normalizado."""
        if not _V4_OK or not self.v4_engine or not self.v4_engine._trained:
            return None
        if len(self.history_buffer) < 20:
            return None
        try:
            result = self.v4_engine.analyze(self.history_buffer)
            return result
        except Exception as e:
            log.debug("V4 erro: %s", e)
            return None

    # ────────────────────────────────────────────────────────────
    #  FUSAO DEMPSTER-SHAFER HIBRIDA
    # ────────────────────────────────────────────────────────────
    def _fuse(self, agent_votes, v4_result, oracle_weights, regime):
        """
        Combina votos dos agentes + V4 em uma decisão final.
        Retorna dict compatível com o resto do sistema.
        """
        # Acumula probabilidades por classe (vermelho=1, preto=2)
        prob_red   = []
        prob_black = []

        # ── Agentes (peso Oracle) ──
        for vote in agent_votes:
            key  = vote['key']
            conf = vote['confidence']
            w    = oracle_weights.get(key, 1.0)
            effective_conf = float(np.clip(conf * w, 0.01, 0.99))

            if vote['vote'] == 1:   # vermelho
                prob_red.append(effective_conf)
                prob_black.append(1.0 - effective_conf)
            elif vote['vote'] == 2:  # preto
                prob_black.append(effective_conf)
                prob_red.append(1.0 - effective_conf)

        # ── V4 (peso fixo V4_WEIGHT) ──
        v4_signal = None
        v4_conf   = 0.0
        if v4_result and v4_result.get('signal') not in ('NO_BET', 'WAIT', None):
            v4_signal = v4_result['signal']
            v4_conf   = float(v4_result.get('confidence', 0.5))
            cp        = v4_result.get('color_probs', {})

            # Usa as probabilidades brutas do V4 ponderadas por V4_WEIGHT
            p_red   = float(cp.get('red',   0.33)) * V4_WEIGHT + 0.33 * (1 - V4_WEIGHT)
            p_black = float(cp.get('black', 0.33)) * V4_WEIGHT + 0.33 * (1 - V4_WEIGHT)
            prob_red.append(p_red)
            prob_black.append(p_black)

        # ── Se nenhum voto, sem sinal ──
        if not prob_red and not prob_black:
            return self._no_signal(regime.get('name', 'unknown'), v4_result)

        # ── Media ponderada ──
        avg_red   = float(np.mean(prob_red))   if prob_red   else 0.33
        avg_black = float(np.mean(prob_black)) if prob_black else 0.33
        avg_white = float(np.clip(1.0 - avg_red - avg_black, 0., 1.))

        # ── Decisão ──
        best_color = max([(avg_red, 'RED', 1),
                          (avg_black, 'BLACK', 2),
                          (avg_white, 'NO_BET', 0)],
                         key=lambda x: x[0])
        signal     = best_color[1]
        confidence = best_color[0]

        # Threshold mínimo
        min_conf = 0.45 if _AGENTS_OK and agent_votes else 0.55
        if confidence < min_conf or signal == 'NO_BET':
            signal = 'NO_BET'

        # Log se houver sinal
        if signal != 'NO_BET':
            r = regime.get('name','?')
            print(f"[SINAL HYBRID] {signal} | Conf: {confidence:.2f} | "
                  f"Regime: {r} | Agentes: {len(agent_votes)} | V4: {v4_signal}")

        return {
            'signal':       signal,
            'confidence':   round(confidence, 4),
            'regime':       regime.get('name', 'unknown'),
            'prob_red':     round(avg_red,   4),
            'prob_black':   round(avg_black, 4),
            'prob_white':   round(avg_white, 4),
            'agent_votes':  len(agent_votes),
            'v4_signal':    v4_signal,
            'v4_conf':      round(v4_conf, 4),
            'v4_regime':    v4_result.get('regime', 'N/A') if v4_result else 'N/A',
        }

    def _no_signal(self, regime_name, v4_result=None):
        return {
            'signal':      'NO_BET',
            'confidence':  0.0,
            'regime':      regime_name,
            'prob_red':    0.33,
            'prob_black':  0.33,
            'prob_white':  0.34,
            'agent_votes': 0,
            'v4_signal':   None,
            'v4_conf':     0.0,
            'v4_regime':   v4_result.get('regime','N/A') if v4_result else 'N/A',
        }

    # ────────────────────────────────────────────────────────────
    #  APRENDIZADO (chamado após resultado real)
    # ────────────────────────────────────────────────────────────
    def learn(self, won, colors_int, regime_str):
        """Atualiza Oracle Q-Table com o resultado da aposta."""
        if _AGENTS_OK and colors_int:
            try:
                regime = _regime_dict(regime_str)
                oracle_learn(won=won, colors=colors_int, regime=regime)
                if len(colors_int) % 50 == 0:
                    oracle_save_state()
            except Exception as e:
                log.debug("Oracle learn erro: %s", e)

    def get_stats(self):
        """Estatísticas do motor híbrido."""
        v4_experts = len(self.v4_engine.experts) if (self.v4_engine and _V4_OK) else 0
        return {
            'version':       '4.0 Ultimate Hybrid',
            'agents_count':  6 if _AGENTS_OK else 0,
            'v4_experts':    v4_experts,
            'total_sources': (6 if _AGENTS_OK else 0) + (1 if _V4_OK else 0),
            'mode':          'High Frequency' if (self.v4_engine and
                              self.v4_engine.high_freq_mode) else 'Standard',
            'v4_weight':     V4_WEIGHT,
            'agents_ok':     _AGENTS_OK,
            'v4_ok':         _V4_OK,
        }


# ── Instância Singleton ─────────────────────────────────────────
pantheon_engine = PantheonHybridEngine()

# ── API de compatibilidade (mantém interface do sistema original) ─
def run_analysis(round_data):
    return pantheon_engine.analyze_round(round_data)

def get_engine_stats():
    return pantheon_engine.get_stats()

if __name__ == '__main__':
    pantheon_engine.initialize()
    fake = {'value': 7, 'color': 'red', 'timestamp': datetime.now().isoformat()}
    res  = pantheon_engine.analyze_round(fake)
    print('Teste Result:', res)