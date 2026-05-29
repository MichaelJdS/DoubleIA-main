import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Importa o novo motor V4
try:
    from leviathan_v4_ultimate import LeviathanV4Ultimate, engine_v4
except ImportError:
    print("ERRO: leviathan_v4_ultimate.py não encontrado!")
    sys.exit(1)

# Mapeamento de cor inteira → string
COLOR_MAP = {0: 'white', 1: 'red', 2: 'black'}

class PantheonEngineV4:
    """
    Wrapper para integrar Leviathan V4 ao sistema existente.
    Mantém compatibilidade com coletor e dashboard.
    Schema real do banco: results_raw(id, round_id, color INT, roll INT, created_at, ...)
    """
    def __init__(self, db_path='blaze_double.db'):
        self.db_path = db_path
        self.v4_engine = LeviathanV4Ultimate()
        self.is_initialized = False
        self.history_buffer = []

    def initialize(self):
        """Carrega dados históricos e inicializa V4"""
        print("[PANTHEON V4] Carregando histórico para treino...")
        conn = sqlite3.connect(self.db_path)

        try:
            # Carrega até 5000 rodadas ordenadas por tempo
            query = ("SELECT roll, color, created_at "
                     "FROM results_raw "
                     "ORDER BY created_at ASC "
                     "LIMIT 5000")
            df = pd.read_sql_query(query, conn)

            if len(df) > 100:
                # Adapta ao schema esperado pelo Leviathan V4:
                #   'value'  → valor numérico (roll)
                #   'result' → alias de roll (usado em _train_base_experts)
                #   'color'  → mantém inteiro (usado em _mine_deep_patterns)
                data_for_v4 = df.rename(columns={'roll': 'value'})
                data_for_v4['result'] = data_for_v4['value']  # alias exigido pelo V4

                self.v4_engine.initialize(data_for_v4)
                self.is_initialized = True
                n = len(self.v4_engine.experts)
                print(f"[PANTHEON V4] Inicialização OK — {n} experts ativos com {len(df)} rodadas.")
            else:
                print("[PANTHEON V4] Dados insuficientes. Modo limitado ativado.")
                self.is_initialized = True

        except Exception as e:
            print(f"[PANTHEON V4] Erro na inicialização: {e}")
            self.is_initialized = True   # Fallback seguro

        finally:
            conn.close()

    def analyze_round(self, round_data):
        """
        Analisa uma nova rodada usando o motor V4.
        round_data aceita:
          {'value': int, 'color': 'red'|'black'|'white'|int, 'timestamp': ...}
        """
        # Normaliza color para string
        color = round_data.get('color', 1)
        if isinstance(color, int):
            round_data = dict(round_data)
            round_data['color'] = COLOR_MAP.get(color, 'red')

        # Atualiza buffer local
        self.history_buffer.append(round_data)
        if len(self.history_buffer) > 200:
            self.history_buffer.pop(0)

        if not self.is_initialized:
            return {'signal': 'WAIT', 'confidence': 0.0,
                    'regime': 'LOADING', 'message': 'Engine loading...'}

        # Chama análise V4 (retorna signal, confidence, regime, etc.)
        result = self.v4_engine.analyze(self.history_buffer)

        # Garante que 'regime' sempre existe no retorno
        result.setdefault('regime', 'N/A')

        # Log de decisão para debug
        if result['signal'] not in ('NO_BET', 'WAIT'):
            conf   = result['confidence']
            regime = result['regime']
            signal = result['signal']
            print(f"[SINAL V4] {signal} | Confiança: {conf:.2f} | Regime: {regime}")

        return result

    def get_stats(self):
        """Retorna estatísticas do motor V4"""
        return {
            'version':       '4.0 Ultimate',
            'experts_count': len(self.v4_engine.experts) + 1,   # +1 meta-learner
            'mode':          'High Frequency' if self.v4_engine.high_freq_mode else 'Conservative',
            'min_threshold': self.v4_engine.min_confidence,
        }

# ── Instância Singleton ──────────────────────────────────────
pantheon_engine = PantheonEngineV4()

# ── Funções de compatibilidade para o resto do sistema ───────
def run_analysis(round_data):
    return pantheon_engine.analyze_round(round_data)

def get_engine_stats():
    return pantheon_engine.get_stats()

if __name__ == "__main__":
    pantheon_engine.initialize()
    fake_data = {'value': 5, 'color': 'red', 'timestamp': datetime.now().isoformat()}
    res = pantheon_engine.analyze_round(fake_data)
    print(f"Teste Result: {res}")