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

class PantheonEngineV4:
    """
    Wrapper para integrar Leviathan V4 ao sistema existente
    Mantém compatibilidade com coletor e dashboard
    """
    def __init__(self, db_path='blaze_data.db'):
        self.db_path = db_path
        self.v4_engine = LeviathanV4Ultimate()
        self.is_initialized = False
        self.history_buffer = []
        
    def initialize(self):
        """Carrega dados históricos e inicializa V4"""
        print("[PANTHEON V4] Carregando histórico para treino...")
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Carrega últimas 5000 rodadas para treino
            query = "SELECT * FROM results_raw ORDER BY created_at DESC LIMIT 5000"
            df = pd.read_sql_query(query, conn)
            
            if len(df) > 100:
                # Prepara dados para o V4
                # Nota: Adaptar colunas conforme seu schema real
                data_for_v4 = df.rename(columns={
                    'roll': 'value', 
                    'color': 'color' # Ajuste conforme necessário
                })
                
                self.v4_engine.initialize(data_for_v4)
                self.is_initialized = True
                print("[PANTHEON V4] Inicialização concluída com sucesso!")
            else:
                print("[PANTHEON V4] Dados insuficientes para treino completo. Modo limitado.")
                self.is_initialized = True # Continua mesmo com poucos dados
                
        except Exception as e:
            print(f"[PANTHEON V4] Erro na inicialização: {e}")
            self.is_initialized = True # Fallback seguro
            
        finally:
            conn.close()

    def analyze_round(self, round_data):
        """
        Analisa uma nova rodada usando o motor V4
        round_data: dict com informações da rodada atual
        """
        # Atualiza buffer local
        self.history_buffer.append(round_data)
        if len(self.history_buffer) > 200:
            self.history_buffer.pop(0)
            
        if not self.is_initialized:
            return {'signal': 'WAIT', 'confidence': 0.0, 'message': 'Engine loading...'}
            
        # Chama análise V4
        result = self.v4_engine.analyze(self.history_buffer)
        
        # Log de decisão para debug
        if result['signal'] != 'NO_BET':
            print(f"[SINAL V4] {result['signal']} | Confiança: {result['confidence']:.2f} | Regime: {result['regime']}")
            
        return result

    def get_stats(self):
        """Retorna estatísticas do motor V4"""
        return {
            'version': '4.0 Ultimate',
            'experts_count': len(self.v4_engine.experts) + 1, # +1 meta
            'mode': 'High Frequency' if self.v4_engine.high_freq_mode else 'Conservative',
            'min_threshold': self.v4_engine.min_confidence
        }

# Instância Singleton
pantheon_engine = PantheonEngineV4()

# Funções de compatibilidade para o resto do sistema
def run_analysis(round_data):
    return pantheon_engine.analyze_round(round_data)

def get_engine_stats():
    return pantheon_engine.get_stats()

if __name__ == "__main__":
    # Teste rápido
    pantheon_engine.initialize()
    fake_data = {'value': 5, 'color': 'red', 'timestamp': datetime.now()}
    res = pantheon_engine.analyze_round(fake_data)
    print(f"Teste Result: {res}")