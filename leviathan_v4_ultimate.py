# ═══════════════════════════════════════════════════════════════════
#  LEVIATHAN v4.0 ULTIMATE — IMPLEMENTACAO COMPLETA CORRIGIDA
#  Fix 1: Target multiclasse (vermelho/preto/branco)
#  Fix 2: Features de cor reais (lags, streaks, freq, white-hazard)
#  Fix 3: Threshold menos punitivo em MEAN_REV
#  Fix 4: Meta-learner consistente treino/inferencia
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Mapeamentos globais ─────────────────────────────────────────
COLOR_INT = {'white': 0, 'red': 1, 'black': 2}
COLOR_STR = {0: 'white', 1: 'red', 2: 'black'}
N_CLASSES  = 3   # branco=0, vermelho=1, preto=2


class LeviathanV4Ultimate:
    """
    LEVIATHAN v4.0 ULTIMATE — Sistema Completo e Corrigido
    ────────────────────────────────────────────────────────
    14 experts multiclasse treinados com features de cor reais
    XGBoost Meta-Learner via out-of-fold stacking real
    Isotonic Calibration por classe
    Hurst Exponent + FFT para tendencias/ciclos
    Deep Pattern Mining (10 rodadas) com lookup probabilistico
    Threshold dinamico regime/horario/volatilidade (ajustado)
    Volume Spike via gaps temporais
    High-Frequency Mode (threshold 0.55)
    25 features: valor + cor + lags + streaks + entropia
    """

    def __init__(self):
        self.experts        = {}      # name -> modelo sklearn
        self.meta_learner   = None    # XGBoost stacker nível 2
        self.calibrators    = {}      # classe -> CalibratedClassifierCV
        self.pattern_db     = {}      # seq_tuple -> {0:p, 1:p, 2:p}
        self._n_expert_feat = 0       # dimensao do vetor level-2 (treino)

        self.min_confidence    = 0.55
        self.high_freq_mode    = True
        self.deep_search_depth = 10
        self._trained          = False

    # ────────────────────────────────────────────────────────────
    #  INICIALIZACAO
    # ────────────────────────────────────────────────────────────
    def initialize(self, historical_data):
        print("[LEVIATHAN V4] Iniciando treinamento completo...")
        df = self._prepare_df(historical_data)
        if len(df) < 60:
            print("[LEVIATHAN V4] Dados insuficientes (min 60 rodadas).")
            return

        X, y = self._build_dataset(df)
        print(f"[LEVIATHAN V4] Dataset: {X.shape[0]} amostras x {X.shape[1]} features | classes={np.unique(y)}")

        self._train_experts(X, y)
        self._train_meta_learner(X, y)
        self._train_calibrators(X, y)
        self._mine_patterns(df)

        self._trained = True
        print(f"[LEVIATHAN V4] Pronto. {len(self.experts)} experts + Meta + Calibradores ativos.")

    # ────────────────────────────────────────────────────────────
    #  PREPARACAO DOS DADOS
    # ────────────────────────────────────────────────────────────
    def _prepare_df(self, data):
        df = data.copy()
        # Normaliza coluna de valor
        if 'value' not in df.columns and 'roll' in df.columns:
            df['value'] = df['roll']
        # Normaliza color para int
        if 'color' in df.columns:
            if df['color'].dtype == object:
                df['color'] = df['color'].map(COLOR_INT).fillna(1).astype(int)
            else:
                df['color'] = df['color'].fillna(1).astype(int)
        return df.dropna(subset=['value', 'color']).reset_index(drop=True)

    def _build_dataset(self, df):
        """
        Constrói X (features) e y (cor alvo) com janela deslizante.
        Usa os 50 registros anteriores para cada ponto.
        """
        vals   = df['value'].values.astype(float)
        colors = df['color'].values.astype(int)
        X, y   = [], []
        window = 50

        for i in range(window, len(df)):
            hist = self._make_history(vals, colors, i, window)
            feat = self._extract_features(hist).flatten()
            X.append(feat)
            y.append(int(colors[i]))

        return np.array(X, dtype=float), np.array(y, dtype=int)

    def _make_history(self, vals, colors, end_idx, window):
        """Monta lista de dicts para _extract_features."""
        start = max(0, end_idx - window)
        return [
            {'value': float(vals[j]),
             'color': COLOR_STR.get(int(colors[j]), 'red')}
            for j in range(start, end_idx)
        ]

    # ────────────────────────────────────────────────────────────
    #  EXTRACAO DE FEATURES — 25 features fixas
    # ────────────────────────────────────────────────────────────
    def _extract_features(self, history):
        """
        25 features fixas:
        [0-7]   media/std por janela (5,10,20,50) do valor numérico
        [8]     Hurst Exponent
        [9-10]  FFT: frequência e amplitude dominantes
        [11]    autocorrelação lag-1 dos valores
        [12-14] freq relativa das 3 cores (últimas 30)
        [15-17] freq relativa das 3 cores (últimas 10)
        [18]    streak atual (comprimento)
        [19]    cor do streak atual (0/1/2)
        [20]    volatilidade relativa
        [21]    entropia de Shannon (últimas 30)
        [22-24] lags da cor: t-1, t-2, t-3 (one-hot → 3 valores)
        """
        df   = pd.DataFrame(history)
        vals = df['value'].values.astype(float) if 'value' in df.columns else np.array([0.])

        feat = []

        # [0-7] media/std por janela
        for w in [5, 10, 20, 50]:
            tail = vals[-w:] if len(vals) >= w else vals
            feat.append(float(np.mean(tail)) if len(tail) > 0 else 0.)
            feat.append(float(np.std(tail))  if len(tail) > 1 else 0.)

        # [8] Hurst
        feat.append(self._hurst(vals))

        # [9-10] FFT
        ff, fa = self._fft_features(vals)
        feat.append(ff); feat.append(fa)

        # [11] autocorrelação lag-1
        if len(vals) > 2:
            ac = float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
            feat.append(0. if np.isnan(ac) else ac)
        else:
            feat.append(0.)

        # [12-17] distribuição de cores em 2 janelas
        colors_raw = df['color'].tolist() if 'color' in df.columns else []
        c_int = [COLOR_INT.get(c, c) if isinstance(c, str) else int(c)
                 for c in colors_raw]

        for window in [30, 10]:
            chunk = c_int[-window:] if len(c_int) >= window else c_int
            n = max(len(chunk), 1)
            for cls in [0, 1, 2]:
                feat.append(sum(1 for c in chunk if c == cls) / n)

        # [18-19] streak atual
        streak_len, streak_col = self._streak(c_int)
        feat.append(float(streak_len))
        feat.append(float(streak_col))

        # [20] volatilidade relativa
        m = float(np.mean(np.abs(vals))) if len(vals) > 0 else 1.
        feat.append(float(np.std(vals)) / max(m, 0.001))

        # [21] entropia de Shannon
        feat.append(self._entropy(c_int[-30:] if len(c_int) >= 30 else c_int))

        # [22-24] lags de cor: t-1, t-2, t-3 normalizados
        for lag in [1, 2, 3]:
            if len(c_int) >= lag:
                feat.append(float(c_int[-lag]) / 2.)   # normaliza para [0,1]
            else:
                feat.append(0.5)

        assert len(feat) == 25, f"Feature dim errada: {len(feat)}"
        return np.array(feat, dtype=float).reshape(1, -1)

    # ────────────────────────────────────────────────────────────
    #  FEATURES AUXILIARES
    # ────────────────────────────────────────────────────────────
    def _hurst(self, series):
        n = len(series)
        if n < 10: return 0.5
        lags = range(2, min(20, n // 2))
        tau  = [np.sqrt(np.std(np.subtract(series[l:], series[:-l])) + 1e-9)
                for l in lags]
        try:
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return float(np.clip(poly[0], 0., 1.))
        except Exception:
            return 0.5

    def _fft_features(self, series):
        if len(series) < 8: return 0., 0.
        try:
            f   = np.abs(fft(series - np.mean(series)))
            f   = f[1:len(f)//2]
            idx = int(np.argmax(f))
            return float((idx+1) / len(series)), float(f[idx] / (len(series)+1e-9))
        except Exception:
            return 0., 0.

    def _streak(self, colors):
        if not colors: return 0, 1
        last = colors[-1]; cnt = 0
        for c in reversed(colors):
            if c == last: cnt += 1
            else: break
        return cnt, last

    def _entropy(self, colors):
        if not colors: return 1.
        try:
            _, counts = np.unique(colors, return_counts=True)
            p = counts / counts.sum()
            return float(-np.sum(p * np.log2(p + 1e-9)))
        except Exception:
            return 1.

    def _pattern_key(self, history):
        c = [COLOR_INT.get(h.get('color','red'), 1)
             if isinstance(h.get('color'), str) else int(h.get('color', 1))
             for h in history[-self.deep_search_depth:]]
        return tuple(c) if len(c) == self.deep_search_depth else None

    # ────────────────────────────────────────────────────────────
    #  14 EXPERTS MULTICLASSE
    # ────────────────────────────────────────────────────────────
    def _train_experts(self, X, y):
        self.experts = {
            # Estatísticos
            'stat_lr_l1':     LogisticRegression(max_iter=1000, C=0.3,
                                                  penalty='l1', solver='saga',
                                                  multi_class='multinomial'),
            'stat_lr_l2':     LogisticRegression(max_iter=1000, C=1.0,
                                                  multi_class='multinomial'),
            'stat_gb':        GradientBoostingClassifier(n_estimators=80,
                                                          max_depth=3,
                                                          learning_rate=0.05),
            # Tree-Based
            'tree_rf_deep':   RandomForestClassifier(n_estimators=150, max_depth=8,
                                                      min_samples_leaf=5,
                                                      class_weight='balanced'),
            'tree_rf_sha':    RandomForestClassifier(n_estimators=100, max_depth=4,
                                                      min_samples_leaf=10,
                                                      class_weight='balanced'),
            'tree_extra':     ExtraTreesClassifier(n_estimators=100, max_depth=6,
                                                    class_weight='balanced'),
            'tree_xgb_fast':  xgb.XGBClassifier(n_estimators=80, max_depth=3,
                                                  learning_rate=0.1,
                                                  num_class=N_CLASSES,
                                                  objective='multi:softprob',
                                                  eval_metric='mlogloss',
                                                  use_label_encoder=False),
            'tree_xgb_deep':  xgb.XGBClassifier(n_estimators=150, max_depth=5,
                                                  learning_rate=0.05,
                                                  num_class=N_CLASSES,
                                                  objective='multi:softprob',
                                                  eval_metric='mlogloss',
                                                  use_label_encoder=False),
            'tree_lgb_fast':  lgb.LGBMClassifier(n_estimators=80, max_depth=3,
                                                   learning_rate=0.1,
                                                   num_class=N_CLASSES,
                                                   objective='multiclass',
                                                   verbose=-1),
            'tree_lgb_deep':  lgb.LGBMClassifier(n_estimators=150, max_depth=5,
                                                   learning_rate=0.05,
                                                   num_class=N_CLASSES,
                                                   objective='multiclass',
                                                   verbose=-1),
            # Especialistas
            'regime_mom':     LogisticRegression(max_iter=500, C=0.1,
                                                  multi_class='multinomial'),
            'regime_rev':     LogisticRegression(max_iter=500, C=0.1,
                                                  multi_class='multinomial'),
            'knn_local':      KNeighborsClassifier(n_neighbors=15,
                                                    metric='manhattan'),
            'svm_rbf':        SVC(kernel='rbf', probability=True, C=1.0,
                                   gamma='scale', decision_function_shape='ovr'),
        }
        trained = 0
        for name, m in self.experts.items():
            try:
                m.fit(X, y)
                trained += 1
            except Exception as e:
                print(f"  [WARN] Expert {name}: {e}")
        print(f"[LEVIATHAN V4] {trained}/14 experts treinados.")

    # ────────────────────────────────────────────────────────────
    #  META-LEARNER REAL (OUT-OF-FOLD STACKING)
    # ────────────────────────────────────────────────────────────
    def _train_meta_learner(self, X, y):
        """
        Gera out-of-fold predict_proba de cada expert (3 colunas cada).
        Concatena → X_meta shape (n, n_experts*3).
        Treina XGBoost multiclasse como stacker de nível 2.
        """
        print("[LEVIATHAN V4] Treinando Meta-Learner (Neural Stacking real)...")
        skf     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_all = []

        for name, model in self.experts.items():
            oof = np.zeros((len(y), N_CLASSES))
            for tr_idx, val_idx in skf.split(X, y):
                try:
                    model.fit(X[tr_idx], y[tr_idx])
                    proba = model.predict_proba(X[val_idx])
                    if proba.shape[1] == N_CLASSES:
                        oof[val_idx] = proba
                    else:
                        oof[val_idx, 1] = proba[:, 1]   # fallback binário
                except Exception:
                    oof[val_idx] = 1/N_CLASSES
            oof_all.append(oof)

        # Re-treina experts com todos os dados
        for name, model in self.experts.items():
            try: model.fit(X, y)
            except Exception: pass

        X_meta = np.hstack(oof_all)   # (n_samples, n_experts * 3)
        self._n_expert_feat = X_meta.shape[1]

        self.meta_learner = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            num_class=N_CLASSES, objective='multi:softprob',
            eval_metric='mlogloss', use_label_encoder=False
        )
        try:
            self.meta_learner.fit(X_meta, y)
            acc = (self.meta_learner.predict(X_meta) == y).mean()
            print(f"[LEVIATHAN V4] Meta-Learner OK | Acc treino: {acc:.2%} | "
                  f"features nivel-2: {self._n_expert_feat}")
        except Exception as e:
            print(f"[LEVIATHAN V4] Meta-Learner erro: {e}")
            self.meta_learner = None

    # ────────────────────────────────────────────────────────────
    #  CALIBRADORES ISOTONICOS POR CLASSE
    # ────────────────────────────────────────────────────────────
    def _train_calibrators(self, X, y):
        print("[LEVIATHAN V4] Calibrando probabilidades (Isotonic por classe)...")
        best_base = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            num_class=N_CLASSES, objective='multi:softprob',
            eval_metric='mlogloss', use_label_encoder=False
        )
        # Calibração multiclasse: uma instância CalibratedClassifierCV
        self.calibrator = CalibratedClassifierCV(best_base, method='isotonic', cv=3)
        try:
            self.calibrator.fit(X, y)
            print("[LEVIATHAN V4] Calibrador Isotonic OK.")
        except Exception as e:
            print(f"[LEVIATHAN V4] Calibrador erro: {e}")
            self.calibrator = None

    # ────────────────────────────────────────────────────────────
    #  DEEP PATTERN MINING
    # ────────────────────────────────────────────────────────────
    def _mine_patterns(self, df):
        colors = df['color'].astype(int).tolist()
        n = self.deep_search_depth
        db = {}
        for i in range(n, len(colors)):
            seq = tuple(colors[i-n:i])
            nxt = colors[i]
            db.setdefault(seq, {0:0, 1:0, 2:0})
            db[seq][nxt] += 1
        self.pattern_db = {}
        for seq, cnts in db.items():
            total = sum(cnts.values())
            if total >= 3:
                self.pattern_db[seq] = {k: v/total for k, v in cnts.items()}
        print(f"[LEVIATHAN V4] Deep Mining: {len(self.pattern_db)} padroes "
              f"(janela={n} rodadas, suporte>=3).")

    # ────────────────────────────────────────────────────────────
    #  ANALISE EM TEMPO REAL
    # ────────────────────────────────────────────────────────────
    def analyze(self, current_history):
        """
        Analise completa.
        Retorna dict: signal, confidence, color_probs, regime, votes, ...
        """
        if len(current_history) < 20:
            return {'signal': 'WAIT', 'confidence': 0.,
                    'regime': 'LOADING', 'color_probs': {}}

        features = self._extract_features(current_history)   # (1, 25)
        regime   = self._detect_regime(current_history)

        # ── Coleta probabilidades de cada expert (3 classes) ──
        all_probas = []   # list of arrays shape (3,)

        for name, model in self.experts.items():
            try:
                p = model.predict_proba(features)[0]
                if len(p) == N_CLASSES:
                    all_probas.append(p)
            except Exception:
                continue

        # ── Meta-Learner nivel-2 ──
        if self.meta_learner and all_probas:
            try:
                meta_input = np.hstack(all_probas).reshape(1, -1)
                expected   = self._n_expert_feat
                if meta_input.shape[1] < expected:
                    meta_input = np.pad(meta_input,
                                        ((0,0),(0, expected - meta_input.shape[1])),
                                        constant_values=1/N_CLASSES)
                elif meta_input.shape[1] > expected:
                    meta_input = meta_input[:, :expected]
                meta_p = self.meta_learner.predict_proba(meta_input)[0]
                if len(meta_p) == N_CLASSES:
                    all_probas.append(meta_p)
            except Exception:
                pass

        # ── Calibrador Isotonic ──
        if self.calibrator:
            try:
                cal_p = self.calibrator.predict_proba(features)[0]
                if len(cal_p) == N_CLASSES:
                    all_probas.append(cal_p)
            except Exception:
                pass

        # ── Deep Pattern Mining ──
        pattern_bonus = None
        key = self._pattern_key(current_history)
        if key and key in self.pattern_db:
            pdb = self.pattern_db[key]
            pattern_bonus = np.array([pdb.get(0,0.), pdb.get(1,0.), pdb.get(2,0.)])
            if max(pattern_bonus) >= 0.55:
                all_probas.append(pattern_bonus)

        if not all_probas:
            return {'signal': 'NO_BET', 'confidence': 0.,
                    'regime': regime, 'color_probs': {}}

        # ── Fusão: media ponderada das probabilidades ──
        stacked   = np.vstack(all_probas)            # (n_sources, 3)
        avg_proba = stacked.mean(axis=0)              # (3,)

        best_class = int(np.argmax(avg_proba))
        best_conf  = float(avg_proba[best_class])

        # ── Threshold dinamico ──
        threshold = self._dynamic_threshold(regime, current_history)

        # ── Volume Spike ──
        volume_ok = not self._volume_spike(current_history)

        # ── Decisão final ──
        signal = 'NO_BET'
        if best_conf >= threshold and volume_ok:
            # Exige margem mínima sobre segunda melhor
            sorted_p = np.sort(avg_proba)[::-1]
            margin   = float(sorted_p[0] - sorted_p[1])
            if margin >= 0.05:   # consenso mínimo
                if   best_class == 1: signal = 'RED'
                elif best_class == 2: signal = 'BLACK'
                # branco: nunca aposta diretamente

        return {
            'signal':        signal,
            'confidence':    best_conf,
            'color_probs':   {COLOR_STR[i]: float(avg_proba[i])
                              for i in range(N_CLASSES)},
            'threshold_used': threshold,
            'regime':        regime,
            'total_experts': len(all_probas),
            'pattern_found': key is not None and key in self.pattern_db,
            'volume_ok':     volume_ok,
            'hurst':         float(self._hurst(
                                np.array([h['value'] for h in current_history],
                                         dtype=float))),
        }

    # ────────────────────────────────────────────────────────────
    #  DETECCAO DE REGIME
    # ────────────────────────────────────────────────────────────
    def _detect_regime(self, history):
        if len(history) < 20: return 'UNKNOWN'
        vals = np.array([h['value'] for h in history], dtype=float)
        ret  = np.diff(vals)
        if len(ret) < 2: return 'UNKNOWN'
        ac = float(np.corrcoef(ret[:-1], ret[1:])[0, 1])
        if   ac >  0.25: return 'MOMENTUM'
        elif ac < -0.25: return 'MEAN_REV'
        return 'RANDOM_WALK'

    # ────────────────────────────────────────────────────────────
    #  THRESHOLD DINAMICO (FIX 3 — menos punitivo)
    # ────────────────────────────────────────────────────────────
    def _dynamic_threshold(self, regime, history):
        base = 0.55 if self.high_freq_mode else 0.65

        # Ajuste por regime (CORRIGIDO: antes MEAN_REV +0.12, agora menor)
        if   regime == 'RANDOM_WALK': base += 0.05   # era +0.12
        elif regime == 'MOMENTUM':    base -= 0.05
        elif regime == 'MEAN_REV':    base -= 0.02   # era penalidade, agora bônus

        # Ajuste por horário
        hour = pd.Timestamp.now().hour
        if   0 <= hour <= 5:   base += 0.04
        elif 9 <= hour <= 11:  base -= 0.02
        elif 20 <= hour <= 23: base += 0.02

        # Ajuste por volatilidade
        if len(history) >= 20:
            vals    = np.array([h['value'] for h in history[-20:]], dtype=float)
            vol_rel = float(np.std(vals)) / max(float(np.mean(np.abs(vals))), .001)
            if vol_rel > 1.5: base += 0.03

        return float(np.clip(base, 0.50, 0.85))

    # ────────────────────────────────────────────────────────────
    #  VOLUME SPIKE (anti-manipulação via gaps temporais)
    # ────────────────────────────────────────────────────────────
    def _volume_spike(self, history):
        ts = [h.get('timestamp') for h in history if h.get('timestamp')]
        if len(ts) < 5: return False
        try:
            t    = pd.to_datetime(ts)
            gaps = t.diff().dropna().dt.total_seconds().values
            if len(gaps) < 4: return False
            avg  = float(np.mean(gaps[:-1]))
            std  = float(np.std(gaps[:-1]))
            return std > 0 and (avg - gaps[-1]) > 2.5 * std
        except Exception:
            return False


# Instância global
engine_v4 = LeviathanV4Ultimate()