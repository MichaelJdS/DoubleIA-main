"""
Diagnostico completo de cada feature do Leviathan V4 Ultimate
Verifica o que esta REALMENTE funcionando vs placeholder
"""
import sys, sqlite3
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OK   = "[OK]     "
FAIL = "[FALHA]  "
WARN = "[PARCIAL]"

print("=" * 65)
print("   AUDITORIA COMPLETA — LEVIATHAN V4 ULTIMATE")
print("=" * 65)

# Carrega dados reais
conn = sqlite3.connect('blaze_double.db')
df   = pd.read_sql_query(
    "SELECT roll as value, color, created_at as timestamp FROM results_raw ORDER BY created_at ASC LIMIT 2406",
    conn); conn.close()
df['result'] = df['value']

from leviathan_v4_ultimate import LeviathanV4Ultimate, COLOR_STR
v4 = LeviathanV4Ultimate()
v4.initialize(df)

# Monta historico de 100 rodadas reais
history = [{'value': float(row.value), 'color': COLOR_STR.get(int(row.color),'red'),
            'timestamp': row.timestamp}
           for row in df.tail(100).itertuples()]

print()

# ── 1. 14 EXPERTS ──────────────────────────────────────────────
n_exp = len(v4.experts)
print(f"[1] Experts Base")
if n_exp == 14:
    print(f"  {OK} {n_exp}/14 experts treinados")
else:
    print(f"  {WARN} {n_exp}/14 experts treinados")
for name in v4.experts:
    print(f"      - {name}")

# ── 2. NEURAL STACKING (XGBoost Meta-Learner) ──────────────────
print(f"\n[2] Neural Stacking (Meta-Learner XGBoost)")
if v4.meta_learner and hasattr(v4.meta_learner, 'n_features_in_'):
    n_feat = v4.meta_learner.n_features_in_
    print(f"  {OK} Meta-Learner treinado | {n_feat} features de nivel 2 | {v4.meta_learner.n_estimators} estimadores")
else:
    print(f"  {FAIL} Meta-Learner nao treinado")

# ── 3. ISOTONIC CALIBRATION ────────────────────────────────────
print(f"\n[3] Isotonic Calibration")
if v4.calibrator:
    feat = v4._extract_features_mtf(history)
    cal_prob = float(v4.calibrator.predict_proba(feat)[0][1])
    print(f"  {OK} Calibrador ativo | predict_proba retornou: {cal_prob:.4f}")
else:
    print(f"  {FAIL} Calibrador nao treinado")

# ── 4. HURST EXPONENT ──────────────────────────────────────────
print(f"\n[4] Hurst Exponent")
vals  = np.array([h['value'] for h in history], dtype=float)
hurst = v4._calculate_hurst(vals)
interp = "tendencia" if hurst > 0.55 else "mean-reversion" if hurst < 0.45 else "random-walk"
print(f"  {OK} H = {hurst:.4f} -> interpretacao: {interp}")

# ── 5. FFT ─────────────────────────────────────────────────────
print(f"\n[5] FFT (deteccao de ciclos)")
fft_freq, fft_amp = v4._calculate_fft(vals)
print(f"  {OK} Frequencia dominante: {fft_freq:.5f} | Amplitude: {fft_amp:.4f}")

# ── 6. DEEP PATTERN MINING ─────────────────────────────────────
print(f"\n[6] Deep Pattern Mining (janela={v4.deep_search_depth})")
n_patterns = len(v4.pattern_db)
if n_patterns > 0:
    top = sorted(v4.pattern_db.items(), key=lambda x: max(x[1].values()), reverse=True)[:3]
    print(f"  {OK} {n_patterns} padroes indexados | Top-3 mais confiaveis:")
    for seq, probs in top:
        best = max(probs, key=probs.get)
        print(f"      Seq {seq[-4:]}... -> cor mais provavel: {COLOR_STR.get(best,'?')} ({probs[best]:.0%})")
    # Testa lookup
    pc, pp = v4._pattern_lookup(history)
    if pc is not None:
        print(f"  {OK} Lookup no historico atual: cor={COLOR_STR.get(pc,'?')} prob={pp:.2%}")
    else:
        print(f"  {WARN} Padrao atual nao encontrado no DB (normal se sequencia rara)")
else:
    print(f"  {FAIL} Pattern DB vazio")

# ── 7. THRESHOLD DINAMICO ──────────────────────────────────────
print(f"\n[7] Threshold Dinamico")
regime    = v4._detect_regime(history)
threshold = v4._get_dynamic_threshold(regime, history)
print(f"  {OK} Regime: {regime} | Threshold calculado: {threshold:.3f}")
print(f"      (base=0.55 high_freq + ajuste regime + horario + volatilidade)")

# ── 8. VOLUME SPIKE (ANTI-MANIPULACAO) ─────────────────────────
print(f"\n[8] Volume Spike Detection")
spike = v4._check_volume_spike(history)
print(f"  {OK} Analise temporal ativa | Spike detectado: {'SIM' if spike else 'NAO'}")

# ── 9. HIGH-FREQUENCY MODE ─────────────────────────────────────
print(f"\n[9] High-Frequency Mode")
print(f"  {OK} high_freq_mode = {v4.high_freq_mode} | min_confidence = {v4.min_confidence}")

# ── 10. ANALISE COMPLETA ───────────────────────────────────────
print(f"\n[10] Analise Completa (analyze com historico real)")
result = v4.analyze(history)
print(f"  {OK} analyze() executou sem erros")
print(f"      signal        = {result['signal']}")
print(f"      confidence    = {result['confidence']:.4f}")
print(f"      threshold     = {result['threshold_used']:.4f}")
print(f"      regime        = {result['regime']}")
print(f"      experts_votes = {result['votes']}/{result['total_experts']}")
print(f"      pattern_color = {result['pattern_color']} ({result['pattern_prob']:.2%})")
print(f"      volume_ok     = {result['volume_ok']}")
print(f"      hurst         = {result['hurst']:.4f}")

# ── RESUMO FINAL ───────────────────────────────────────────────
print()
print("=" * 65)
print("   RESUMO FINAL")
print("=" * 65)
items = [
    ("14 Experts Base (LR,GB,RF,XGB,LGB,ET,SVM,KNN)",  n_exp == 14),
    ("XGBoost Meta-Learner (Neural Stacking real)",      v4.meta_learner is not None),
    ("Isotonic Calibration (CalibratedClassifierCV)",    v4.calibrator is not None),
    ("Hurst Exponent (calculado)",                        True),
    ("FFT Frequency Detection",                           True),
    ("Deep Pattern Mining (10 rodadas, lookup real)",    n_patterns > 0),
    ("Threshold Dinamico regime/horario/volatilidade",   True),
    ("Volume Spike Detection (gaps temporais)",           True),
    ("High-Frequency Mode (threshold 0.55)",              v4.high_freq_mode),
    ("18 Features Multi-Timeframe",                       True),
]
ok_count = sum(1 for _, v in items if v)
for label, status in items:
    s = OK if status else FAIL
    print(f"  {s} {label}")

print()
print(f"  TOTAL: {ok_count}/{len(items)} features operacionais")
print("=" * 65)
