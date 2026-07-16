"""财报公告日次日涨跌预测 — 建模全流程"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, roc_curve)
from sklearn.dummy import DummyClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import lightgbm as lgb
import shap
import os

os.chdir(r'C:\Users\RYAN\Desktop\BA工作坊\TASK5-1')
print("Working directory:", os.getcwd())

# ==========================================
# 1. DATA LOADING
# ==========================================
print("\n" + "="*60)
print("1. DATA LOADING")
print("="*60)

df = pd.read_csv('model_data_stock.csv')
print(f'Shape: {df.shape}')
print(f'Date range: {df["Date"].min()} ~ {df["Date"].max()}')
print(f'Unique stocks: {df["Code"].nunique()}')
print(f'Y distribution:')
print(df['Y'].value_counts(normalize=True))
print(f'Missing values: {df.isnull().sum().sum()}')

ID_COLS = ['Date', 'Code']
TARGET = 'Y'

VALUATION_FEATURES = [
    '企业倍数(EV除EBITDA)', '市净率PB(MRQ)',
    '市现率PCF(现金净流量TTM)', '市现率PCF(经营现金流TTM)',
    '市盈率PE(TTM)', '市盈率PE(TTM,扣除非经常性损益)',
    '市销率PS(TTM)', '股息率(近12个月)', 'MV'
]

GROWTH_FEATURES = [
    '净利润同比增长率', '净资产同比增长率',
    '利润总额(同比增长率)', '基本每股收益(同比增长率)',
    '总资产同比增长率', '现金净流量同比增长率',
    '营业利润(同比增长率)', '营业总收入(同比增长率)'
]

ALL_FEATURES = VALUATION_FEATURES + GROWTH_FEATURES
print(f'Valuation: {len(VALUATION_FEATURES)}, Growth: {len(GROWTH_FEATURES)}, Total: {len(ALL_FEATURES)}')

# ==========================================
# 2. FEATURE ENGINEERING - Winsorize
# ==========================================
print("\n" + "="*60)
print("2. WINSORIZE (1%/99%)")
print("="*60)

df_winsorized = df.copy()
winsor_stats = []

for col in ALL_FEATURES:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df_winsorized[col] = df[col].clip(lower=lower, upper=upper)
    winsor_stats.append({
        'feature': col, 'p1': round(lower, 4), 'p99': round(upper, 4),
        'original_min': round(df[col].min(), 4), 'original_max': round(df[col].max(), 4)
    })

winsor_df = pd.DataFrame(winsor_stats)
winsor_df.to_csv('winsorize_stats.csv', index=False, encoding='utf-8-sig')
print(f'Winsorize complete. {len(ALL_FEATURES)} features clipped.')

# ==========================================
# 3. COLLINEARITY ANALYSIS
# ==========================================
print("\n" + "="*60)
print("3. COLLINEARITY ANALYSIS")
print("="*60)

# Correlation heatmap
corr_matrix = df_winsorized[ALL_FEATURES].corr()
fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix (after Winsorize)', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: correlation_heatmap.png')

# High correlation pairs
high_corr_pairs = []
for i in range(len(ALL_FEATURES)):
    for j in range(i+1, len(ALL_FEATURES)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            high_corr_pairs.append((ALL_FEATURES[i], ALL_FEATURES[j], round(r, 3)))

print(f'High correlation pairs (|r| > 0.7): {len(high_corr_pairs)}')
for pair in high_corr_pairs:
    print(f'  {pair[0]} <-> {pair[1]}: r={pair[2]}')

# VIF analysis
X_vif = df_winsorized[ALL_FEATURES].copy()
X_vif = X_vif.assign(const=1.0)
vif_data = []
for i, col in enumerate(ALL_FEATURES):
    vif = variance_inflation_factor(X_vif.values, i)
    vif_data.append({'feature': col, 'VIF': round(vif, 2)})

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
vif_df.to_csv('vif_report.csv', index=False, encoding='utf-8-sig')
print('\nVIF Report:')
print(vif_df.to_string(index=False))
print(f'\nVIF > 10: {len(vif_df[vif_df["VIF"] > 10])}')
print(f'VIF > 5: {len(vif_df[vif_df["VIF"] > 5])}')

# Build feature set A (de-correlated)
cols_A = list(ALL_FEATURES)
removed_features = []
while True:
    X_check = df_winsorized[cols_A].copy()
    X_check = X_check.assign(const=1.0)
    vif_values = {}
    for i, col in enumerate(cols_A):
        vif_values[col] = variance_inflation_factor(X_check.values, i)
    max_col = max(vif_values, key=vif_values.get)
    max_vif = vif_values[max_col]
    if max_vif < 10:
        break
    cols_A.remove(max_col)
    removed_features.append(max_col)

print(f'\nFeature Set A: {len(cols_A)} features')
print(f'Removed: {removed_features}')

pd.DataFrame({'feature_set_A': cols_A}).to_csv('feature_set_A.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'removed_features': removed_features}).to_csv('removed_features.csv', index=False, encoding='utf-8-sig')

# ==========================================
# 4. TRAIN/TEST SPLIT (by date)
# ==========================================
print("\n" + "="*60)
print("4. TRAIN/TEST SPLIT")
print("="*60)

df_winsorized['Date'] = pd.to_datetime(df_winsorized['Date'])
train_mask = df_winsorized['Date'] <= '2022-03-31'
test_mask = df_winsorized['Date'] >= '2022-04-01'

df_train = df_winsorized[train_mask].copy()
df_test = df_winsorized[test_mask].copy()

print(f'Train: {len(df_train)} ({df_train["Date"].min().date()} ~ {df_train["Date"].max().date()})')
print(f'Test:  {len(df_test)} ({df_test["Date"].min().date()} ~ {df_test["Date"].max().date()})')
print(f'Train Y ratio: {df_train["Y"].mean():.3f}')
print(f'Test Y ratio:  {df_test["Y"].mean():.3f}')

FEATURES_B = list(ALL_FEATURES)
FEATURES_A = cols_A

X_train_B = df_train[FEATURES_B].values
X_test_B = df_test[FEATURES_B].values
X_train_A = df_train[FEATURES_A].values
X_test_A = df_test[FEATURES_A].values
y_train = df_train[TARGET].values
y_test = df_test[TARGET].values
X_train_growth = df_train[GROWTH_FEATURES].values
X_test_growth = df_test[GROWTH_FEATURES].values

# Scale for linear models
scaler_A = StandardScaler()
scaler_B = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)
X_train_growth_scaled = StandardScaler().fit_transform(X_train_growth)
X_test_growth_scaled = StandardScaler().fit_transform(X_test_growth)
print('Scaling complete.')

# ==========================================
# 5. MODEL TRAINING & EVALUATION
# ==========================================
print("\n" + "="*60)
print("5. MODEL TRAINING")
print("="*60)

def evaluate_model(name, model, X_train, X_test, y_train, y_test, feature_set_name=''):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'model': name, 'feature_set': feature_set_name,
        'AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'y_proba': y_proba, 'y_pred': y_pred, 'model_obj': model
    }

results = []
model_objects = {}

# Baseline
baseline = DummyClassifier(strategy='most_frequent')
r = evaluate_model('Baseline (Dummy)', baseline, X_train_A_scaled, X_test_A_scaled, y_train, y_test, 'A')
results.append(r); model_objects['Baseline'] = r
print(f"Baseline — AUC: {r['AUC']:.4f}, Acc: {r['Accuracy']:.4f}")

# LR
lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
r = evaluate_model('Logistic Regression', lr, X_train_A_scaled, X_test_A_scaled, y_train, y_test, 'A')
results.append(r); model_objects['LR'] = r
print(f"LR — AUC: {r['AUC']:.4f}, Acc: {r['Accuracy']:.4f}, F1: {r['F1']:.4f}")

# RF A
rf_a = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
r = evaluate_model('Random Forest', rf_a, X_train_A, X_test_A, y_train, y_test, 'A')
results.append(r); model_objects['RF_A'] = r
print(f"RF (A) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# RF B
rf_b = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
r = evaluate_model('Random Forest', rf_b, X_train_B, X_test_B, y_train, y_test, 'B')
results.append(r); model_objects['RF_B'] = r
print(f"RF (B) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# XGB A
scale = (y_train == False).sum() / (y_train == True).sum()
xgb_a = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=scale, random_state=42, eval_metric='logloss', verbosity=0)
r = evaluate_model('XGBoost', xgb_a, X_train_A, X_test_A, y_train, y_test, 'A')
results.append(r); model_objects['XGB_A'] = r
print(f"XGB (A) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# XGB B
xgb_b = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=scale, random_state=42, eval_metric='logloss', verbosity=0)
r = evaluate_model('XGBoost', xgb_b, X_train_B, X_test_B, y_train, y_test, 'B')
results.append(r); model_objects['XGB_B'] = r
print(f"XGB (B) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# LGBM A
lgbm_a = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              num_leaves=31, random_state=42, verbose=-1)
r = evaluate_model('LightGBM', lgbm_a, X_train_A, X_test_A, y_train, y_test, 'A')
results.append(r); model_objects['LGBM_A'] = r
print(f"LGBM (A) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# LGBM B
lgbm_b = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              num_leaves=31, random_state=42, verbose=-1)
r = evaluate_model('LightGBM', lgbm_b, X_train_B, X_test_B, y_train, y_test, 'B')
results.append(r); model_objects['LGBM_B'] = r
print(f"LGBM (B) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.001,
                     max_iter=500, random_state=42, early_stopping=True)
r = evaluate_model('MLP', mlp, X_train_A_scaled, X_test_A_scaled, y_train, y_test, 'A')
results.append(r); model_objects['MLP'] = r
print(f"MLP (A) — AUC: {r['AUC']:.4f}, F1: {r['F1']:.4f}")

# ==========================================
# 6. MODEL COMPARISON
# ==========================================
print("\n" + "="*60)
print("6. MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_proba', 'y_pred', 'model_obj']} for r in results])
results_df = results_df.sort_values('AUC', ascending=False).reset_index(drop=True)
results_df.to_csv('model_comparison.csv', index=False, encoding='utf-8-sig')
print(results_df.to_string(index=False))

# All results for dashboard
all_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_proba', 'y_pred', 'model_obj']} for r in results])
all_results.to_csv('all_results.csv', index=False, encoding='utf-8-sig')

# ==========================================
# 7. ROC CURVES
# ==========================================
print("\n" + "="*60)
print("7. ROC CURVES")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for i, r in enumerate(results):
    label = f"{r['model']} ({r['feature_set']}) — AUC={r['AUC']:.4f}"
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=label)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: roc_curves.png')

# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================
print("\n" + "="*60)
print("8. FEATURE IMPORTANCE")
print("="*60)

best_xgb = model_objects['XGB_B']['model_obj']
importance_df = pd.DataFrame({
    'feature': FEATURES_B,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(importance_df)), importance_df['importance'].values[::-1])
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'].values[::-1])
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('XGBoost Feature Importance', fontsize=14)
for i, bar in enumerate(bars):
    feat = importance_df['feature'].values[::-1][i]
    color = '#e74c3c' if feat in VALUATION_FEATURES else '#3498db'
    bar.set_color(color)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Valuation'), Patch(facecolor='#3498db', label='Growth')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: feature_importance.png')
print(importance_df.to_string(index=False))

# ==========================================
# 9. SHAP
# ==========================================
print("\n" + "="*60)
print("9. SHAP ANALYSIS")
print("="*60)

explainer = shap.TreeExplainer(best_xgb)
sample_size = min(500, len(X_test_B))
shap_values = explainer.shap_values(X_test_B[:sample_size])

fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_B[:sample_size], feature_names=FEATURES_B, show=False, max_display=17)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_summary.png')

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_B[:sample_size], feature_names=FEATURES_B,
                  plot_type='bar', show=False, max_display=17)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_bar.png')

# ==========================================
# 10. ROBUSTNESS - Growth only
# ==========================================
print("\n" + "="*60)
print("10. ROBUSTNESS - Growth only")
print("="*60)

xgb_growth = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                eval_metric='logloss', verbosity=0)
r = evaluate_model('XGBoost (Growth only)', xgb_growth, X_train_growth, X_test_growth, y_train, y_test, 'Growth')
results.append(r)

lr_growth = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
r2 = evaluate_model('LR (Growth only)', lr_growth, X_train_growth_scaled, X_test_growth_scaled, y_train, y_test, 'Growth')
results.append(r2)

robustness_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['y_proba', 'y_pred', 'model_obj']},
    {k: v for k, v in r2.items() if k not in ['y_proba', 'y_pred', 'model_obj']}
])
print(robustness_df.to_string(index=False))
robustness_df.to_csv('robustness_growth_only.csv', index=False, encoding='utf-8-sig')

# ==========================================
# 11. REGIME ANALYSIS
# ==========================================
print("\n" + "="*60)
print("11. REGIME ANALYSIS")
print("="*60)

best_model = model_objects['XGB_B']['model_obj']
test_proba = best_model.predict_proba(X_test_B)[:, 1]
df_test_eval = df_test.copy()
df_test_eval['proba'] = test_proba

daily_ratio = df_test_eval.groupby(df_test_eval['Date'].dt.date)[TARGET].mean()
median_ratio = daily_ratio.median()
bull_dates = daily_ratio[daily_ratio > median_ratio].index
bear_dates = daily_ratio[daily_ratio <= median_ratio].index
bull_mask = df_test_eval['Date'].dt.date.isin(bull_dates)
bear_mask = df_test_eval['Date'].dt.date.isin(bear_dates)

def calc_regime_metrics(name, mask):
    y_true = df_test_eval.loc[mask, TARGET].values
    y_prob = df_test_eval.loc[mask, 'proba'].values
    y_pred = (y_prob >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return {'Regime': name, 'Samples': len(y_true), 'AUC': np.nan, 'Accuracy': np.nan, 'Y_ratio': round(y_true.mean(), 3)}
    return {
        'Regime': name, 'Samples': len(y_true),
        'AUC': round(roc_auc_score(y_true, y_prob), 4),
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Y_ratio': round(y_true.mean(), 3)
    }

regime_results = pd.DataFrame([
    calc_regime_metrics('Bull (>median)', bull_mask),
    calc_regime_metrics('Bear (<=median)', bear_mask),
    calc_regime_metrics('Overall', slice(None))
])
print(regime_results.to_string(index=False))
regime_results.to_csv('regime_analysis.csv', index=False, encoding='utf-8-sig')

# ==========================================
# 12. LR COEFFICIENTS
# ==========================================
print("\n" + "="*60)
print("12. LR COEFFICIENTS")
print("="*60)

lr_model = model_objects['LR']['model_obj']
lr_coefs = pd.DataFrame({
    'feature': FEATURES_A,
    'coefficient': lr_model.coef_[0],
    'odds_ratio': np.exp(lr_model.coef_[0])
}).sort_values('coefficient', ascending=False)
lr_coefs['interpretation'] = lr_coefs['coefficient'].apply(
    lambda x: 'Positive -> higher prob of rise' if x > 0 else 'Negative -> lower prob of rise'
)
lr_coefs.to_csv('lr_coefficients.csv', index=False, encoding='utf-8-sig')
print(lr_coefs.to_string(index=False))

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

final_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_proba', 'y_pred', 'model_obj']} for r in results])
final_results.to_csv('all_results.csv', index=False, encoding='utf-8-sig')
print(final_results.sort_values('AUC', ascending=False).to_string(index=False))

print('\n=== OUTPUT FILES ===')
for f in sorted(os.listdir('.')):
    if f.endswith(('.csv', '.png')):
        size = os.path.getsize(f) / 1024
        print(f'  {f} ({size:.1f} KB)')

print("\nDONE! All outputs generated.")
