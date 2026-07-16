# -*- coding: utf-8 -*-
"""
Bootstrap 集成 — 20次自助采样 → 弱模型训练 → 概率平均
对比: 单模型 vs Bagging集成 vs Bootstrap集成
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
import sys, io, os, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# ============================================================
# 0. 加载 + 预处理 (复用 P0 脚本逻辑)
# ============================================================
print("=" * 60)
print("Bootstrap 集成方案")
print("=" * 60)

df_raw = pd.read_csv("model_data_stock.csv")
dates = sorted(df_raw['Date'].unique())

# env_market_pe
pe_raw = {d: df_raw[df_raw['Date'] == d]['市盈率PE(TTM,扣除非经常性损益)'].median() for d in dates}
threshold = np.median([pe_raw['2021-06-30'], pe_raw['2021-09-30']])
for d in dates:
    df_raw.loc[df_raw['Date'] == d, 'env_market_pe'] = 1 if pe_raw[d] > threshold else 0

# t→t+1
pairs = []
for i in range(len(dates) - 1):
    dt, dt1 = dates[i], dates[i + 1]
    df_t = df_raw[df_raw['Date'] == dt].copy()
    y_next = df_raw[df_raw['Date'] == dt1][['Code', 'Y']].copy()
    y_next.rename(columns={'Y': 'Y_next'}, inplace=True)
    merged = df_t.merge(y_next, on='Code', how='inner')
    merged['feature_date'] = dt
    pairs.append(merged)
df_model = pd.concat(pairs, ignore_index=True)

# Train (Q2+Q3+Q4) + Val (Q1)  — 用更多 Train 数据做 Bootstrap
df_tr = df_model[df_model['feature_date'].isin(['2021-06-30', '2021-09-30', '2021-12-31'])].copy()
df_va = df_model[df_model['feature_date'] == '2022-03-31'].copy()

regular_feats = [c for c in df_model.columns
                 if c not in ['Date', 'feature_date', 'label_date', 'Code', 'Y', 'Y_next',
                              '市盈率PE(TTM)', 'env_market_pe'] and not c.startswith('Code.')]
env_col = 'env_market_pe'
log_features = ['企业倍数(EV除EBITDA)', '市净率PB(MRQ)', '市销率PS(TTM)', 'MV']
pca_features = ['净利润同比增长率', '利润总额(同比增长率)', '营业利润(同比增长率)', '基本每股收益(同比增长率)']

# ============================================================
# 1. 预处理 (winsorize+log+PCA+interact+scale)
# ============================================================
def prepare_data(df_tr_sub, df_va_sub):
    X_tr = df_tr_sub[regular_feats].copy()
    X_va = df_va_sub[regular_feats].copy()

    # Winsorize
    for col in regular_feats:
        lo, hi = X_tr[col].quantile(0.01), X_tr[col].quantile(0.99)
        X_tr[col], X_va[col] = X_tr[col].clip(lo, hi), X_va[col].clip(lo, hi)

    # Log
    for col in log_features:
        X_tr[col] = np.log1p(X_tr[col].clip(lower=0))
        X_va[col] = np.log1p(X_va[col].clip(lower=0))

    # PCA
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_tr[pca_features].values)
    X_tr['profit_pc1'] = pca.transform(X_tr[pca_features].values)[:, 0]
    X_va['profit_pc1'] = pca.transform(X_va[pca_features].values)[:, 0]
    X_tr.drop(columns=pca_features, inplace=True)
    X_va.drop(columns=pca_features, inplace=True)

    # Interaction features
    env_tr = df_tr_sub[env_col].values.reshape(-1, 1)
    env_va = df_va_sub[env_col].values.reshape(-1, 1)

    X_tr_arr = np.column_stack([X_tr.values, env_tr, X_tr.values * env_tr])
    X_va_arr = np.column_stack([X_va.values, env_va, X_va.values * env_va])

    # StandardScaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_arr)
    X_va_s = scaler.transform(X_va_arr)

    return X_tr_s, X_va_s, df_tr_sub['Y_next'].values, df_va_sub['Y_next'].values

print("\n[1] 预处理...")
X_tr_s, X_va_s, y_tr, y_va = prepare_data(df_tr, df_va)
n_features = X_tr_s.shape[1]
print(f"  Train: {len(X_tr_s):,}  Val: {len(X_va_s):,}  特征: {n_features} (含交互)")
print(f"  Val 正例占比: {y_va.mean()*100:.1f}%")

# ============================================================
# 2. Bootstrap 集成
# ============================================================
N_BOOT = 20
print(f"\n[2] Bootstrap 集成 ({N_BOOT} 次采样)...")

# --- LR Bootstrap ---
print("  LR Bootstrap...")
lr_probs_boot = []
for i in range(N_BOOT):
    idx = np.random.choice(len(X_tr_s), size=len(X_tr_s), replace=True)
    lr = LogisticRegression(C=0.01, penalty='l1', class_weight='balanced',
                             solver='saga', max_iter=2000, random_state=i)
    lr.fit(X_tr_s[idx], y_tr[idx])
    lr_probs_boot.append(lr.predict_proba(X_va_s)[:, 1])
    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{N_BOOT}")

lr_ensemble = np.mean(lr_probs_boot, axis=0)

# --- XGBoost Bootstrap ---
print("  XGBoost Bootstrap...")
xgb_probs_boot = []
for i in range(N_BOOT):
    idx = np.random.choice(len(X_tr_s), size=len(X_tr_s), replace=True)
    xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                               random_state=i, eval_metric='auc', verbosity=0)
    xgb_m.fit(X_tr_s[idx], y_tr[idx])
    xgb_probs_boot.append(xgb_m.predict_proba(X_va_s)[:, 1])
    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{N_BOOT}")

xgb_ensemble = np.mean(xgb_probs_boot, axis=0)

# ============================================================
# 3. 对比基准: 单模型
# ============================================================
print(f"\n[3] 单模型基准...")
lr_single = LogisticRegression(C=0.01, penalty='l1', class_weight='balanced',
                                solver='saga', max_iter=2000, random_state=42)
lr_single.fit(X_tr_s, y_tr)
lr_single_prob = lr_single.predict_proba(X_va_s)[:, 1]

xgb_single = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                 random_state=42, eval_metric='auc', verbosity=0)
xgb_single.fit(X_tr_s, y_tr)
xgb_single_prob = xgb_single.predict_proba(X_va_s)[:, 1]

# ============================================================
# 4. 评估与对比
# ============================================================
print(f"\n[4] 评估对比")

def evaluate(name, prob, y_true):
    auc = roc_auc_score(y_true, prob)
    ic, _ = spearmanr(prob, y_true)
    top100 = y_true[np.argsort(prob)[-100:]].mean()
    top200 = y_true[np.argsort(prob)[-200:]].mean()
    top500 = y_true[np.argsort(prob)[-500:]].mean()
    return {'模型': name, 'AUC': round(auc, 4), 'IC': round(ic, 4),
            'Top100': round(top100, 4), 'Top200': round(top200, 4), 'Top500': round(top500, 4)}

results = [
    evaluate('LR (单模型)',       lr_single_prob, y_va),
    evaluate('LR (Bootstrap)',    lr_ensemble,    y_va),
    evaluate('XGBoost (单模型)',   xgb_single_prob, y_va),
    evaluate('XGBoost (Bootstrap)', xgb_ensemble,   y_va),
]

df_res = pd.DataFrame(results)
print(f"\n  {'模型':<22s} {'AUC':>8s} {'IC':>8s} {'Top100':>8s} {'Top200':>8s} {'Top500':>8s}")
print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for _, r in df_res.iterrows():
    print(f"  {r['模型']:<22s} {r['AUC']:>8.4f} {r['IC']:>+8.4f} {r['Top100']:>8.4f} {r['Top200']:>8.4f} {r['Top500']:>8.4f}")

# 提升幅度
lr_auc_delta = df_res[df_res['模型'] == 'LR (Bootstrap)']['AUC'].values[0] - \
               df_res[df_res['模型'] == 'LR (单模型)']['AUC'].values[0]
xgb_auc_delta = df_res[df_res['模型'] == 'XGBoost (Bootstrap)']['AUC'].values[0] - \
                df_res[df_res['模型'] == 'XGBoost (单模型)']['AUC'].values[0]
print(f"\n  AUC 提升: LR Δ={lr_auc_delta:+.4f}  XGB Δ={xgb_auc_delta:+.4f}")

# ============================================================
# 5. Bootstrap 的多样性分析
# ============================================================
print(f"\n[5] Bootstrap 多样性分析")

# LR 模型间的预测相关性
lr_corr_matrix = np.corrcoef(lr_probs_boot)
lr_avg_corr = (lr_corr_matrix.sum() - N_BOOT) / (N_BOOT * (N_BOOT - 1))
print(f"  LR 模型间平均预测相关系数: {lr_avg_corr:.4f} (越低越多样)")

xgb_corr_matrix = np.corrcoef(xgb_probs_boot)
xgb_avg_corr = (xgb_corr_matrix.sum() - N_BOOT) / (N_BOOT * (N_BOOT - 1))
print(f"  XGB模型间平均预测相关系数: {xgb_avg_corr:.4f} (越低越多样)")

# 个体模型 AUC 分布
lr_aucs = [roc_auc_score(y_va, p) for p in lr_probs_boot]
xgb_aucs = [roc_auc_score(y_va, p) for p in xgb_probs_boot]
print(f"  LR 个体 AUC:  mean={np.mean(lr_aucs):.4f}  std={np.std(lr_aucs):.4f}  min={np.min(lr_aucs):.4f}  max={np.max(lr_aucs):.4f}")
print(f"  XGB个体 AUC:  mean={np.mean(xgb_aucs):.4f}  std={np.std(xgb_aucs):.4f}  min={np.min(xgb_aucs):.4f}  max={np.max(xgb_aucs):.4f}")

# ============================================================
# 6. 最终保存
# ============================================================
df_res.to_csv(os.path.join(OUTPUT_DIR, 'bootstrap_results.csv'), index=False, encoding='utf-8-sig')
print(f"\n  -> bootstrap_results.csv")

print(f"\n{'='*60}")
print(f"Bootstrap 集成完成")
print(f"{'='*60}")
