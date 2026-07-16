# -*- coding: utf-8 -*-
"""
P0 方案 — 扩展窗口交叉验证 + env×因子交互特征
每窗口独立预处理 (fit on Train) → LR/XGBoost → 汇总评估
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
import sys, io, json, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0. 加载 + 环境特征 + t→t+1 配对
# ============================================================
print("=" * 65)
print("P0 方案 — 扩展窗口 CV + env×因子交互特征")
print("=" * 65)

df_raw = pd.read_csv("model_data_stock.csv")
dates = sorted(df_raw['Date'].unique())

# 环境特征: 二分类
pe_raw = {d: df_raw[df_raw['Date'] == d]['市盈率PE(TTM,扣除非经常性损益)'].median() for d in dates}
threshold = np.median([pe_raw['2021-06-30'], pe_raw['2021-09-30']])
for d in dates:
    df_raw.loc[df_raw['Date'] == d, 'env_market_pe'] = 1 if pe_raw[d] > threshold else 0
print(f"env_market_pe 阈值: {threshold:.2f}, { {d: int(df_raw[df_raw['Date']==d]['env_market_pe'].iloc[0]) for d in dates} }")

# t→t+1 配对
pairs = []
for i in range(len(dates) - 1):
    dt, dt1 = dates[i], dates[i + 1]
    df_t = df_raw[df_raw['Date'] == dt].copy()
    y_next = df_raw[df_raw['Date'] == dt1][['Code', 'Y']].copy()
    y_next.rename(columns={'Y': 'Y_next'}, inplace=True)
    merged = df_t.merge(y_next, on='Code', how='inner')
    merged['feature_date'] = dt
    merged['label_date'] = dt1
    pairs.append(merged)
df_model = pd.concat(pairs, ignore_index=True)

# 常规特征列
drop_base = ['Date', 'feature_date', 'label_date', 'Code', 'Y', 'Y_next', '市盈率PE(TTM)', 'env_market_pe']
regular_feats = [c for c in df_model.columns if c not in drop_base and not c.startswith('Code.')]
env_col = 'env_market_pe'

print(f"常规特征: {len(regular_feats)} | 环境特征: 1 | 配对样本: {len(df_model):,}")

# ============================================================
# 1. 预处理函数 (per-window)
# ============================================================
log_features = ['企业倍数(EV除EBITDA)', '市净率PB(MRQ)', '市销率PS(TTM)', 'MV']
pca_features = ['净利润同比增长率', '利润总额(同比增长率)', '营业利润(同比增长率)', '基本每股收益(同比增长率)']

def preprocess(X_tr, X_va, X_te, feat_list):
    """winsorize+log+PCA, fit on Train, 返回处理后的 numpy + 特征名列表"""
    X_tr_p, X_va_p, X_te_p = X_tr.copy(), X_va.copy(), X_te.copy()

    # Winsorize
    for col in feat_list:
        lo, hi = X_tr_p[col].quantile(0.01), X_tr_p[col].quantile(0.99)
        for X in [X_tr_p, X_va_p, X_te_p]:
            X[col] = X[col].clip(lo, hi)

    # Log
    for col in log_features:
        if col in X_tr_p.columns:
            for X in [X_tr_p, X_va_p, X_te_p]:
                X[col] = np.log1p(X[col].clip(lower=0))

    # PCA
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_tr_p[pca_features].values)
    for X in [X_tr_p, X_va_p, X_te_p]:
        X['profit_pc1'] = pca.transform(X[pca_features].values)[:, 0]
        X.drop(columns=pca_features, inplace=True)

    # 更新特征名
    new_feats = [c for c in X_tr_p.columns]

    # 添加交互项: env × 每个常规特征 (仅当 env 列存在时)
    X_tr_np, X_va_np, X_te_np = X_tr_p.values, X_va_p.values, X_te_p.values

    return X_tr_np, X_va_np, X_te_np, new_feats

# ============================================================
# 2. 定义窗口
# ============================================================
feature_dates_all = ['2021-06-30', '2021-09-30', '2021-12-31', '2022-03-31']
windows = [
    {'name': 'W1', 'train_dates': ['2021-06-30'], 'val_dates': ['2021-09-30']},
    {'name': 'W2', 'train_dates': ['2021-06-30', '2021-09-30'], 'val_dates': ['2021-12-31']},
    {'name': 'W3', 'train_dates': ['2021-06-30', '2021-09-30', '2021-12-31'], 'val_dates': ['2022-03-31']},
]

# ============================================================
# 3. 每个窗口训练 + 评估
# ============================================================
results = []

for w in windows:
    print(f"\n{'─'*65}")
    print(f"[{w['name']}] Train={w['train_dates']}  Val={w['val_dates']}")
    print(f"{'─'*65}")

    # 数据划分
    df_tr = df_model[df_model['feature_date'].isin(w['train_dates'])].copy()
    df_va = df_model[df_model['feature_date'].isin(w['val_dates'])].copy()

    # 预处理
    X_tr_raw = df_tr[regular_feats].copy()
    X_va_raw = df_va[regular_feats].copy()
    X_tr_np, X_va_np, _, proc_feats = preprocess(X_tr_raw, X_va_raw, X_va_raw, regular_feats)

    y_tr = df_tr['Y_next'].values
    y_va = df_va['Y_next'].values
    env_tr = df_tr[env_col].values.reshape(-1, 1)
    env_va = df_va[env_col].values.reshape(-1, 1)

    # 交互特征
    X_tr_inter = np.column_stack([X_tr_np, env_tr, X_tr_np * env_tr])
    X_va_inter = np.column_stack([X_va_np, env_va, X_va_np * env_va])
    inter_names = proc_feats + ['env_market_pe'] + [f'{f}×env' for f in proc_feats]
    n_features = len(inter_names)

    # StandardScaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_inter)
    X_va_s = scaler.transform(X_va_inter)

    # 模型 1: LR (轻量 GridSearch)
    lr = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
    lr_grid = GridSearchCV(lr, {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced']},
                           cv=min(3, max(2, len(df_tr)//500)), scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_tr_s, y_tr)
    lr_prob = lr_grid.best_estimator_.predict_proba(X_va_s)[:, 1]
    lr_auc = roc_auc_score(y_va, lr_prob)
    lr_ic, _ = spearmanr(lr_prob, y_va)

    # 模型 2: XGBoost (轻量)
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                   random_state=42, eval_metric='auc', verbosity=0)
    xgb_model.fit(X_tr_s, y_tr)
    xgb_prob = xgb_model.predict_proba(X_va_s)[:, 1]
    xgb_auc = roc_auc_score(y_va, xgb_prob)
    xgb_ic, _ = spearmanr(xgb_prob, y_va)

    # 特征重要性
    xgb_imp = dict(zip(inter_names, xgb_model.feature_importances_))
    top3 = sorted(xgb_imp.items(), key=lambda x: -x[1])[:3]

    env_val = int(env_va[0, 0]) if len(env_va) > 0 else -1
    results.append({
        'window': w['name'],
        'train_n': len(X_tr_s), 'val_n': len(X_va_s),
        'env_train': f"{env_tr.mean():.2f}", 'env_val': env_val,
        'lr_auc': lr_auc, 'lr_ic': lr_ic,
        'xgb_auc': xgb_auc, 'xgb_ic': xgb_ic,
        'n_features': n_features,
    })

    print(f"  样本: Train={len(X_tr_s):,} Val={len(X_va_s):,} | 特征: {n_features} (含交互)")
    print(f"  env: Train均值={env_tr.mean():.1f} Val={env_val}")
    print(f"  LR:  AUC={lr_auc:.4f}  IC={lr_ic:+.4f}  best={lr_grid.best_params_}")
    print(f"  XGB: AUC={xgb_auc:.4f}  IC={xgb_ic:+.4f}")
    print(f"  Top3重要性: {', '.join(f'{n}({v:.3f})' for n, v in top3)}")

# ============================================================
# 4. 汇总
# ============================================================
print(f"\n{'='*65}")
print(f"汇总: 扩展窗口交叉验证 + 交互特征")
print(f"{'='*65}")

df_res = pd.DataFrame(results)
print(f"\n{'窗口':<8} {'Train':>6} {'Val':>6} {'特征':>5} {'env_T':>6} {'env_V':>5} {'LR_AUC':>8} {'LR_IC':>8} {'XGB_AUC':>8} {'XGB_IC':>8}")
print(f"{'─'*8} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for _, r in df_res.iterrows():
    print(f"{r['window']:<8} {r['train_n']:>6,} {r['val_n']:>6,} {r['n_features']:>5} "
          f"{r['env_train']:>6} {r['env_val']:>5} "
          f"{r['lr_auc']:>8.4f} {r['lr_ic']:>+8.4f} {r['xgb_auc']:>8.4f} {r['xgb_ic']:>+8.4f}")

print(f"\n平均:")
print(f"  LR:  AUC={df_res['lr_auc'].mean():.4f}  IC={df_res['lr_ic'].mean():.4f}  波动(std)={df_res['lr_auc'].std():.4f}")
print(f"  XGB: AUC={df_res['xgb_auc'].mean():.4f}  IC={df_res['xgb_ic'].mean():.4f}  波动(std)={df_res['xgb_auc'].std():.4f}")

# 保存
df_res.to_csv(os.path.join(OUTPUT_DIR, 'expanding_cv_results.csv'), index=False, encoding='utf-8-sig')
print(f"\n  -> expanding_cv_results.csv")

# ============================================================
# 5. 最终模型: 用全部 3 窗 Train 数据训练 → Test 评估
# ============================================================
print(f"\n{'='*65}")
print(f"最终模型: Train=W1+W2+W3 全部数据 → Test(2022Q1)")

df_tr_final = df_model[df_model['feature_date'].isin(['2021-06-30', '2021-09-30', '2021-12-31'])].copy()
df_te_final = df_model[df_model['feature_date'] == '2022-03-31'].copy()

X_tr_raw_f = df_tr_final[regular_feats].copy()
X_te_raw_f = df_te_final[regular_feats].copy()
X_tr_np_f, _, X_te_np_f, proc_feats_f = preprocess(X_tr_raw_f, X_te_raw_f, X_te_raw_f, regular_feats)

y_tr_f = df_tr_final['Y_next'].values
y_te_f = df_te_final['Y_next'].values
env_tr_f = df_tr_final[env_col].values.reshape(-1, 1)
env_te_f = df_te_final[env_col].values.reshape(-1, 1)

X_tr_inter_f = np.column_stack([X_tr_np_f, env_tr_f, X_tr_np_f * env_tr_f])
X_te_inter_f = np.column_stack([X_te_np_f, env_te_f, X_te_np_f * env_te_f])

scaler_f = StandardScaler()
X_tr_f = scaler_f.fit_transform(X_tr_inter_f)
X_te_f = scaler_f.transform(X_te_inter_f)

# LR
lr_f = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
lr_grid_f = GridSearchCV(lr_f, {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced']},
                          cv=3, scoring='roc_auc', n_jobs=-1)
lr_grid_f.fit(X_tr_f, y_tr_f)
lr_prob_f = lr_grid_f.best_estimator_.predict_proba(X_te_f)[:, 1]
lr_auc_f = roc_auc_score(y_te_f, lr_prob_f)
lr_ic_f, _ = spearmanr(lr_prob_f, y_te_f)

# XGB
xgb_f = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                            random_state=42, eval_metric='auc', verbosity=0)
xgb_f.fit(X_tr_f, y_tr_f)
xgb_prob_f = xgb_f.predict_proba(X_te_f)[:, 1]
xgb_auc_f = roc_auc_score(y_te_f, xgb_prob_f)
xgb_ic_f, _ = spearmanr(xgb_prob_f, y_te_f)

print(f"\n  Test 评估 (feature_date=2022Q1, label=2022Q2):")
print(f"    样本: Train={len(X_tr_f):,} Test={len(X_te_f):,}")
print(f"    LR:  AUC={lr_auc_f:.4f}  IC={lr_ic_f:+.4f}")
print(f"    XGB: AUC={xgb_auc_f:.4f}  IC={xgb_ic_f:+.4f}")

# Precision@TopK
for k in [100, 200, 500]:
    top_k = np.argsort(xgb_prob_f)[-k:]
    print(f"    XGB Top{k}: Precision={y_te_f[top_k].mean():.4f}")

print(f"\n{'='*65}")
print(f"P0 方案完成")
print(f"{'='*65}")
