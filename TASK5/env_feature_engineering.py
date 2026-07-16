# -*- coding: utf-8 -*-
"""
环境特征工程 v2 — 精简版: 仅一个市场状态指标
从原始数据中计算每期全市场PE中位数作为环境特征
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_FILE = "model_data_stock.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("环境特征工程 v2 — 单一市场状态指标")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
dates = sorted(df['Date'].unique())

# ============================================================
# 构建市场状态: env_market_pe = 二分类 (基于 Train 中位数阈值)
# ============================================================
print(f"\n市场状态指标: env_market_pe (二分类)")

# Step 1: 计算每期 PE 中位数
env_pe_raw = {}
for d in dates:
    sub = df[df['Date'] == d]
    pe_med = sub['市盈率PE(TTM,扣除非经常性损益)'].median()
    env_pe_raw[d] = round(pe_med, 2)

# Step 2: 阈值 = Train 日期 (2021-06-30, 2021-09-30) 的 PE 中位数
train_dates_th = ['2021-06-30', '2021-09-30']
threshold = np.median([env_pe_raw[d] for d in train_dates_th])

print(f"  定义: 全市场PE(TTM,扣非)中位数 > {threshold:.2f} → 1 (高估值), ≤ {threshold:.2f} → 0 (低估值)")
print(f"  阈值: {threshold:.2f} (Train 内部 PE 中位数)")
print(f"  来源: 从原始数据直接计算, 在t时刻可观测, 无前视偏差")
print()

env_map = {}
for d in dates:
    env_map[d] = 1 if env_pe_raw[d] > threshold else 0
    label = "高估值" if env_map[d] == 1 else "低估值"
    print(f"  {d} (n={len(df[df['Date']==d]):,}): PE={env_pe_raw[d]:.2f} → env_market_pe = {env_map[d]} ({label})")

# ============================================================
# 合并到原始数据
# ============================================================
env_df = pd.DataFrame([{'Date': d, 'env_market_pe': v} for d, v in env_map.items()])
df_env = df.merge(env_df, on='Date', how='left')

# ============================================================
# t → t+1 配对 + 划分 (复用 Phase 1 逻辑)
# ============================================================
print(f"\n[t→t+1 配对 + 划分]")

pairs = []
for i in range(len(dates) - 1):
    dt, dt1 = dates[i], dates[i + 1]
    df_t = df_env[df_env['Date'] == dt].copy()
    y_next = df_env[df_env['Date'] == dt1][['Code', 'Y']].copy()
    y_next.rename(columns={'Y': 'Y_next'}, inplace=True)
    merged = df_t.merge(y_next, on='Code', how='inner')
    merged['feature_date'] = dt
    merged['label_date'] = dt1
    pairs.append(merged)

df_model = pd.concat(pairs, ignore_index=True)

df_train = df_model[df_model['feature_date'].isin(['2021-06-30', '2021-09-30'])].copy()
df_val   = df_model[df_model['feature_date'].isin(['2021-12-31'])].copy()
df_test  = df_model[df_model['feature_date'].isin(['2022-03-31'])].copy()

print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# ============================================================
# 预处理 (环境特征仅过 StandardScaler, 不缩尾/不log/不PCA)
# ============================================================
print(f"\n[预处理]")

# 常规特征
drop_base = ['Date', 'feature_date', 'label_date', 'Code', 'Y', 'Y_next', '市盈率PE(TTM)', 'env_market_pe']
regular_feats = [c for c in df_train.columns if c not in drop_base and not c.startswith('Code.')]
env_feats = ['env_market_pe']

print(f"  常规特征: {len(regular_feats)} | 环境特征: {len(env_feats)} | 总计: {len(regular_feats)+len(env_feats)}")

# Winsorize (仅常规)
X_tr_r, X_v_r, X_te_r = df_train[regular_feats].copy(), df_val[regular_feats].copy(), df_test[regular_feats].copy()
for col in regular_feats:
    lo, hi = X_tr_r[col].quantile(0.01), X_tr_r[col].quantile(0.99)
    X_tr_r[col], X_v_r[col], X_te_r[col] = X_tr_r[col].clip(lo, hi), X_v_r[col].clip(lo, hi), X_te_r[col].clip(lo, hi)

# Log1p (仅指定)
for col in ['企业倍数(EV除EBITDA)', '市净率PB(MRQ)', '市销率PS(TTM)', 'MV']:
    if col in X_tr_r.columns:
        X_tr_r[col] = np.log1p(X_tr_r[col].clip(lower=0))
        X_v_r[col]   = np.log1p(X_v_r[col].clip(lower=0))
        X_te_r[col]  = np.log1p(X_te_r[col].clip(lower=0))

# PCA (profit_cluster)
pca_feats = ['净利润同比增长率', '利润总额(同比增长率)', '营业利润(同比增长率)', '基本每股收益(同比增长率)']
pca = PCA(n_components=1, random_state=42)
pca.fit(X_tr_r[pca_feats].values)
X_tr_r['profit_pc1'] = pca.transform(X_tr_r[pca_feats].values)[:, 0]
X_v_r['profit_pc1']   = pca.transform(X_v_r[pca_feats].values)[:, 0]
X_te_r['profit_pc1']  = pca.transform(X_te_r[pca_feats].values)[:, 0]
X_tr_r.drop(columns=pca_feats, inplace=True)
X_v_r.drop(columns=pca_feats, inplace=True)
X_te_r.drop(columns=pca_feats, inplace=True)
print(f"  PCA: profit_cluster ×4 → profit_pc1 (解释方差 {pca.explained_variance_ratio_[0]*100:.1f}%)")

# 合并环境特征
X_tr_all = np.column_stack([X_tr_r.values, df_train[env_feats].values])
X_v_all   = np.column_stack([X_v_r.values,   df_val[env_feats].values])
X_te_all  = np.column_stack([X_te_r.values,  df_test[env_feats].values])
all_feats = list(X_tr_r.columns) + env_feats

# StandardScaler
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr_all)
X_v  = scaler.transform(X_v_all)
X_te = scaler.transform(X_te_all)

y_tr = df_train['Y_next'].values
y_v  = df_val['Y_next'].values
y_te = df_test['Y_next'].values

print(f"  StandardScaler: ✓ ({len(all_feats)} 特征)")

# ============================================================
# 导出
# ============================================================
for name, X, y, dfo in [('train', X_tr, y_tr, df_train),
                          ('val',   X_v,  y_v,  df_val),
                          ('test',  X_te, y_te, df_test)]:
    df_out = pd.DataFrame(X, columns=all_feats)
    df_out['Y_next'] = y
    df_out['Code'] = dfo['Code'].values
    df_out['feature_date'] = dfo['feature_date'].values
    df_out.to_csv(os.path.join(OUTPUT_DIR, f'{name}_processed.csv'), index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"环境特征工程 v2 完成")
print(f"新增特征: env_market_pe (1个)")
print(f"最终特征: {len(all_feats)} ({len(X_tr_r.columns)} 常规 + 1 环境)")
print(f"{'='*60}")
