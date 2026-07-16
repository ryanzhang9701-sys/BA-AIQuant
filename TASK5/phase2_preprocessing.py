# -*- coding: utf-8 -*-
"""
Phase 2 — 预处理流水线
Winsorize(P1/P99) → Log1p → PCA(profit_cluster) → 相关性/VIF重算 → StandardScaler
全部统计量仅在 Train 上拟合
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0. 加载 Phase 1 输出
# ============================================================
print("=" * 60)
print("Phase 2 — 预处理流水线")
print("=" * 60)

df_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_raw.csv'))
df_val   = pd.read_csv(os.path.join(OUTPUT_DIR, 'val_raw.csv'))
df_test  = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_raw.csv'))

print(f"\n加载: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")

# 确定特征列
id_cols = ['feature_date', 'label_date', 'Code']
y_col = 'Y_next'
# drop: ID列 + 标签 + PE(TTM) + 原始Y + merge产生的重复Code列
drop_cols_base = id_cols + [y_col, '市盈率PE(TTM)', 'Y']
# 处理 Code.1 等可能的重复列
drop_cols = [c for c in drop_cols_base if c in df_train.columns]\
          + [c for c in df_train.columns if c.startswith('Code.')]
feat_cols = [c for c in df_train.columns if c not in drop_cols]

print(f"建模特征数: {len(feat_cols)} (已剔除 PE(TTM))")

# 分类特征
log_features = ['企业倍数(EV除EBITDA)', '市净率PB(MRQ)', '市销率PS(TTM)', 'MV']
pca_features = ['净利润同比增长率', '利润总额(同比增长率)', '营业利润(同比增长率)', '基本每股收益(同比增长率)']

X_train_raw = df_train[feat_cols].copy()
X_val_raw   = df_val[feat_cols].copy()
X_test_raw  = df_test[feat_cols].copy()
y_train = df_train[y_col].values
y_val   = df_val[y_col].values
y_test  = df_test[y_col].values

# ============================================================
# Step 1a — Winsorize (fit on Train)
# ============================================================
print(f"\n[Step 1a] Winsorize (P1/P99, fit on Train)...")

X_train_win = X_train_raw.copy()
X_val_win   = X_val_raw.copy()
X_test_win  = X_test_raw.copy()

winsorize_bounds = {}
for col in feat_cols:
    lo = X_train_raw[col].quantile(0.01)
    hi = X_train_raw[col].quantile(0.99)
    winsorize_bounds[col] = {'P1': lo, 'P99': hi}
    X_train_win[col] = X_train_raw[col].clip(lo, hi)
    X_val_win[col]   = X_val_raw[col].clip(lo, hi)
    X_test_win[col]  = X_test_raw[col].clip(lo, hi)

# 报告: 缩尾前后对比
print("\n  缩尾前后对比 (Train 集):")
print(f"  {'特征':<22s} {'原始Min':>12s} {'原始Max':>12s} {'缩尾Min':>12s} {'缩尾Max':>12s} {'偏度变化':>10s}")
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
comp_rows = []
for col in feat_cols:
    orig_min = X_train_raw[col].min()
    orig_max = X_train_raw[col].max()
    win_min  = X_train_win[col].min()
    win_max  = X_train_win[col].max()
    skew_before = round(X_train_raw[col].skew(), 2)
    skew_after  = round(X_train_win[col].skew(), 2)
    comp_rows.append({
        'feature': col[:22], 'orig_min': orig_min, 'orig_max': orig_max,
        'win_min': win_min, 'win_max': win_max,
        'skew_before': skew_before, 'skew_after': skew_after,
        'skew_change': round(skew_after - skew_before, 2)
    })
    print(f"  {col[:22]:<22s} {orig_min:>12.2f} {orig_max:>12.2f} {win_min:>12.4f} {win_max:>12.4f} {comp_rows[-1]['skew_change']:>+10.2f}")

pd.DataFrame(comp_rows).to_csv(os.path.join(OUTPUT_DIR, 'winsorize_comparison.csv'), index=False, encoding='utf-8-sig')
print(f"  -> winsorize_comparison.csv")

# ============================================================
# Step 1b — Log 变换
# ============================================================
print(f"\n[Step 1b] Log1p 变换...")

skew_log = []
for col in feat_cols:
    s_before = round(X_train_win[col].skew(), 2)
    if col in log_features:
        X_train_win[col] = np.log1p(X_train_win[col].clip(lower=0))
        X_val_win[col]   = np.log1p(X_val_win[col].clip(lower=0))
        X_test_win[col]  = np.log1p(X_test_win[col].clip(lower=0))
    s_after = round(X_train_win[col].skew(), 2)
    skew_log.append({'feature': col[:25], 'skew_after_winsorize': s_before,
                     'skew_after_log': s_after, 'log_applied': col in log_features})

for r in skew_log:
    if r['log_applied']:
        print(f"  {r['feature']:<25s} skew: {r['skew_after_winsorize']:>+7.2f} → {r['skew_after_log']:>+7.2f}")

pd.DataFrame(skew_log).to_csv(os.path.join(OUTPUT_DIR, 'skew_comparison.csv'), index=False, encoding='utf-8-sig')
print(f"  -> skew_comparison.csv")

# ============================================================
# Step 1c — PCA (profit_cluster, fit on Train)
# ============================================================
print(f"\n[Step 1c] PCA — profit_cluster ×4 → profit_pc1...")

pca = PCA(n_components=1, random_state=42)
pca_data_train = X_train_win[pca_features].values
pca_data_val   = X_val_win[pca_features].values
pca_data_test  = X_test_win[pca_features].values

pca.fit(pca_data_train)
pc1_train = pca.transform(pca_data_train)[:, 0]
pc1_val   = pca.transform(pca_data_val)[:, 0]
pc1_test  = pca.transform(pca_data_test)[:, 0]

# PCA 报告
pca_loadings = pca.components_[0]
pca_report = {
    'explained_variance_ratio': round(pca.explained_variance_ratio_[0], 4),
    'loadings': {pca_features[i]: round(float(pca_loadings[i]), 4) for i in range(len(pca_features))}
}
print(f"  解释方差比: {pca_report['explained_variance_ratio']:.4f} ({pca_report['explained_variance_ratio']*100:.1f}%)")
print(f"  载荷:")
for feat, load in pca_report['loadings'].items():
    print(f"    {feat[:20]:<20s} {load:>+.4f}")

with open(os.path.join(OUTPUT_DIR, 'pca_report.json'), 'w', encoding='utf-8') as f:
    json.dump(pca_report, f, ensure_ascii=False, indent=2)

# 替换: 移除 4 个原始列，加入 profit_pc1
X_train_proc = X_train_win.drop(columns=pca_features).copy()
X_val_proc   = X_val_win.drop(columns=pca_features).copy()
X_test_proc  = X_test_win.drop(columns=pca_features).copy()

X_train_proc['profit_pc1'] = pc1_train
X_val_proc['profit_pc1']   = pc1_val
X_test_proc['profit_pc1']  = pc1_test

proc_cols = list(X_train_proc.columns)
print(f"  特征数: {len(feat_cols)} → {len(proc_cols)} (剔除4个profit + 新增1个profit_pc1)")
print(f"  -> pca_report.json")

# ============================================================
# Step 1d — 缩尾后重算 相关性 / VIF
# ============================================================
print(f"\n[Step 1d] 缩尾后重算 Spearman/Pearson/VIF...")

pearson_after = X_train_proc[proc_cols].corr('pearson')
spearman_after = X_train_proc[proc_cols].corr('spearman')

# 找高相关对
high_pairs_after = []
for i in range(len(proc_cols)):
    for j in range(i + 1, len(proc_cols)):
        sp = spearman_after.iloc[i, j]
        if abs(sp) > 0.7:
            high_pairs_after.append((proc_cols[i], proc_cols[j], round(sp, 4), round(pearson_after.iloc[i, j], 4)))

print(f"  高相关对 (|Spearman|>0.7): {len(high_pairs_after)} (缩尾前: 7)")
for c1, c2, sp, ps in high_pairs_after:
    print(f"    {c1[:15]} ↔ {c2[:15]}  S={sp:+.4f} P={ps:+.4f}")

# VIF
def calc_vif(X_arr):
    n, k = X_arr.shape
    v = {}
    for j in range(k):
        y = X_arr[:, j]
        Xo = np.delete(X_arr, j, axis=1)
        XtX = Xo.T @ Xo + np.eye(Xo.shape[1]) * 1e-8
        beta = np.linalg.solve(XtX, Xo.T @ y)
        yp = Xo @ beta
        r2 = 1 - np.sum((y - yp)**2) / max(np.sum((y - np.mean(y))**2), 1e-15)
        v[proc_cols[j]] = round(1/(1-r2) if r2 < 0.9999 else 100, 2)
    return v

vif_vals = calc_vif(X_train_proc.values.astype(np.float64))
vif_sorted = sorted(vif_vals.items(), key=lambda x: -x[1])

print(f"\n  VIF (缩尾+log+PCA后):")
print(f"  {'特征':<30s} {'VIF':>8s}")
print(f"  {'-'*30} {'-'*8}")
vif_warns = []
for col, vif in vif_sorted:
    warn = ' ⚠️ >10' if vif > 10 else (' ⚡ 5-10' if vif > 5 else '')
    if warn:
        vif_warns.append((col, vif))
    print(f"  {col:<30s} {vif:>8.2f}{warn}")

# 保存
pd.DataFrame({'feature': list(vif_vals.keys()), 'VIF': list(vif_vals.values())}
            ).sort_values('VIF', ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, 'vif_after_winsorize.csv'), index=False, encoding='utf-8-sig')
pearson_after.to_csv(os.path.join(OUTPUT_DIR, 'corr_after_winsorize.csv'), encoding='utf-8-sig')

print(f"  -> vif_after_winsorize.csv, corr_after_winsorize.csv")

# ============================================================
# Step 1e — StandardScaler (fit on Train)
# ============================================================
print(f"\n[Step 1e] StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_proc)
X_val_scaled   = scaler.transform(X_val_proc)
X_test_scaled  = scaler.transform(X_test_proc)

# 验证
train_means = X_train_scaled.mean(axis=0)
train_stds  = X_train_scaled.std(axis=0)
print(f"  Train 均值范围: [{train_means.min():.6f}, {train_means.max():.6f}] (期望 ≈0)")
print(f"  Train 标准差范围: [{train_stds.min():.6f}, {train_stds.max():.6f}] (期望 ≈1)")

# ============================================================
# 6. 导出处理后的数据
# ============================================================
print(f"\n[导出] 处理后的数据...")

def save_processed(name, X_scaled, y, df_orig, scaler_fitted):
    df_out = pd.DataFrame(X_scaled, columns=proc_cols)
    df_out[y_col] = y
    df_out['Code'] = df_orig['Code'].values
    df_out['feature_date'] = df_orig['feature_date'].values
    path = os.path.join(OUTPUT_DIR, f'{name}_processed.csv')
    df_out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  {path} ({len(df_out):,} 行 × {len(df_out.columns)} 列)")
    return path

save_processed('train', X_train_scaled, y_train, df_train, scaler)
save_processed('val',   X_val_scaled,   y_val,   df_val,   scaler)
save_processed('test',  X_test_scaled,  y_test,  df_test,  scaler)

# ============================================================
# 7. 最终摘要
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 2 完成")
print(f"{'='*60}")
print(f"特征数: {len(feat_cols)} → {len(proc_cols)}")
print(f"  - 剔除 PE(TTM) (1)")
print(f"  - profit_cluster ×4 → profit_pc1 (1)")
print(f"  - 保留: {len(proc_cols) - 1} 独立特征 + profit_pc1")

print(f"\n处理摘要:")
print(f"  Winsorize (P1/P99): {len(feat_cols)} 特征")
print(f"  Log1p: {len(log_features)} 特征 ({log_features})")
print(f"  PCA: {len(pca_features)} → 1 (解释方差 {pca_report['explained_variance_ratio']*100:.1f}%)")
print(f"  VIF 最大值: {max(v[1] for v in vif_vals.values()):.2f} (缩尾前: ~1)")
print(f"  StandardScaler: ✓")

if vif_warns:
    print(f"\n⚠️ VIF 警告 ({len(vif_warns)} 个特征 VIF>5):")
    for col, vif in vif_warns:
        print(f"  {col}: {vif}")
else:
    print(f"\n✓ 无 VIF 问题 (全部 < 5)")
