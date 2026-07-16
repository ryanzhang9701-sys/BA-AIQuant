# -*- coding: utf-8 -*-
"""
Phase 3 — 逻辑回归 Baseline
GridSearchCV(L1/L2, C, class_weight) → 系数 → Val评估 → ROC曲线
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_score
from scipy.stats import spearmanr
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0. 加载 Phase 2 处理后的数据
# ============================================================
print("=" * 60)
print("Phase 3 — 逻辑回归 Baseline")
print("=" * 60)

train = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_processed.csv'))
val   = pd.read_csv(os.path.join(OUTPUT_DIR, 'val_processed.csv'))
test  = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_processed.csv'))

y_col = 'Y_next'
id_cols = ['feature_date', 'label_date', 'Code', y_col]
feat_cols = [c for c in train.columns if c not in id_cols]

X_train = train[feat_cols].values
y_train = train[y_col].values
X_val   = val[feat_cols].values
y_val   = val[y_col].values
X_test  = test[feat_cols].values
y_test  = test[y_col].values

val_dates = val['feature_date'].values

print(f"特征数: {len(feat_cols)}")
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Val 正例占比: {y_val.mean()*100:.1f}%")

# ============================================================
# 1. GridSearchCV
# ============================================================
print(f"\n[1] GridSearchCV 超参数搜索...")

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None]
}

lr = LogisticRegression(
    solver='saga',       # saga supports both L1 and L2
    max_iter=2000,
    random_state=42,
    n_jobs=1
)

grid = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', verbose=0, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳 CV AUC: {grid.best_score_:.4f}")

# ============================================================
# 2. 系数表
# ============================================================
print(f"\n[2] 逻辑回归系数...")

best_lr = grid.best_estimator_
coef = best_lr.coef_[0]

# 系数排名
coef_df = pd.DataFrame({
    'feature': feat_cols,
    'coefficient': coef,
    'abs_coef': np.abs(coef)
}).sort_values('abs_coef', ascending=False)

# 方向检验
print(f"\n  {'特征':<30s} {'系数':>12s} {'方向':>6s}")
print(f"  {'-'*30} {'-'*12} {'-'*6}")
for _, row in coef_df.iterrows():
    direction = '涨正' if row['coefficient'] > 0 else '跌负'
    print(f"  {row['feature']:<30s} {row['coefficient']:>+12.6f} {direction:>6s}")

coef_df.to_csv(os.path.join(OUTPUT_DIR, 'lr_coefficients.csv'), index=False, encoding='utf-8-sig')
print(f"  -> lr_coefficients.csv")

# ============================================================
# 3. Val 集评估
# ============================================================
print(f"\n[3] Val 集评估...")

y_pred_proba = best_lr.predict_proba(X_val)[:, 1]
y_pred = best_lr.predict(X_val)

# AUC
auc_val = roc_auc_score(y_val, y_pred_proba)
print(f"  AUC: {auc_val:.4f}")

# IC (Rank IC per date)
print(f"\n  IC (Rank IC) 按日期:")
ic_values = []
for d in sorted(np.unique(val_dates)):
    mask = val_dates == d
    if mask.sum() > 10:
        ic, _ = spearmanr(y_pred_proba[mask], y_val[mask])
        ic_values.append(ic)
        print(f"    {d}: IC={ic:.4f}  (n={mask.sum()})")

mean_ic = np.mean(ic_values) if ic_values else 0
std_ic = np.std(ic_values) if ic_values else 0
icir = mean_ic / std_ic if std_ic > 0 else 0
print(f"  Mean IC: {mean_ic:.4f}")
print(f"  ICIR:    {icir:.4f}")

# Precision@TopK
print(f"\n  Precision@TopK:")
for k in [100, 200, 500]:
    top_idx = np.argsort(y_pred_proba)[-k:]
    prec = y_val[top_idx].mean()
    print(f"    Top {k:>3d}: {prec:.4f} ({prec*100:.1f}%)")

# ============================================================
# 4. ROC 曲线 (ECharts HTML)
# ============================================================
print(f"\n[4] 生成 ROC 曲线...")

fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_data = [[round(float(f), 5), round(float(t), 5)] for f, t in zip(fpr, tpr)]

roc_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ROC 曲线 — 逻辑回归</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f6f8;}}
.header{{text-align:center;padding:16px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
.header h1{{font-size:17px;color:#1a1a2e;margin-bottom:4px;}}
.header p{{font-size:11px;color:#888;}}
#main{{width:100%;height:550px;}}
</style>
</head>
<body>
<div class="header"><h1>ROC 曲线 — 逻辑回归 Baseline</h1>
<p>Val 集 AUC = {auc_val:.4f} | Mean IC = {mean_ic:.4f} | ICIR = {icir:.4f}</p></div>
<div id="main"></div>
<script>
var c=echarts.init(document.getElementById('main'));
c.setOption({{
    tooltip: {{trigger: 'axis'}},
    grid: {{left: '10%', right: '10%', top: '8%', bottom: '10%'}},
    xAxis: {{type: 'value', name: 'False Positive Rate', min: 0, max: 1}},
    yAxis: {{type: 'value', name: 'True Positive Rate', min: 0, max: 1}},
    series: [
        {{type: 'line', data: {json.dumps(roc_data, ensure_ascii=False)},
          lineStyle: {{color: '#e74c3c', width: 3}}, symbol: 'none',
          areaStyle: {{color: 'rgba(231,76,60,0.1)'}},
          name: 'LR (AUC={auc_val:.4f})'}},
        {{type: 'line', data: [[0,0],[1,1]],
          lineStyle: {{color: '#999', type: 'dashed', width: 2}},
          symbol: 'none', name: 'Random'}}
    ]
}});
window.addEventListener('resize',function(){{c.resize();}});
</script>
</body>
</html>"""

roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.html')
with open(roc_path, 'w', encoding='utf-8') as f:
    f.write(roc_html)
print(f"  -> roc_curve.html")

# ============================================================
# 5. Test 集快速预览
# ============================================================
print(f"\n[5] Test 集快速预览 (正式评估在 Phase 5)...")

y_test_proba = best_lr.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_test_proba)

ic_test_vals = []
for d in sorted(np.unique(test['feature_date'].values)):
    mask = test['feature_date'].values == d
    if mask.sum() > 10:
        ic, _ = spearmanr(y_test_proba[mask], y_test[mask])
        ic_test_vals.append(ic)

print(f"  Test AUC:   {auc_test:.4f}")
print(f"  Test IC:    {np.mean(ic_test_vals):.4f}" if ic_test_vals else "  N/A")

# ============================================================
# 6. 汇总
# ============================================================
print(f"\n{'='*60}")
print(f"Phase 3 完成")
print(f"{'='*60}")
print(f"\n  模型:     LogisticRegression(C={best_lr.C}, penalty={best_lr.penalty}, class_weight={best_lr.class_weight})")
print(f"  Val AUC:  {auc_val:.4f}")
print(f"  Val IC:   {mean_ic:.4f}")
print(f"  Val ICIR: {icir:.4f}")
print(f"  Test AUC: {auc_test:.4f}")
print(f"\n  Top-5 特征:")
for _, row in coef_df.head(5).iterrows():
    print(f"    {row['feature']:<30s} {row['coefficient']:>+12.6f}")
