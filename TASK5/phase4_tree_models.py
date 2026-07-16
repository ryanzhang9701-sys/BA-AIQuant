# -*- coding: utf-8 -*-
"""
Phase 4 — 树模型 (XGBoost + LightGBM + Random Forest)
GridSearchCV 调参 → 特征重要性 → Val评估对比 → 可视化
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTPUT_DIR = "outputs"

# ============================================================
# 0. 加载数据
# ============================================================
print("=" * 60)
print("Phase 4 — 树模型 (XGBoost / LightGBM / Random Forest)")
print("=" * 60)

train = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_processed.csv'))
val   = pd.read_csv(os.path.join(OUTPUT_DIR, 'val_processed.csv'))
test  = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_processed.csv'))

y_col = 'Y_next'
id_cols = ['feature_date', 'label_date', 'Code', y_col]
feat_cols = [c for c in train.columns if c not in id_cols]

X_train, y_train = train[feat_cols].values, train[y_col].values
X_val,   y_val   = val[feat_cols].values,   val[y_col].values
X_test,  y_test  = test[feat_cols].values,  test[y_col].values

print(f"特征: {len(feat_cols)} | Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ============================================================
# 1. XGBoost
# ============================================================
print(f"\n[1] XGBoost GridSearchCV...")

xgb_param = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1.0],
    'reg_lambda': [1.0, 5.0],
}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', random_state=42,
    eval_metric='auc', use_label_encoder=False, verbosity=0
)
xgb_grid = GridSearchCV(xgb_model, xgb_param, cv=3, scoring='roc_auc', verbose=0, n_jobs=-1)
xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_
xgb_val_prob = xgb_best.predict_proba(X_val)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_prob)
xgb_test_prob = xgb_best.predict_proba(X_test)[:, 1]
xgb_test_auc = roc_auc_score(y_test, xgb_test_prob)

# IC
xgb_val_ic, _ = spearmanr(xgb_val_prob, y_val)
xgb_test_ic, _ = spearmanr(xgb_test_prob, y_test)

print(f"  最佳参数: {xgb_grid.best_params_}")
print(f"  Val AUC:  {xgb_val_auc:.4f}  IC: {xgb_val_ic:+.4f}")
print(f"  Test AUC: {xgb_test_auc:.4f}  IC: {xgb_test_ic:+.4f}")

# ============================================================
# 2. LightGBM
# ============================================================
print(f"\n[2] LightGBM GridSearchCV...")

lgb_param = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, -1],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1.0],
    'reg_lambda': [0, 5.0],
}

lgb_model = lgb.LGBMClassifier(
    objective='binary', random_state=42, verbose=-1
)
lgb_grid = GridSearchCV(lgb_model, lgb_param, cv=3, scoring='roc_auc', verbose=0, n_jobs=-1)
lgb_grid.fit(X_train, y_train)

lgb_best = lgb_grid.best_estimator_
lgb_val_prob = lgb_best.predict_proba(X_val)[:, 1]
lgb_val_auc = roc_auc_score(y_val, lgb_val_prob)
lgb_test_prob = lgb_best.predict_proba(X_test)[:, 1]
lgb_test_auc = roc_auc_score(y_test, lgb_test_prob)

lgb_val_ic, _ = spearmanr(lgb_val_prob, y_val)
lgb_test_ic, _ = spearmanr(lgb_test_prob, y_test)

print(f"  最佳参数: {lgb_grid.best_params_}")
print(f"  Val AUC:  {lgb_val_auc:.4f}  IC: {lgb_val_ic:+.4f}")
print(f"  Test AUC: {lgb_test_auc:.4f}  IC: {lgb_test_ic:+.4f}")

# ============================================================
# 3. Random Forest
# ============================================================
print(f"\n[3] Random Forest GridSearchCV...")

rf_param = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf_model, rf_param, cv=3, scoring='roc_auc', verbose=0, n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_val_prob = rf_best.predict_proba(X_val)[:, 1]
rf_val_auc = roc_auc_score(y_val, rf_val_prob)
rf_test_prob = rf_best.predict_proba(X_test)[:, 1]
rf_test_auc = roc_auc_score(y_test, rf_test_prob)

rf_val_ic, _ = spearmanr(rf_val_prob, y_val)
rf_test_ic, _ = spearmanr(rf_test_prob, y_test)

print(f"  最佳参数: {rf_grid.best_params_}")
print(f"  Val AUC:  {rf_val_auc:.4f}  IC: {rf_val_ic:+.4f}")
print(f"  Test AUC: {rf_test_auc:.4f}  IC: {rf_test_ic:+.4f}")

# ============================================================
# 4. 模型对比汇总 + 特征重要性
# ============================================================
print(f"\n[4] 模型对比汇总")

# 追加 LR 基线
try:
    lr_model_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'lr_coefficients.csv'))
    lr_had = True
except:
    lr_had = False

comparison = pd.DataFrame([
    {'模型': 'LogisticRegression', 'Val AUC': 0.4114, 'Val IC': -0.1526, 'Test AUC': 0.5802, 'Test IC': 0.1089},
    {'模型': 'XGBoost',           'Val AUC': xgb_val_auc, 'Val IC': xgb_val_ic, 'Test AUC': xgb_test_auc, 'Test IC': xgb_test_ic},
    {'模型': 'LightGBM',          'Val AUC': lgb_val_auc, 'Val IC': lgb_val_ic, 'Test AUC': lgb_test_auc, 'Test IC': lgb_test_ic},
    {'模型': 'RandomForest',      'Val AUC': rf_val_auc,  'Val IC': rf_val_ic,  'Test AUC': rf_test_auc,  'Test IC': rf_test_ic},
])

comparison.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')

print(f"\n  {'模型':<20s} {'Val AUC':>10s} {'Val IC':>10s} {'Test AUC':>10s} {'Test IC':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for _, r in comparison.iterrows():
    print(f"  {r['模型']:<20s} {r['Val AUC']:>10.4f} {r['Val IC']:>+10.4f} {r['Test AUC']:>10.4f} {r['Test IC']:>+10.4f}")

# 特征重要性 (用 XGBoost)
xgb_imp = pd.DataFrame({
    'feature': feat_cols,
    'importance': xgb_best.feature_importances_
}).sort_values('importance', ascending=True)

print(f"\n  XGBoost Top-10 特征重要性:")
for _, r in xgb_imp.tail(10).iloc[::-1].iterrows():
    bar = '█' * int(r['importance'] * 80)
    print(f"    {r['feature']:<30s} {r['importance']:.4f} {bar}")

# ============================================================
# 5. 特征重要性可视化 (ECharts)
# ============================================================
print(f"\n[5] 特征重要性可视化...")

feat_names = xgb_imp['feature'].tolist()
feat_vals  = xgb_imp['importance'].tolist()

imp_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>特征重要性 — XGBoost</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f6f8;}}
.header{{text-align:center;padding:16px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
.header h1{{font-size:17px;color:#1a1a2e;margin-bottom:4px;}}
.header p{{font-size:11px;color:#888;}}
#main{{width:100%;height:600px;}}
</style>
</head>
<body>
<div class="header"><h1>XGBoost 特征重要性排名</h1>
<p>Gain-based importance | Val AUC={xgb_val_auc:.4f} | Test AUC={xgb_test_auc:.4f}</p></div>
<div id="main"></div>
<script>
var c=echarts.init(document.getElementById('main'));
c.setOption({{
    tooltip: {{trigger:'axis',axisPointer:{{type:'shadow'}}}},
    grid: {{left:'3%',right:'12%',top:'5%',bottom:'5%',containLabel:true}},
    xAxis: {{type:'value',name:'Feature Importance'}},
    yAxis: {{type:'category',data:{json.dumps(feat_names,ensure_ascii=False)},inverse:true,
             axisLabel:{{fontSize:10}}}},
    series: [{{type:'bar',
        data: {json.dumps([round(v,5) for v in feat_vals])},
        itemStyle: {{color: new echarts.graphic.LinearGradient(0,0,1,0,[
            {{offset:0,color:'#3498db'}},{{offset:1,color:'#e74c3c'}}
        ])}},
        label: {{show:true,position:'right',fontSize:9,formatter: '{{c}}'}}
    }}]
}});
window.addEventListener('resize',function(){{c.resize();}});
</script>
</body>
</html>"""

with open(os.path.join(OUTPUT_DIR, 'feature_importance.html'), 'w', encoding='utf-8') as f:
    f.write(imp_html)
print("  -> feature_importance.html")

# ============================================================
# 6. 最佳模型在 Test 上的详细指标
# ============================================================
print(f"\n[6] 最佳模型(XGBoost) Test 详细评估...")

print(f"  Test AUC: {xgb_test_auc:.4f}")

# Precision@TopK
for k in [100, 200, 500]:
    top_idx = np.argsort(xgb_test_prob)[-k:]
    prec = y_test[top_idx].mean()
    print(f"    Top {k:>3d}: Precision={prec:.4f} ({prec*100:.1f}%)")

# IC per date (Test has 1 date)
test_dates = sorted(np.unique(test['feature_date'].values))
for d in test_dates:
    mask = test['feature_date'].values == d
    if mask.sum() > 10:
        ic, _ = spearmanr(xgb_test_prob[mask], y_test[mask])
        print(f"    {d}: IC={ic:.4f}")

# 对比 XGBoost vs LR
print(f"\n  XGBoost vs LR 对比:")
print(f"    Val  AUC: {0.4114:.4f} → {xgb_val_auc:.4f} (Δ={xgb_val_auc-0.4114:+.4f})")
print(f"    Val  IC:  {-0.1526:.4f} → {xgb_val_ic:.4f} (Δ={xgb_val_ic+0.1526:+.4f})")
print(f"    Test AUC: {0.5802:.4f} → {xgb_test_auc:.4f} (Δ={xgb_test_auc-0.5802:+.4f})")

print(f"\n{'='*60}")
print(f"Phase 4 完成")
print(f"{'='*60}")
