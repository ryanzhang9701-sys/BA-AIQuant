#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive pipeline diagnostic - checks every step before training."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PIPELINE DIAGNOSTIC REPORT")
print("=" * 60)

# ==== 1. Load ====
print("\n[1] Data Loading")
df = pd.read_csv('model_data.csv', encoding='utf-8')
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y/%m/%d')
dates = sorted(df['Date'].unique())
print(f"  Rows: {len(df)}, Stocks: {df['Code'].nunique()}, Date range: {dates[0]} ~ {dates[-1]} ({len(dates)} periods)")

# ==== 2. Target ====
print("\n[2] Target (Next_Ret)")
nr = df['Next_Ret'].dropna()
print(f"  Valid: {len(nr)}/{len(df)}")
print(f"  Stats: mean={nr.mean():.4f}, std={nr.std():.4f}, min={nr.min():.4f}, max={nr.max():.4f}")
print(f"  Y=1 rate (Next_Ret>0): {(nr>0).mean():.2%}")

# ==== 3. Column mapping ====
print("\n[3] Column Renaming")
rename_map = {}
for c in df.columns:
    new = c.strip()
    if '(' in new: new = new.split('(')[0].strip()
    new = new.replace(' ', '_').replace('/', '_').replace('（', '').replace('）', '')
    rename_map[c] = new
df.rename(columns=rename_map, inplace=True)

RAW_IDX = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
NAMES = ['EV_EBITDA','PB','PCF_NetCash','PCF_Operating','PE_TTM','PE_TTM_Deducted',
         'PS_TTM','Dividend_Yield','MV','Profit_Growth','NetAsset_Growth',
         'TotalProfit_Growth','EPS_Growth','TotalAsset_Growth','NetCash_Growth',
         'OperatingCF_Growth','OperatingProfit_Growth','Revenue1_Growth','Revenue2_Growth']

# ==== 4. Factor value check ====
print("\n[4] Raw Factor Quality")
issues = []
for idx, name in zip(RAW_IDX, NAMES):
    vals = pd.to_numeric(df.iloc[:, idx], errors='coerce')
    valid = vals.notna().sum()
    pct = valid / len(df) * 100
    status = "OK" if pct > 50 else ("WARN" if pct > 10 else "BAD")
    if status != "OK": issues.append(f"  {name}: valid={valid} ({pct:.1f}%) [{status}]")
    print(f"  {name:25s}: {status:4s}  valid={valid:5d}/{len(df)} ({pct:5.1f}%)  range=[{vals.min():.1f}, {vals.max():.1f}]")

# ==== 5. Date splits ====
print("\n[5] Walk-Forward Date Splits")
WINDOWS = [
    ('W1', '2020/03/31', '2020/12/31', '2021/03/31', '2021/06/30'),
    ('W2', '2020/03/31', '2021/03/31', '2021/06/30', '2021/09/30'),
    ('W3', '2020/03/31', '2021/06/30', '2021/09/30', '2021/12/31'),
    ('W4', '2020/03/31', '2021/09/30', '2021/12/31', '2022/03/31'),
    ('W5', '2020/03/31', '2021/12/31', '2022/03/31', '2022/06/30'),
]
for wname, ts, te, ve, tte in WINDOWS:
    train_n = ((df['Date']>=ts)&(df['Date']<=te)).sum()
    val_n = ((df['Date']>te)&(df['Date']<=ve)).sum()
    test_n = ((df['Date']>ve)&(df['Date']<=tte)).sum()
    ok = train_n > 1000 and val_n > 500 and test_n > 500
    status = "OK" if ok else "WARN"
    print(f"  {wname}: train={train_n:5d}  val={val_n:5d}  test={test_n:5d}  [{status}]")

# ==== 6. Full feature pipeline test ====
print("\n[6] Feature Pipeline End-to-End (W1)")
train_mask = (df['Date']>='2020/03/31')&(df['Date']<='2020/12/31')
df_t = df[train_mask].dropna(subset=['Next_Ret']).copy()

# Map
for idx, name in zip(RAW_IDX, NAMES):
    df_t[name] = pd.to_numeric(df_t.iloc[:, idx], errors='coerce')

# Winsorize
def win(s, lo=0.01, hi=0.99):
    if s.dropna().empty: return s
    return s.clip(*s.quantile([lo, hi]).values)

for name in NAMES:
    df_t[name] = df_t.groupby('Date')[name].transform(win)

# Rank
RANK = [('PE_TTM','R_PE','desc'),('PB','R_PB','desc'),('PS_TTM','R_PS','desc'),
        ('EV_EBITDA','R_EV','desc'),('Profit_Growth','R_Profit_Growth','asc'),
        ('Revenue2_Growth','R_Revenue_Growth','asc'),('Dividend_Yield','R_Dividend','asc'),
        ('MV','R_MV','asc')]
for src, dst, d in RANK:
    df_t[dst] = df_t.groupby('Date')[src].rank(pct=True, ascending=(d=='asc')).fillna(0.5)
df_t['MV_Log'] = np.log(df_t['MV'].clip(lower=0.01))
df_t['Value_Composite'] = df_t[['R_PE','R_PB','R_PS','R_EV']].mean(axis=1)
df_t['Growth_Composite'] = df_t[['R_Profit_Growth','R_Revenue_Growth']].mean(axis=1)
df_t['GARP_Signal'] = (df_t['Value_Composite']+df_t['Growth_Composite'])/2
df_t['Quality_Score'] = df_t[['R_Profit_Growth','R_Revenue_Growth','R_Dividend']].mean(axis=1)

ALL_F = NAMES + ['MV_Log'] + [d for _,d,_ in RANK] + ['Value_Composite','Growth_Composite','GARP_Signal','Quality_Score']
df_t['Y'] = (df_t['Next_Ret']>0).astype(int)

print(f"  Total features: {len(ALL_F)}")
missing_f = [f for f in ALL_F if f not in df_t.columns]
if missing_f:
    print(f"  MISSING FEATURES: {missing_f}")
else:
    print("  All features present")

# Impute and check
for f in ALL_F:
    df_t[f] = df_t[f].fillna(df_t[f].median() if df_t[f].notna().any() else 0)

# Check for NaN
na_count = df_t[ALL_F].isna().sum().sum()
print(f"  NaN after impute: {na_count}")

# Check for infinite
inf_count = np.isinf(df_t[ALL_F].values).sum()
print(f"  Inf after impute: {inf_count}")

# Quick model check
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = df_t[ALL_F].values
y = df_t['Y'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
m = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
m.fit(X_tr, y_tr)
p = m.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, p)
p_mean = p.mean()
p_max = p.max()
print(f"\n[7] Quick Model Sanity (Logistic Regression)")
print(f"  AUC: {auc:.4f}")
print(f"  P mean: {p_mean:.4f}, P max: {p_max:.4f}")
print(f"  P distribution: min={p.min():.4f}, Q25={np.percentile(p,25):.4f}, Q50={np.percentile(p,50):.4f}, Q75={np.percentile(p,75):.4f}")

if auc < 0.5:
    print("  WARNING: AUC < 0.5 — model worse than random!")
elif auc < 0.52:
    print("  CAUTION: AUC barely above 0.5 — signal is very weak")
else:
    print("  OK: AUC > 0.52")

if p_max < 0.62:
    print(f"  WARNING: Max predicted P = {p_max:.4f} < T_buy(0.62) — ALL stocks would be filtered out!")
    print(f"  SUGGESTION: Lower T_buy to {max(0.5, p_max-0.02):.2f} or improve feature engineering")

# ==== Issues summary ====
print("\n" + "=" * 60)
print("ISSUES FOUND:")
print("=" * 60)
if issues:
    for i in issues: print(i)
else:
    print("  No data quality issues")
if auc < 0.52:
    print("  MODEL: AUC too low, need better features or target engineering")
if p_max < 0.62:
    print(f"  MODEL: Max prob ({p_max:.3f}) < T_buy (0.62), will result in 0 holdings")
print("\nDone.")
