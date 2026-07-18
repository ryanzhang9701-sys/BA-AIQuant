#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化选股 ML 建模 — 特征工程 + Walk-Forward 训练 + 回测 + 模型对比
Spec: ml_model_spec.md
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                              accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import ParameterGrid
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'model_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy parameters — loaded from config file, user-editable
CONFIG_FILE = os.path.join(BASE_DIR, 'strategy_config.json')
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    STRATEGY = json.load(f)
print(f"Strategy: {STRATEGY}", flush=True)

# Walk-Forward windows
WINDOWS = [
    ('W1', '2020/03/31', '2020/12/31', '2021/03/31', '2021/06/30'),
    ('W2', '2020/03/31', '2021/03/31', '2021/06/30', '2021/09/30'),
    ('W3', '2020/03/31', '2021/06/30', '2021/09/30', '2021/12/31'),
    ('W4', '2020/03/31', '2021/09/30', '2021/12/31', '2022/03/31'),
    ('W5', '2020/03/31', '2021/12/31', '2022/03/31', '2022/06/30'),
]

# ============================================================
# DATA LOADING & CLEANING
# ============================================================
def load_and_clean(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    # Standardize date format: pd.to_datetime + strftime gives leading zeros (2020/03/31)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y/%m/%d')

    # Rename columns for safe access
    rename_map = {}
    for c in df.columns:
        new = c.strip()
        if '(' in new: new = new.split('(')[0].strip()
        new = new.replace(' ', '_').replace('/', '_').replace('（', '').replace('）', '')
        rename_map[c] = new
    df.rename(columns=rename_map, inplace=True)

    # Drop rows without Next_Ret
    df = df.dropna(subset=['Next_Ret']).copy()

    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================
RAW_FACTORS = [
    '企业倍数_EV', '市净率PB', '市现率PCF', '市现率PCF_经营',
    '市盈率PE', '市盈率PE_扣非', '市销率PS', '股息率',
    'MV', '净利润同比增长率', '净资产同比增长率', '利润总额同比增长率',
    '基本每股收益同比增长率', '总资产同比增长率', '现金净流量同比增长率',
    '经营活动产生的现金流量净额同比增长率', '营业利润同比增长率',
    '营业总收入同比增长率', '营业收入同比增长率',
]

FEAT_COL_NAMES = [
    'EV_EBITDA', 'PB', 'PCF_NetCash', 'PCF_Operating',
    'PE_TTM', 'PE_TTM_Deducted', 'PS_TTM', 'Dividend_Yield',
    'MV', 'Profit_Growth', 'NetAsset_Growth', 'TotalProfit_Growth',
    'EPS_Growth', 'TotalAsset_Growth', 'NetCash_Growth',
    'OperatingCF_Growth', 'OperatingProfit_Growth',
    'Revenue1_Growth', 'Revenue2_Growth',
]

# Column indices in the renamed dataframe (0-based, after Date/Code)
RAW_COL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

RANK_TARGETS = [
    ('PE_TTM', 'R_PE', 'desc'),
    ('PB', 'R_PB', 'desc'),
    ('PS_TTM', 'R_PS', 'desc'),
    ('EV_EBITDA', 'R_EV', 'desc'),
    ('Profit_Growth', 'R_Profit_Growth', 'asc'),
    ('Revenue2_Growth', 'R_Revenue_Growth', 'asc'),
    ('Dividend_Yield', 'R_Dividend', 'asc'),
    ('MV', 'R_MV', 'asc'),
]

COMPOSITES = [
    ('Value_Composite', ['R_PE', 'R_PB', 'R_PS', 'R_EV']),
    ('Growth_Composite', ['R_Profit_Growth', 'R_Revenue_Growth']),
    ('GARP_Signal', ['Value_Composite', 'Growth_Composite']),
    ('Quality_Score', ['R_Profit_Growth', 'R_Revenue_Growth', 'R_Dividend']),
]


def winsorize_series(s, lo=0.01, hi=0.99):
    if s.dropna().empty: return s
    lo_v, hi_v = s.quantile(lo), s.quantile(hi)
    return s.clip(lo_v, hi_v)


def build_features(df):
    """Build all 31 features. Returns df with features added."""
    df = df.copy()

    # ---- Level 1: Raw factors (use positional indices to avoid duplicate name issues) ----
    for idx, name in zip(RAW_COL_INDICES, FEAT_COL_NAMES):
        df[name] = pd.to_numeric(df.iloc[:, idx], errors='coerce')

    # Winsorize (within each date)
    for name in FEAT_COL_NAMES:
        df[name] = df.groupby('Date')[name].transform(winsorize_series)

    # MV log
    df['MV_Log'] = np.log(df['MV'].clip(lower=0.01))

    # ---- Level 2: Rank normalize (done per split, placeholder here) ----
    for src, dst, direction in RANK_TARGETS:
        df[dst] = np.nan

    # ---- Level 3: Composites ----
    for name, components in COMPOSITES:
        cols_present = [c for c in components]
        if cols_present:
            df[name] = np.nan

    return df


def apply_rank_normalize(df):
    """Apply rank normalization within each Date group. Modifies df in-place."""
    for src, dst, direction in RANK_TARGETS:
        if src not in df.columns: continue
        if direction == 'desc':
            df[dst] = df.groupby('Date')[src].rank(pct=True, ascending=False)
        else:
            df[dst] = df.groupby('Date')[src].rank(pct=True, ascending=True)
        df[dst] = df[dst].fillna(0.5)

    for name, components in COMPOSITES:
        cols_present = [c for c in components if c in df.columns]
        if cols_present:
            df[name] = df[cols_present].mean(axis=1)

    return df


def get_feature_columns():
    """Return list of all 31 feature column names."""
    base = FEAT_COL_NAMES + ['MV_Log']
    ranks = [dst for _, dst, _ in RANK_TARGETS]
    comps = [name for name, _ in COMPOSITES]
    return base + ranks + comps


# ============================================================
# MODEL TRAINING
# ============================================================
def train_logistic(X_train, y_train, X_val, y_val):
    """Logistic Regression with L2 regularization."""
    best_score, best_model = -np.inf, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
        m.fit(X_train, y_train)
        score = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if score > best_score:
            best_score, best_model = score, m
    return best_model, best_score


def train_rf(X_train, y_train, X_val, y_val):
    """Random Forest with grid search."""
    best_score, best_model = -np.inf, None
    grid = list(ParameterGrid({
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt', None],
    }))[::3]  # Subsample grid for speed
    for params in grid[:8]:
        m = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        score = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        if score > best_score:
            best_score, best_model = score, m
    return best_model, best_score


def train_xgb(X_train, y_train, X_val, y_val):
    """XGBoost with grid search."""
    from xgboost import XGBClassifier
    best_score, best_model = -np.inf, None
    for lr in [0.05, 0.1]:
        for depth in [3, 5]:
            m = XGBClassifier(max_depth=depth, learning_rate=lr, n_estimators=200,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=1.0,
                              eval_metric='logloss', random_state=42, verbosity=0)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
            if score > best_score:
                best_score, best_model = score, m
    return best_model, best_score


def train_lgb(X_train, y_train, X_val, y_val):
    """LightGBM with grid search."""
    from lightgbm import LGBMClassifier
    best_score, best_model = -np.inf, None
    for lr in [0.05, 0.1]:
        for leaves in [15, 31]:
            m = LGBMClassifier(num_leaves=leaves, learning_rate=lr, n_estimators=200,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.1, reg_lambda=1.0,
                               random_state=42, verbose=-1)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            score = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
            if score > best_score:
                best_score, best_model = score, m
    return best_model, best_score


def calibrate_model(model, X_val, y_val):
    """Platt scaling — fit a sigmoid on model's raw predictions. Much faster than cv=3."""
    p_raw = model.predict_proba(X_val)[:, 1]
    # Avoid log(0) / log(1-p) issues
    p_raw = np.clip(p_raw, 1e-12, 1 - 1e-12)
    log_odds = np.log(p_raw / (1 - p_raw))
    cal_lr = LogisticRegression(C=1000, solver='lbfgs', max_iter=500)
    cal_lr.fit(log_odds.reshape(-1, 1), y_val)
    return cal_lr


# ============================================================
# BACKTEST
# ============================================================
def estimate_b(train_df):
    """Estimate odds ratio b from training data."""
    pos = train_df[train_df['Next_Ret'] > 0]['Next_Ret']
    neg = train_df[train_df['Next_Ret'] <= 0]['Next_Ret']
    if len(pos) == 0 or len(neg) == 0: return 1.0
    b_val = pos.mean() / abs(neg.mean())
    return max(b_val, 0.5)


def compute_kelly_position(p_win, b, alpha, cap_stock):
    """Compute final position weight for a single stock."""
    f_kelly = max(0, (b * p_win - (1 - p_win)) / b)
    f_raw = f_kelly * alpha
    return min(f_raw, cap_stock)


def run_backtest_window(test_df, model, calibrator, b, strategy):
    """Run one window's backtest. Returns portfolio return and positions."""
    T_buy = strategy['T_buy']
    T_sell = strategy['T_sell']
    alpha = strategy['alpha']
    cap_stock = strategy['cap_stock']
    N_max = strategy['N_max']
    beta = strategy['beta']

    if len(test_df) == 0: return 0.0, pd.DataFrame()

    X_test = test_df[FEATURE_COLS].values
    p_raw = model.predict_proba(X_test)[:, 1]
    # Platt calibration: convert probabilities to log-odds, then apply calibrator
    p_raw_clip = np.clip(p_raw, 1e-12, 1 - 1e-12)
    log_odds_test = np.log(p_raw_clip / (1 - p_raw_clip))
    p_cal = calibrator.predict_proba(log_odds_test.reshape(-1, 1))[:, 1]
    test_df = test_df.copy()
    test_df['p_raw'] = p_raw
    test_df['p_cal'] = p_cal

    # Filter candidates
    candidates = test_df[test_df['p_cal'] > T_buy].copy()
    candidates = candidates.sort_values('p_cal', ascending=False).head(N_max)

    if len(candidates) == 0: return 0.0, candidates

    # Position sizing
    candidates['f_kelly'] = candidates['p_cal'].apply(
        lambda p: max(0, (b * p - (1 - p)) / b))
    candidates['f_raw'] = candidates['f_kelly'] * alpha
    candidates['f_raw'] = candidates['f_raw'].clip(upper=cap_stock)

    # Total position scaling
    F_raw = candidates['f_raw'].sum()
    if F_raw > beta and F_raw > 0:
        scale = beta / F_raw
        candidates['f_final'] = candidates['f_raw'] * scale
    else:
        candidates['f_final'] = candidates['f_raw']

    # Portfolio return
    port_ret = (candidates['f_final'] * candidates['Next_Ret']).sum()

    return port_ret, candidates


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    import sys
    print("=" * 60, flush=True)
    print("量化选股 ML 建模 — Walk-Forward Backtest")
    print("=" * 60)

    # ---- Load ----
    print("\n[1/5] Loading data...")
    df = load_and_clean(DATA_FILE)
    print(f"  Loaded: {len(df)} rows, {df['Code'].nunique()} stocks")

    # ---- Features ----
    print("\n[2/5] Building features...")
    df = build_features(df)
    df['Y'] = (df['Next_Ret'] > 0).astype(int)
    global FEATURE_COLS
    FEATURE_COLS = get_feature_columns()
    print(f"  Features: {len(FEATURE_COLS)}")

    # ---- Walk-Forward ----
    print("\n[3/5] Walk-Forward training & backtest...")
    all_results = []
    all_positions = []
    model_metrics = []

    for wname, train_start, train_end, val_end, test_end in WINDOWS:
        print(f"\n  --- {wname} ---", flush=True)
        train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
        val_mask = (df['Date'] > train_end) & (df['Date'] <= val_end)
        test_mask = (df['Date'] > val_end) & (df['Date'] <= test_end)

        df_train = apply_rank_normalize(df[train_mask].copy())
        df_val = apply_rank_normalize(df[val_mask].copy())
        df_test = apply_rank_normalize(df[test_mask].copy())

        # Impute missing
        for _df in [df_train, df_val, df_test]:
            for c in FEATURE_COLS:
                if c in _df.columns:
                    med = _df[c].median()
                    _df[c] = _df[c].fillna(med if not np.isnan(med) else 0)

        X_train = df_train[FEATURE_COLS].values
        y_train = df_train['Y'].values
        X_val = df_val[FEATURE_COLS].values
        y_val = df_val['Y'].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print(f"    Skipping {wname}: classes train={np.unique(y_train)}, val={np.unique(y_val)}, sizes=({len(y_train)},{len(y_val)})")
            continue

        # Train 4 models
        models = {}
        print(f"    Training Logistic...", flush=True)
        models['Logistic'], _ = train_logistic(X_train, y_train, X_val, y_val)
        print(f"    Training RandomForest...", flush=True)
        models['RandomForest'], _ = train_rf(X_train, y_train, X_val, y_val)
        print(f"    Training XGBoost...", flush=True)
        models['XGBoost'], _ = train_xgb(X_train, y_train, X_val, y_val)
        print(f"    Training LightGBM...", flush=True)
        models['LightGBM'], _ = train_lgb(X_train, y_train, X_val, y_val)

        # Calibrate
        calibrators = {}
        for name, m in models.items():
            calibrators[name] = calibrate_model(m, X_val, y_val)

        # Estimate b
        b = estimate_b(df_train)

        # Backtest each model
        for mname in models:
            port_ret, positions = run_backtest_window(
                df_test, models[mname], calibrators[mname], b, STRATEGY)
            all_results.append({
                'Window': wname, 'Model': mname,
                'Test_Date': test_end, 'Portfolio_Return': port_ret,
                'N_Holdings': len(positions),
                'Avg_P_cal': positions['p_cal'].mean() if len(positions) > 0 else 0,
            })
            print(f"    {mname}: return={port_ret:.4f}, holdings={len(positions)}", flush=True)

            # Validation metrics
            p_raw_val = models[mname].predict_proba(X_val)[:, 1]
            p_raw_val_clip = np.clip(p_raw_val, 1e-12, 1 - 1e-12)
            lo_val = np.log(p_raw_val_clip / (1 - p_raw_val_clip))
            p_val = calibrators[mname].predict_proba(lo_val.reshape(-1, 1))[:, 1]
            model_metrics.append({
                'Window': wname, 'Model': mname,
                'AUC_Val': roc_auc_score(y_val, p_val),
                'LogLoss_Val': log_loss(y_val, p_val),
                'Brier_Val': brier_score_loss(y_val, p_val),
            })

        # Save models
        wdir = os.path.join(MODEL_DIR, wname)
        os.makedirs(wdir, exist_ok=True)
        joblib.dump({'models': models, 'calibrators': calibrators, 'b': b,
                     'train_dates': (train_start, train_end, val_end, test_end)},
                    os.path.join(wdir, 'all_models.pkl'))
        print(f"    Saved to {wdir}/all_models.pkl", flush=True)

    # ---- Aggregate ----
    print("\n[4/5] Aggregating results...")
    df_results = pd.DataFrame(all_results)
    df_metrics = pd.DataFrame(model_metrics)

    # Per-model summary
    summary = df_results.groupby('Model').agg(
        Cumulative_Return=('Portfolio_Return', 'sum'),
        Mean_Quarterly_Return=('Portfolio_Return', 'mean'),
        Std_Quarterly_Return=('Portfolio_Return', 'std'),
        Avg_Holdings=('N_Holdings', 'mean'),
        Win_Rate=('Portfolio_Return', lambda x: (x > 0).mean()),
    ).reset_index()

    # Sharpe (annualized, quarterly)
    summary['Sharpe'] = summary['Mean_Quarterly_Return'] / summary['Std_Quarterly_Return'].clip(lower=1e-6) * 2

    # Max drawdown per model
    for model_name in summary['Model']:
        returns = df_results[df_results['Model'] == model_name]['Portfolio_Return'].values
        cum = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        summary.loc[summary['Model'] == model_name, 'MaxDD'] = abs(dd.min())

    summary['Calmar'] = summary['Mean_Quarterly_Return'] * 4 / summary['MaxDD'].clip(lower=1e-6)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"  {row['Model']:15s} | CumRet={row['Cumulative_Return']:.3f} | "
              f"Sharpe={row['Sharpe']:.2f} | MaxDD={row['MaxDD']:.2%} | "
              f"WinRate={row['Win_Rate']:.0%}")

    # Save
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'), index=False, encoding='utf-8-sig')
    summary.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'), index=False, encoding='utf-8-sig')

    # Save best model meta
    if summary['Sharpe'].notna().any():
        best = summary.loc[summary['Sharpe'].idxmax()]
    else:
        best = summary.iloc[0]
    rankings = summary.sort_values('Sharpe', ascending=False, na_position='last').reset_index(drop=True)
    rankings['Rank'] = range(1, len(rankings) + 1)
    meta = {
        'best_model': best['Model'],
        'best_sharpe': float(best['Sharpe']),
        'best_cumret': float(best['Cumulative_Return']),
        'model_rankings': rankings[['Model', 'Sharpe', 'Cumulative_Return', 'MaxDD', 'Rank']].to_dict('records'),
        'strategy_params': STRATEGY,
    }
    with open(os.path.join(MODEL_DIR, 'best_model_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n  Best model: {meta['best_model']} (Sharpe={meta['best_sharpe']:.2f})")
    print(f"\n[5/5] Results saved to {OUTPUT_DIR}/")

    return df_results, summary, df_metrics


if __name__ == '__main__':
    main()
