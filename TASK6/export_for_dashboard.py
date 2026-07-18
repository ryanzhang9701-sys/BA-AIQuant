#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export model predictions and parameters for interactive dashboard."""
import os, json, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_FILE = os.path.join(BASE_DIR, 'model_data.csv')

RAW_IDX = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
NAMES = ['EV_EBITDA','PB','PCF_NetCash','PCF_Operating','PE_TTM','PE_TTM_Deducted',
         'PS_TTM','Dividend_Yield','MV','Profit_Growth','NetAsset_Growth',
         'TotalProfit_Growth','EPS_Growth','TotalAsset_Growth','NetCash_Growth',
         'OperatingCF_Growth','OperatingProfit_Growth','Revenue1_Growth','Revenue2_Growth']
RANK = [('PE_TTM','R_PE','desc'),('PB','R_PB','desc'),('PS_TTM','R_PS','desc'),
        ('EV_EBITDA','R_EV','desc'),('Profit_Growth','R_Profit_Growth','asc'),
        ('Revenue2_Growth','R_Revenue_Growth','asc'),('Dividend_Yield','R_Dividend','asc'),
        ('MV','R_MV','asc')]
MODEL_NAMES = ['Logistic','RandomForest','XGBoost','LightGBM']

def load_clean():
    df = pd.read_csv(DATA_FILE, encoding='utf-8')
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y/%m/%d')
    rename_map = {}
    for c in df.columns:
        new = c.strip()
        if '(' in new: new = new.split('(')[0].strip()
        new = new.replace(' ','_').replace('/','_').replace('（','').replace('）','')
        rename_map[c] = new
    df.rename(columns=rename_map, inplace=True)
    df = df.dropna(subset=['Next_Ret']).copy()
    for idx, name in zip(RAW_IDX, NAMES):
        df[name] = pd.to_numeric(df.iloc[:, idx], errors='coerce')
    return df

def win(s, lo=0.01, hi=0.99):
    if s.dropna().empty: return s
    return s.clip(*s.quantile([lo, hi]).values)

def apply_features(df_sub):
    d = df_sub.copy()
    for n in NAMES:
        d[n] = d.groupby('Date')[n].transform(win)
    for src, dst, dir_ in RANK:
        d[dst] = d.groupby('Date')[src].rank(pct=True, ascending=(dir_=='asc')).fillna(0.5)
    d['MV_Log'] = np.log(d['MV'].clip(lower=0.01))
    d['Value_Composite'] = d[['R_PE','R_PB','R_PS','R_EV']].mean(axis=1)
    d['Growth_Composite'] = d[['R_Profit_Growth','R_Revenue_Growth']].mean(axis=1)
    d['GARP_Signal'] = (d['Value_Composite']+d['Growth_Composite'])/2
    d['Quality_Score'] = d[['R_Profit_Growth','R_Revenue_Growth','R_Dividend']].mean(axis=1)
    ALL_F = NAMES + ['MV_Log'] + [d_ for _,d_,_ in RANK] + ['Value_Composite','Growth_Composite','GARP_Signal','Quality_Score']
    for f in ALL_F:
        d[f] = d[f].fillna(d[f].median() if d[f].notna().any() else 0)
    return d, ALL_F

def extract_model_info(model_dict, calibrators, fnames):
    """Extract model-specific info for dashboard."""
    info = {}
    for mname in MODEL_NAMES:
        m = model_dict[mname]
        if mname == 'Logistic':
            coefs = [float(c) for c in m.coef_[0]]
            coef_list = [{'feature': f, 'coef': round(c, 6)} for f, c in zip(fnames, coefs)]
            coef_list.sort(key=lambda x: abs(x['coef']), reverse=True)
            info[mname] = {'type': 'coefficients', 'data': coef_list}
        else:
            if hasattr(m, 'feature_importances_'):
                imp = [float(v) for v in m.feature_importances_]
                imp_list = [{'feature': f, 'importance': round(v, 6)} for f, v in zip(fnames, imp)]
                imp_list.sort(key=lambda x: x['importance'], reverse=True)
                info[mname] = {'type': 'feature_importance', 'data': imp_list}
            else:
                info[mname] = {'type': 'none', 'data': []}
    return info

def main():
    print("Loading data...")
    df = load_clean()
    df['Y'] = (df['Next_Ret'] > 0).astype(int)

    all_predictions = []

    for wname in sorted(os.listdir(MODEL_DIR)):
        wpath = os.path.join(MODEL_DIR, wname)
        pkl = os.path.join(wpath, 'all_models.pkl')
        if not os.path.isfile(pkl): continue

        print(f"Processing {wname}...")
        bundle = joblib.load(pkl)
        models = bundle['models']
        calibrators = bundle['calibrators']
        b_val = bundle['b']
        ts, te, ve, tte = bundle['train_dates']

        train_mask = (df['Date']>=ts)&(df['Date']<=te)
        val_mask = (df['Date']>te)&(df['Date']<=ve)
        test_mask = (df['Date']>ve)&(df['Date']<=tte)

        df_train, feat_cols = apply_features(df[train_mask].copy())
        df_val, _ = apply_features(df[val_mask].copy())
        df_test, _ = apply_features(df[test_mask].copy())

        # For W1, save feature columns and model info
        if wname == 'W1':
            model_info = extract_model_info(models, calibrators, feat_cols)
            with open(os.path.join(OUTPUT_DIR, 'model_info.json'), 'w', encoding='utf-8') as f:
                json.dump({'feature_names': feat_cols, 'models': model_info}, f, ensure_ascii=False, indent=2)
            print(f"  Saved model_info.json ({len(feat_cols)} features)")

        # Predict for test set
        for mname in MODEL_NAMES:
            m = models[mname]
            cal = calibrators[mname]
            p_raw = m.predict_proba(df_test[feat_cols].values)[:, 1]
            p_raw_c = np.clip(p_raw, 1e-12, 1-1e-12)
            lo = np.log(p_raw_c/(1-p_raw_c))
            p_cal = cal.predict_proba(lo.reshape(-1,1))[:, 1]

            for i, (_, row) in enumerate(df_test.iterrows()):
                all_predictions.append({
                    'window': wname,
                    'model': mname,
                    'code': str(row['Code']),
                    'date': row['Date'],
                    'p_raw': round(float(p_raw[i]), 6),
                    'p_cal': round(float(p_cal[i]), 6),
                    'next_ret': round(float(row['Next_Ret']), 6),
                    'b': round(float(b_val), 4),
                })

    # Save predictions
    with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    print(f"Saved {len(all_predictions)} predictions to predictions.json")

if __name__ == '__main__':
    main()
