"""Generate MA Crossover + MA200 Trend Filter variant — Notebook + Dashboard"""
import os, json, pandas as pd, numpy as np
import nbformat as nbf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, 'data')
OUT  = os.path.join(PROJ, 'TASK3')
os.makedirs(OUT, exist_ok=True)

NAME_MAP = {
    '002594.SZ': 'BYD', '002594.HK': 'BYD(HK)',
    '603986.SH': 'GigaDevice', '688981.SH': 'SMIC', '688981.HK': 'SMIC(HK)',
}

# Load all stocks
stocks = []
for dname in sorted(os.listdir(DATA)):
    dpath = os.path.join(DATA, dname)
    if not os.path.isdir(dpath): continue
    csv_a = os.path.join(dpath, 'daily_adjusted.csv')
    csv_hk = os.path.join(dpath, 'daily_hk.csv')
    parts = dname.split('_')
    code = parts[0] if parts else 'unknown'
    if os.path.exists(csv_a):
        df = pd.read_csv(csv_a, encoding='utf-8-sig', parse_dates=['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        sid = code
        stocks.append(dict(id=sid, label=f'{sid} {NAME_MAP.get(sid,"")}',
            market='A', data=df[['trade_date','close_qfq']].copy(), price_col='close_qfq'))
    if os.path.exists(csv_hk):
        df = pd.read_csv(csv_hk, encoding='utf-8-sig', parse_dates=['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        sid = code.split('.')[0] + '.HK'
        stocks.append(dict(id=sid, label=f'{sid} {NAME_MAP.get(sid,"")}',
            market='HK', data=df[['trade_date','close']].copy(), price_col='close'))

print(f'Loaded {len(stocks)} stocks')

# Stock data for embedding
stock_data = {}
for s in stocks:
    df = s['data']
    stock_data[s['id']] = dict(
        label=s['label'], market=s['market'],
        dates=df['trade_date'].dt.strftime('%Y-%m-%d').tolist(),
        price=df[s['price_col']].round(2).tolist())

DATA_JSON = json.dumps(stock_data, ensure_ascii=False)
default_id = '603986.SH' if '603986.SH' in stock_data else list(stock_data.keys())[0]

stock_options = '\n'.join(
    f'<option value="{sid}" {"selected" if sid==default_id else ""}>{info["label"]}</option>'
    for sid, info in stock_data.items())

# ============================================================
# 1. Generate Notebook
# ============================================================
nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13.0'}
}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def cd(s): cells.append(nbf.v4.new_code_cell(s))

md('''# MA Crossover + MA200 Trend Filter — Backtest

> Variant: Trend Filter (spec 7.2) | Base: MA_CROSSOVER_V1
> Logic: Golden Cross is only valid when price > MA200

## How it works

The base MA crossover generates a golden cross whenever MA_fast crosses above MA_slow.
In a downtrend, many of these golden crosses are false — they are dead-cat bounces.

The MA200 trend filter adds one simple rule: **only buy when price is above MA200**.
This eliminates all golden crosses that occur during bear markets.

## Pipeline

1. Load Data -> 2. Calculate MA(fast) + MA(slow) + MA200
3. Generate signals: golden cross only counts if price > MA200
4. Run backtest with costs -> 5. Metrics -> 6. Visualize

## Controls

- CHOSEN: switch stocks (603986.SH / 002594.SZ / 688981.SH / HK variants)
- FAST_PERIOD / SLOW_PERIOD: tune MA parameters
- MA200_PERIOD: trend filter period (default 200)
- ENABLE_FILTER: toggle trend filter on/off to see the difference''')

cd('''# ============================================================
# 1. Parameters
# ============================================================
FAST_PERIOD     = 10
SLOW_PERIOD     = 60
MA200_PERIOD    = 200       # Trend filter period
MA_TYPE         = "ema"
ENABLE_FILTER   = True      # True = with MA200 trend filter, False = base strategy

INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE  = 0.0005
SLIPPAGE_RATE   = 0.001

print(f"Strategy: {MA_TYPE.upper()}({FAST_PERIOD})/{MA_TYPE.upper()}({SLOW_PERIOD})")
print(f"Trend Filter: {'ON (price > MA200)' if ENABLE_FILTER else 'OFF (base strategy)'}")
print(f"Costs: comm={COMMISSION_RATE:.3%} stamp={STAMP_TAX_RATE:.3%} slip={SLIPPAGE_RATE:.1%}")''')

cd('''# ============================================================
# 2. Load Data
# ============================================================
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(PROJECT, "data")

STOCKS = {}
for dname in sorted(os.listdir(DATA_DIR)):
    dpath = os.path.join(DATA_DIR, dname)
    if not os.path.isdir(dpath): continue
    for fname in ['daily_adjusted.csv', 'daily_hk.csv']:
        fp = os.path.join(dpath, fname)
        if os.path.exists(fp):
            code = dname.split('_')[0]
            label = f"{code} ({'A' if 'adjusted' in fname else 'HK'})"
            STOCKS[label] = fp

CHOSEN = list(STOCKS.keys())[2]  # Default: 603986.SH (GigaDevice)

df = pd.read_csv(STOCKS[CHOSEN], encoding="utf-8-sig", parse_dates=["trade_date"])
df = df.sort_values("trade_date").reset_index(drop=True)
col = "close_qfq" if "close_qfq" in df.columns else "close"
df["price"] = df[col]
print(f"Stock: {CHOSEN} | Rows: {len(df)} | {df.trade_date.min().date()} ~ {df.trade_date.max().date()}")''')

cd('''# ============================================================
# 3. Moving Averages (including MA200)
# ============================================================
def calc_ma(s, n, t):
    return s.ewm(span=n, adjust=False).mean() if t == "ema" else s.rolling(n).mean()

df["ma_fast"] = calc_ma(df["price"], FAST_PERIOD, MA_TYPE)
df["ma_slow"] = calc_ma(df["price"], SLOW_PERIOD, MA_TYPE)
df["ma200"]   = calc_ma(df["price"], MA200_PERIOD, "sma")  # MA200 always SMA

# Warmup: only drop the minimum needed for fast/slow MAs.
# MA200 will have NaN for the first 200 bars -- that's fine for chart display.
# When filter is ON, bars with NaN MA200 are treated as "filter not available, safe to skip".
warmup = max(FAST_PERIOD, SLOW_PERIOD)
df_v = df.iloc[warmup:].copy().reset_index(drop=True)

fast_lbl = f"{MA_TYPE.upper()}({FAST_PERIOD})"
slow_lbl = f"{MA_TYPE.upper()}({SLOW_PERIOD})"
print(f"Warmup: {warmup} bars | Valid: {len(df_v)} rows")
print(f"MA200 latest: {df_v['ma200'].iloc[-1]:.2f}")
print(f"Price vs MA200: {df_v['price'].iloc[-1]:.2f} {'>' if df_v['price'].iloc[-1] > df_v['ma200'].iloc[-1] else '<'} {df_v['ma200'].iloc[-1]:.2f}")''')

cd('''# ============================================================
# 4. Signal Generation (with MA200 trend filter)
# ============================================================
# Base cross signal
df_v["signal_raw"] = (df_v["ma_fast"] > df_v["ma_slow"]).astype(int)
df_v["cross_raw"]  = df_v["signal_raw"].diff()

# Trend filter: price must be above MA200 for golden cross.
# When MA200 is NaN (not enough history), accept the signal.
if ENABLE_FILTER:
    df_v["trend_ok"] = ((df_v["price"] > df_v["ma200"]) | df_v["ma200"].isna()).astype(int)
    df_v["signal"] = (df_v["signal_raw"] & df_v["trend_ok"]).astype(int)
else:
    df_v["signal"] = df_v["signal_raw"]

df_v["cross"]    = df_v["signal"].diff()
df_v["position"] = df_v["signal"].shift(1).fillna(0).astype(int)

n_g_raw = int((df_v["cross_raw"] == 1).sum())
n_g     = int((df_v["cross"] == 1).sum())
n_d     = int((df_v["cross"] == -1).sum())
print(f"Raw golden crosses: {n_g_raw}")
print(f"Filtered golden crosses: {n_g} (filtered out {n_g_raw - n_g})")
print(f"Death crosses: {n_d} | Trades: {n_g}")

# Show filtered signals
changes = df_v[df_v["cross_raw"] != 0].copy()
changes["raw_label"] = changes["cross_raw"].map({1: "RAW Golden", -1: "RAW Death"})
changes["filtered"]  = df_v.loc[changes.index, "cross"].map({1: "Valid", -1: "Death", 0: "FILTERED"})
changes[["trade_date","price","ma200","raw_label","filtered"]].head(15)''')

cd('''# ============================================================
# 5. Backtest
# ============================================================
df_v["price_ret"] = df_v["price"].pct_change()
df_v["gross_ret"] = df_v["position"] * df_v["price_ret"]
df_v["trade_act"] = df_v["position"].diff().abs()
df_v["cost"]      = df_v["trade_act"] * (COMMISSION_RATE + SLIPPAGE_RATE)
df_v.loc[df_v["position"].diff() < 0, "cost"] += STAMP_TAX_RATE
df_v["strat_ret"] = df_v["gross_ret"] - df_v["cost"]
df_v["equity"]    = (1 + df_v["strat_ret"].fillna(0)).cumprod()
df_v["bench_eq"]  = (1 + df_v["price_ret"].fillna(0)).cumprod()
pk_s = df_v["equity"].cummax(); pk_b = df_v["bench_eq"].cummax()
df_v["dd"]       = (df_v["equity"] - pk_s) / pk_s
df_v["bench_dd"] = (df_v["bench_eq"] - pk_b) / pk_b
print(f"Strategy: {df_v.equity.iloc[-1]:.4f} ({df_v.equity.iloc[-1]-1:+.2%})")
print(f"Benchmark: {df_v.bench_eq.iloc[-1]:.4f} ({df_v.bench_eq.iloc[-1]-1:+.2%})")''')

cd('''# ============================================================
# 6. Performance Metrics
# ============================================================
ret = df_v["strat_ret"].dropna(); N = len(ret)
total_ret = df_v["equity"].iloc[-1] - 1
bench_ret = df_v["bench_eq"].iloc[-1] - 1
ann_ret   = (1+total_ret)**(252/N)-1
ann_bench = (1+bench_ret)**(252/N)-1
rc = ret[ret!=0] if len(ret[ret!=0])>0 else ret
sharpe = np.sqrt(252)*rc.mean()/rc.std() if rc.std()>0 else 0
max_dd = df_v["dd"].min(); max_dd_b = df_v["bench_dd"].min()
calmar = ann_ret/abs(max_dd) if max_dd!=0 else 0

g_entry = df_v[df_v["cross"]==1]
trades_ret = []
for i in range(len(g_entry)):
    ei = df_v[df_v["trade_date"]==g_entry["trade_date"].iloc[i]].index[0]
    xi = len(df_v)-1 if i+1>=len(g_entry) else df_v[df_v["trade_date"]==g_entry["trade_date"].iloc[i+1]].index[0]-1
    trades_ret.append(df_v.loc[xi,"price"]/df_v.loc[ei,"price"]-1)
wins=[r for r in trades_ret if r>0]; loses=[r for r in trades_ret if r<=0]
wr=len(wins)/len(trades_ret) if trades_ret else 0
aw=np.mean(wins) if wins else 0
al=np.mean([abs(r) for r in loses]) if loses else 0
plr=aw/al if al>0 else float("inf")

from IPython.display import display, HTML
display(HTML(f"""<table style="border-collapse:collapse;width:100%;font-size:14px">
<tr style="background:#f8f9fa;font-weight:bold"><td style="padding:10px 16px">Metric</td><td style="padding:10px 16px;text-align:right">Filtered Strategy</td><td style="padding:10px 16px;text-align:right">Benchmark</td></tr>
<tr><td>Total Return</td><td style="text-align:right;color:{'#D85A30' if total_ret>0 else '#1D9E75'}">{total_ret:+.2%}</td><td style="text-align:right;color:{'#D85A30' if bench_ret>0 else '#1D9E75'}">{bench_ret:+.2%}</td></tr>
<tr style="background:#f8f9fa"><td>Annual Return</td><td style="text-align:right">{ann_ret:+.2%}</td><td style="text-align:right">{ann_bench:+.2%}</td></tr>
<tr><td>Sharpe (ann)</td><td style="text-align:right;font-weight:bold">{sharpe:.2f}</td><td style="text-align:right">-</td></tr>
<tr style="background:#f8f9fa"><td>Max Drawdown</td><td style="text-align:right;color:#D85A30">{max_dd:+.2%}</td><td style="text-align:right;color:#D85A30">{max_dd_b:+.2%}</td></tr>
<tr><td>Calmar Ratio</td><td style="text-align:right;font-weight:bold">{calmar:.2f}</td><td style="text-align:right">-</td></tr>
<tr style="background:#f8f9fa"><td>Win Rate</td><td style="text-align:right">{wr:.1%}</td><td style="text-align:right">-</td></tr>
<tr><td>P/L Ratio</td><td style="text-align:right;font-weight:bold">{plr:.2f}</td><td style="text-align:right">-</td></tr>
<tr style="background:#f8f9fa"><td>Total Trades</td><td style="text-align:right">{len(trades_ret)}</td><td style="text-align:right">-</td></tr>
</table>"""))''')

cd('''# ============================================================
# 7. Visualization
# ============================================================
import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(3,1,figsize=(16,12),gridspec_kw={"height_ratios":[2,1.5,1]})
dts = df_v["trade_date"]

ax=axes[0]
ax.plot(dts, df_v["price"], "#B4B2A9", lw=0.8, alpha=0.7, label="Close")
ax.plot(dts, df_v["ma_fast"], "#D85A30", lw=1.2, label=fast_lbl)
ax.plot(dts, df_v["ma_slow"], "#534AB7", lw=1.2, label=slow_lbl)
ax.plot(dts, df_v["ma200"], "#378ADD", lw=1.5, ls="dotted", alpha=0.7, label="MA200")
ax.fill_between(dts, 0, df_v["price"].max()*1.2, where=(df_v["price"]>df_v["ma200"]), alpha=0.05, color="#378ADD")
g=df_v[df_v["cross"]==1]; d=df_v[df_v["cross"]==-1]
ax.scatter(g["trade_date"], g["price"], c="#1D9E75", marker="^", s=80, zorder=5, label="Golden (filtered)")
ax.set_title(f"{CHOSEN} - Price & MA + MA200 Filter", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=8); ax.set_ylabel("Price"); ax.grid(True,alpha=0.3)

ax=axes[1]
ax.fill_between(dts, 1, df_v["equity"], alpha=0.1, color="#378ADD")
ax.plot(dts, df_v["equity"], "#378ADD", lw=1.8, label="Filtered Strategy")
ax.plot(dts, df_v["bench_eq"], "#B4B2A9", lw=1.2, ls="dashed", label="Benchmark")
ax.axhline(1, c="black", lw=0.5, ls="dotted")
ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9); ax.set_ylabel("Equity"); ax.grid(True,alpha=0.3)

ax=axes[2]
ax.fill_between(dts, df_v["dd"], 0, alpha=0.3, color="#D85A30")
ax.fill_between(dts, df_v["bench_dd"], 0, alpha=0.15, color="#B4B2A9")
ax.plot(dts, df_v["dd"], "#D85A30", lw=1, label="Strategy DD")
ax.plot(dts, df_v["bench_dd"], "#B4B2A9", lw=0.8, ls="dashed", label="Benchmark DD")
ax.axhline(0, c="black", lw=0.3); ax.axhline(-0.25, c="#D85A30", lw=0.5, ls="dotted", alpha=0.5)
ax.set_title("Drawdown", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=9); ax.set_ylabel("Drawdown"); ax.grid(True,alpha=0.3)
for a in axes:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()''')

cd(f'''# ============================================================
# 8. Summary
# ============================================================
filter_label = "ON (price > MA200)" if ENABLE_FILTER else "OFF (base strategy)"
print(f"""
{{'='*60}}
  MA Crossover + Trend Filter Backtest: {{CHOSEN}}
  {{MA_TYPE.upper()}}({{FAST_PERIOD}})/{{MA_TYPE.upper()}}({{SLOW_PERIOD}}) with MA200 filter
  Filter: {{filter_label}}
  Return: {{total_ret:+.2%}} | Sharpe: {{sharpe:.2f}} | MaxDD: {{max_dd:+.2%}} | WinRate: {{wr:.1%}}
  Trades: {{len(trades_ret)}} | P/L Ratio: {{plr:.2f}}
{{'='*60}}

Set ENABLE_FILTER=False to compare with base strategy.
Switch CHOSEN to change stocks.
Based on ma_crossover_strategy_spec.md variant 7.2
""")''')

nb.cells = cells
nb_path = os.path.join(OUT, 'ma_crossover_trend_filter.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook: {nb_path} ({len(cells)} cells)')

# ============================================================
# 2. Generate Dashboard HTML
# ============================================================
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MA Crossover + Trend Filter Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root{{--bg:#fff;--bg2:#f8f9fa;--t:#1a1a2e;--t2:#555;--bd:#e0e0e0;--gn:#1D9E75;--rd:#D85A30;--pu:#534AB7;--bl:#378ADD;--ra:10px}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:var(--t);background:#f5f5f5;line-height:1.7}}
.hd{{background:#fff;border-bottom:1px solid var(--bd);padding:10px 24px}}
.hd h1{{font-size:18px;font-weight:600;margin-bottom:4px}}
.hd .sub{{font-size:11px;color:#888;margin-bottom:8px}}
.ctrls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px}}
.ctrls select{{padding:4px 10px;border:1px solid var(--bd);border-radius:6px;font-size:12px;background:#fff;min-width:130px}}
.ctrls input[type=number]{{padding:4px 6px;border:1px solid var(--bd);border-radius:6px;font-size:12px;width:52px;text-align:center;background:#fff}}
.ctrls label{{color:#888;font-size:11px;white-space:nowrap}}
.ctrls .sep{{width:1px;height:18px;background:var(--bd);margin:0 2px}}
.presets{{display:flex;gap:3px}}
.presets button{{padding:4px 9px;border:1px solid var(--bd);border-radius:14px;font-size:11px;cursor:pointer;background:#fff;transition:all .12s;white-space:nowrap}}
.presets button:hover{{border-color:var(--pu);color:var(--pu)}}
.presets button.active{{background:var(--pu);color:#fff;border-color:var(--pu)}}
.toggle-row{{display:flex;gap:12px;align-items:center;font-size:11px}}
.toggle-row label{{display:flex;align-items:center;gap:3px;cursor:pointer;color:var(--t2)}}
.toggle-row input[type=checkbox]{{accent-color:var(--pu)}}
.btn-go{{padding:4px 16px;background:var(--pu);color:#fff;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500}}
.filter-indicator{{display:inline-block;padding:1px 10px;border-radius:10px;font-size:11px;font-weight:500}}
.filter-on{{background:#E1F5EE;color:#0F6E56}}.filter-off{{background:#FAECE7;color:#D85A30}}
.container{{max-width:1200px;margin:0 auto;padding:12px 20px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:6px;margin-bottom:12px}}
.mt{{background:#fff;border:1px solid var(--bd);border-radius:var(--ra);padding:10px;text-align:center}}
.mt .lbl{{font-size:10px;color:#888;margin-bottom:1px}}.mt .val{{font-size:16px;font-weight:600}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:var(--ra);padding:12px;margin-bottom:10px}}
.card h3{{font-size:13px;font-weight:600;margin-bottom:8px;padding-left:6px;border-left:3px solid var(--pu)}}
.chart{{width:100%;height:320px}}
.chart-sm{{width:100%;height:280px}}
.up{{color:var(--rd)}}.down{{color:var(--gn)}}.neutral{{color:var(--t)}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.badge{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:10px;font-weight:500}}
.badge.win{{background:#E1F5EE;color:#0F6E56}}.badge.loss{{background:#FAECE7;color:#D85A30}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:var(--bg2);text-align:left;padding:5px 10px;font-weight:600;border-bottom:2px solid var(--bd)}}
td{{padding:4px 10px;border-bottom:1px solid var(--bd)}}tr:hover td{{background:#fafafa}}
.foot{{padding:8px 14px;font-size:11px;color:#888;text-align:center}}
@media(max-width:768px){{.grid-2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<div class="hd">
  <h1>MA Crossover + MA200 Trend Filter</h1>
  <div class="ctrls">
    <label>Stock</label>
    <select id="stockSelect" onchange="runBacktest()">{stock_options}</select>
    <span class="sep"></span>
    <label>MA</label>
    <select id="maType" onchange="runBacktest()">
      <option value="ema" selected>EMA</option><option value="sma">SMA</option>
    </select>
    <label>Fast</label><input type="number" id="fastP" value="10" min="3" max="50" onchange="validateParams()">
    <label>Slow</label><input type="number" id="slowP" value="60" min="10" max="300" onchange="validateParams()">
    <div class="presets" id="pGroup">
      <button onclick="setPreset(5,20)" data-p="agg">5/20</button>
      <button class="active" onclick="setPreset(10,60)" data-p="bal">10/60</button>
      <button onclick="setPreset(20,120)" data-p="con">20/120</button>
    </div>
    <span class="sep"></span>
    <div class="toggle-row">
      <label><input type="checkbox" id="chkFilter" checked onchange="runBacktest()"><b>Trend Filter</b> (price > MA200)</label>
      <label><input type="checkbox" id="chkCommission" checked onchange="runBacktest()">Comm</label>
      <label><input type="checkbox" id="chkStamp" checked onchange="runBacktest()">Stamp</label>
      <label><input type="checkbox" id="chkSlippage" checked onchange="runBacktest()">Slip</label>
    </div>
    <button class="btn-go" onclick="runBacktest()">Run</button>
  </div>
</div>

<div class="container">
  <div class="metrics" id="metrics"></div>
  <div class="grid-2">
    <div class="card"><h3>Price + MA200 Filter Zone</h3><div class="chart-sm" id="pC"></div></div>
    <div class="card"><h3>MA Comparison (auto Y range)</h3><div class="chart-sm" id="maC"></div></div>
  </div>
  <div class="grid-2">
    <div class="card"><h3>Equity Curve</h3><div class="chart-sm" id="eC"></div></div>
    <div class="card"><h3>Drawdown</h3><div class="chart-sm" id="dC"></div></div>
  </div>
  <div class="card"><h3>Trade Log</h3><div style="overflow-x:auto"><table id="tradeTable"><tr><th colspan="4">Click Run</th></tr></table></div></div>
</div>

<div class="foot">MA200 Trend Filter (spec 7.2) | T+1 execution | Commission(0.03%)+Stamp(0.05% sell)+Slippage(0.1%) | Past != Future</div>

<script>
var DATA = {DATA_JSON};
var DEFAULT = "{default_id}";
var charts = {{}};
["pC","maC","eC","dC"].forEach(function(id){{charts[id]=echarts.init(document.getElementById(id))}});
window.addEventListener("resize",function(){{for(var k in charts)charts[k].resize()}});

function calcEMA(v,n){{var a=2/(n+1),r=[v[0]],e=v[0];for(var i=1;i<v.length;i++){{e=a*v[i]+(1-a)*e;r.push(e)}}return r}}
function calcSMA(v,n){{var r=[],s=0;for(var i=0;i<v.length;i++){{s+=v[i];if(i>=n)s-=v[i-n];r.push(i>=n-1?s/n:null)}}return r}}
function calcMA(v,n,t){{return t==="ema"?calcEMA(v,n):calcSMA(v,n)}}

function runBacktest(){{
  var sid=document.getElementById("stockSelect").value;
  var mt=document.getElementById("maType").value;
  var fp=parseInt(document.getElementById("fastP").value)||10;
  var sp=parseInt(document.getElementById("slowP").value)||60;
  if(fp>=sp){{alert("Fast<Slow!");return}}
  var useFilter=document.getElementById("chkFilter").checked;
  var useComm=document.getElementById("chkCommission").checked;
  var useStamp=document.getElementById("chkStamp").checked;
  var useSlip=document.getElementById("chkSlippage").checked;
  var COMM=useComm?0.0003:0,STAMP=useStamp?0.0005:0,SLIP=useSlip?0.001:0;

  var sd=DATA[sid],dates=sd.dates,price=sd.price,N=dates.length;
  var maF=calcMA(price,fp,mt),maS=calcMA(price,sp,mt),ma200=calcSMA(price,200);
  var w=Math.max(fp,sp); // MA200 warms up later; NaN = filter not available, accept signal

  var sigRaw=new Array(N).fill(0),crossRaw=new Array(N).fill(0);
  for(var i=w;i<N;i++){{
    if(maF[i]!==null&&maS[i]!==null){{
      sigRaw[i]=maF[i]>maS[i]?1:0;
      if(i>w)crossRaw[i]=sigRaw[i]-sigRaw[i-1];
    }}
  }}
  var nGoldenRaw=0;for(var i=0;i<N;i++)if(crossRaw[i]===1)nGoldenRaw++;

  var signal=new Array(N).fill(0),cross=new Array(N).fill(0),pos=new Array(N).fill(0);
  for(var i=w;i<N;i++){{
    if(maF[i]!==null&&maS[i]!==null){{
      var rawS=maF[i]>maS[i]?1:0;
      // Trend filter: only applies if MA200 is available AND filter is enabled
      var ma200OK = !useFilter || ma200[i]===null || price[i]>ma200[i] ? 1 : 0;
      signal[i]=rawS&ma200OK;
      if(i>w)cross[i]=signal[i]-signal[i-1];
      pos[i]=i>w?signal[i-1]:0;
    }}
  }}
  var nGolden=0;for(var i=0;i<N;i++)if(cross[i]===1)nGolden++;
  var filteredOut=nGoldenRaw-nGolden;

  var equity=new Array(N).fill(1),bench=new Array(N).fill(1);
  var dd=new Array(N).fill(0),bdd=new Array(N).fill(0);
  var peak=1,bpeak=1;
  for(var i=w+1;i<N;i++){{
    var pr=price[i]/price[i-1]-1;bench[i]=bench[i-1]*(1+pr);
    var act=Math.abs(pos[i]-pos[i-1]),cost=act*(COMM+SLIP);
    if(pos[i]<pos[i-1])cost+=STAMP;
    var sr=pos[i]*pr-cost;equity[i]=equity[i-1]*(1+sr);
    if(equity[i]>peak)peak=equity[i];dd[i]=(equity[i]-peak)/peak;
    if(bench[i]>bpeak)bpeak=bench[i];bdd[i]=(bench[i]-bpeak)/bpeak;
  }}

  var Nv=N-w-1;
  var tRet=equity[N-1]-1,bRet=bench[N-1]-1;
  var aRet=Nv>0?Math.pow(1+tRet,252/Nv)-1:0;
  var aBen=Nv>0?Math.pow(1+bRet,252/Nv)-1:0;
  var rets=[];for(var i=w+1;i<N;i++){{var pr=price[i]/price[i-1]-1,act=Math.abs(pos[i]-pos[i-1]),cost=act*(COMM+SLIP);if(pos[i]<pos[i-1])cost+=STAMP;rets.push(pos[i]*pr-cost)}}
  var mn=rets.reduce(function(a,b){{return a+b}},0)/rets.length;
  var vr=rets.reduce(function(a,b){{return a+(b-mn)*(b-mn)}},0)/rets.length;
  var sharpe=vr>0?Math.sqrt(252)*mn/Math.sqrt(vr):0;
  var maxDd=Math.min.apply(null,dd.slice(w));
  var calmar=maxDd!==0?aRet/Math.abs(maxDd):0;

  var trades=[];
  for(var i=w+1;i<N;i++){{if(cross[i]===1){{var exP=price[N-1];for(var j=i+1;j<N;j++){{if(cross[j]===1){{exP=price[j-1];break}}}}trades.push({{date:dates[i],entry:price[i],exit:exP,ret:exP/price[i]-1}})}}}}
  var wins=trades.filter(function(t){{return t.ret>0}});
  var wr=trades.length>0?wins.length/trades.length:0;

  document.getElementById("metrics").innerHTML=
    '<div class="mt"><div class="lbl">Total Return</div><div class="val '+(tRet>0?"up":"down")+'">'+(tRet*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Benchmark</div><div class="val '+(bRet>0?"up":"down")+'">'+(bRet*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Annual</div><div class="val neutral">'+(aRet*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Sharpe</div><div class="val neutral">'+sharpe.toFixed(2)+'</div></div>'+
    '<div class="mt"><div class="lbl">Max DD</div><div class="val down">'+(maxDd*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Calmar</div><div class="val neutral">'+calmar.toFixed(2)+'</div></div>'+
    '<div class="mt"><div class="lbl">Win Rate</div><div class="val neutral">'+(wr*100).toFixed(1)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Filtered</div><div class="val '+ (useFilter?'down':'neutral') +'">'+(useFilter?filteredOut+' sigs':'off')+'</div></div>';

  document.getElementById("tradeTable").innerHTML='<tr><th>Date</th><th>Entry</th><th>Result</th><th>Return</th></tr>'+
    trades.map(function(t){{var wl=t.ret>0?"win":"loss";return'<tr><td>'+t.date+'</td><td>'+t.entry.toFixed(2)+'</td><td><span class="badge '+wl+'">'+(t.ret>0?'Win':'Loss')+'</span></td><td class="'+(t.ret>0?"up":"down")+'">'+(t.ret*100).toFixed(2)+'%</td></tr>'}}).join('');

  var gScat=[],dScat=[],gRawScat=[];
  for(var i=w+1;i<N;i++){{
    if(cross[i]===1)gScat.push([dates[i],price[i]]);
    if(cross[i]===-1)dScat.push([dates[i],price[i]]);
    if(crossRaw[i]===1&&cross[i]===0)gRawScat.push([dates[i],price[i]]);
  }}

  var ft=mt.toUpperCase(),fl=ft+"("+fp+")",sl=ft+"("+sp+")";

  // MA Y range
  var maMin=Infinity,maMax=-Infinity;
  for(var i=w;i<N;i++){{if(maF[i]!==null){{if(maF[i]<maMin)maMin=maF[i];if(maF[i]>maMax)maMax=maF[i]}}if(maS[i]!==null){{if(maS[i]<maMin)maMin=maS[i];if(maS[i]>maMax)maMax=maS[i]}}}}
  var pad=(maMax-maMin)*0.08;maMin-=pad;maMax+=pad;

  // Chart 1: Price + MA200 + signals
  var pSeries=[
    {{name:"Close",type:"line",data:price,lineStyle:{{color:"#B4B2A9",width:1}},symbol:"none",z:1}},
    {{name:"MA200",type:"line",data:ma200,lineStyle:{{color:"#378ADD",width:1.5,type:"dotted"}},symbol:"none",z:2}},
    {{name:"Golden (valid)",type:"scatter",data:gScat,symbol:"triangle",symbolSize:12,itemStyle:{{color:"#1D9E75"}},z:4}},
    {{name:"Death",type:"scatter",data:dScat,symbol:"triangle",symbolRotate:180,symbolSize:10,itemStyle:{{color:"#D85A30"}},z:4}}
  ];
  if(useFilter&&gRawScat.length>0){{
    pSeries.push({{name:"Golden (filtered)",type:"scatter",data:gRawScat,symbol:"circle",symbolSize:8,itemStyle:{{color:"#D85A30",opacity:0.5}},z:3}});
  }}

  charts.pC.setOption({{
    tooltip:{{trigger:"axis"}},
    legend:{{data:pSeries.map(function(x){{return x.name}}),top:2}},
    grid:{{left:60,right:20,top:38,bottom:50}},
    xAxis:{{type:"category",data:dates,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},
    yAxis:{{type:"value",axisLabel:{{formatter:function(v){{return v.toFixed(0)}}}}}},
    dataZoom:[{{type:"inside"}},{{type:"slider",bottom:4,height:22}}],
    series:pSeries
  }});

  // Chart 2: MA comparison
  charts.maC.setOption({{
    tooltip:{{trigger:"axis"}},
    legend:{{data:[fl,sl,"Golden","Death"],top:2}},
    grid:{{left:60,right:20,top:38,bottom:50}},
    xAxis:{{type:"category",data:dates,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},
    yAxis:{{type:"value",min:Math.floor(maMin),max:Math.ceil(maMax),
      axisLabel:{{formatter:function(v){{return v.toFixed(1)}}}}}},
    dataZoom:[{{type:"inside"}},{{type:"slider",bottom:4,height:22}}],
    series:[
      {{name:fl,type:"line",data:maF,lineStyle:{{color:"#D85A30",width:1.8}},symbol:"none",z:2}},
      {{name:sl,type:"line",data:maS,lineStyle:{{color:"#534AB7",width:1.8}},symbol:"none",z:2}},
      {{name:"Golden",type:"scatter",data:gScat,symbol:"triangle",symbolSize:10,itemStyle:{{color:"#1D9E75"}},z:3}},
      {{name:"Death",type:"scatter",data:dScat,symbol:"triangle",symbolRotate:180,symbolSize:10,itemStyle:{{color:"#D85A30"}},z:3}}
    ]
  }});

  // Chart 3: Equity
  charts.eC.setOption({{
    tooltip:{{trigger:"axis",formatter:function(p){{return p[0].axisValue+"<br/>"+p.map(function(x){{return x.marker+x.seriesName+": "+(x.value*100).toFixed(2)+"%"}}).join("<br/>")}}}},
    legend:{{data:["Strategy","Buy & Hold"],top:2}},
    grid:{{left:60,right:20,top:38,bottom:50}},
    xAxis:{{type:"category",data:dates,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},
    yAxis:{{type:"value",axisLabel:{{formatter:function(v){{return(v*100).toFixed(0)+"%"}}}}}},
    dataZoom:[{{type:"inside"}},{{type:"slider",bottom:4,height:22}}],
    series:[
      {{name:"Strategy",type:"line",data:equity,lineStyle:{{color:"#378ADD",width:2}},areaStyle:{{color:"rgba(55,138,221,0.06)"}},symbol:"none",z:2}},
      {{name:"Buy & Hold",type:"line",data:bench,lineStyle:{{color:"#B4B2A9",width:1.2,type:"dashed"}},symbol:"none",z:1}}
    ]
  }});

  // Chart 4: Drawdown
  charts.dC.setOption({{
    tooltip:{{trigger:"axis",formatter:function(p){{return p[0].axisValue+"<br/>"+p.map(function(x){{return x.marker+x.seriesName+": "+(x.value*100).toFixed(2)+"%"}}).join("<br/>")}}}},
    legend:{{data:["Strategy DD","Benchmark DD"],top:2}},
    grid:{{left:60,right:20,top:38,bottom:50}},
    xAxis:{{type:"category",data:dates,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},
    yAxis:{{type:"value",axisLabel:{{formatter:function(v){{return(v*100).toFixed(0)+"%"}}}},max:0}},
    dataZoom:[{{type:"inside"}},{{type:"slider",bottom:4,height:22}}],
    series:[
      {{name:"Strategy DD",type:"line",data:dd,lineStyle:{{color:"#D85A30",width:1.5}},areaStyle:{{color:"rgba(216,90,48,0.1)"}},symbol:"none",z:2}},
      {{name:"Benchmark DD",type:"line",data:bdd,lineStyle:{{color:"#B4B2A9",width:1,type:"dashed"}},symbol:"none",z:1}},
      {{name:"Halt",type:"line",markLine:{{silent:true,symbol:"none",lineStyle:{{color:"#D85A30",type:"dotted",width:1}},data:[{{yAxis:-0.25,label:{{formatter:"-25%",color:"#D85A30"}}}}]}},z:0}}
    ]
  }});

  var sub=document.querySelector(".sub");
  sub.innerHTML=(useFilter?'<span class="filter-indicator filter-on">TREND FILTER ON</span>':'<span class="filter-indicator filter-off">FILTER OFF</span>')+
    ' | '+sd.label+' | '+ft+'('+fp+')/'+ft+'('+sp+') | MA200 filter | '+nGolden+' trades ('+filteredOut+' filtered out)';
}}

function validateParams(){{var f=parseInt(document.getElementById("fastP").value)||10;var s=parseInt(document.getElementById("slowP").value)||60;if(f>=s)document.getElementById("slowP").value=f+1}}
function setPreset(f,s){{document.getElementById("fastP").value=f;document.getElementById("slowP").value=s;var bs=document.querySelectorAll("#pGroup button");bs.forEach(function(b){{b.classList.remove("active")}});var n=f===5?"agg":f===10?"bal":"con";var t=document.querySelector('[data-p="'+n+'"]');if(t)t.classList.add("active");runBacktest()}}

runBacktest();
</script>
</body>
</html>'''

html_path = os.path.join(OUT, 'ma_crossover_trend_dashboard.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Dashboard: {html_path}')
print(f'Size: {os.path.getsize(html_path)/1024:.0f} KB')
print('All done!')
