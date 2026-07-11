import os, json
import pandas as pd, numpy as np
import nbformat as nbf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(PROJ, 'TASK3')
DATA = os.path.join(PROJ, 'data')
os.makedirs(OUT, exist_ok=True)

# Scan data dirs using stock codes only (avoids encoding issues)
stock_map = {}
for dname in os.listdir(DATA):
    dpath = os.path.join(DATA, dname)
    if not os.path.isdir(dpath):
        continue
    csv_path = os.path.join(dpath, 'daily_adjusted.csv')
    if not os.path.exists(csv_path):
        continue
    parts = dname.split('_')
    code = parts[0] if parts else 'unknown'
    stock_map[code] = (dpath, csv_path)

print('Found stocks:', list(stock_map.keys()))

# Use the stock the user requested: 603986
TARGET = '603986.SH'
if TARGET not in stock_map:
    TARGET = list(stock_map.keys())[0]

stock_dir, data_file = stock_map[TARGET]
print(f'Using: {TARGET} -> {data_file}')

# Build notebook
nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13.0'}
}
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def cd(s): cells.append(nbf.v4.new_code_cell(s))

data_file_fwd = data_file.replace('\\', '/')
stock_entries = []
for code, (d, csv) in stock_map.items():
    stock_entries.append(f'    "{code}": "{csv.replace(chr(92), "/")}",')
stock_dict = '\n'.join(stock_entries)

md(f'''# MA Crossover Strategy Backtest

> Strategy ID: MA_CROSSOVER_V1 | Based on: ma_crossover_strategy_spec.md
> Type: Trend Following | Market: A-Shares (Long Only)
> Target: {TARGET}

## Pipeline

1. Load Data -> 2. Calculate MA -> 3. Generate Signals -> 4. Run Backtest -> 5. Metrics -> 6. Visualize

## Switch Stocks

Change the CHOSEN variable in Cell 2. Available: {", ".join(stock_map.keys())}''')

cd(f'''# ============================================================
# 1. Parameters (per spec section 2)
# ============================================================
FAST_PERIOD   = 10
SLOW_PERIOD   = 60
MA_TYPE       = "ema"
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE  = 0.0005
SLIPPAGE_RATE   = 0.001
print(f"Strategy: {{MA_TYPE.upper()}}({{FAST_PERIOD}}) / {{MA_TYPE.upper()}}({{SLOW_PERIOD}})")
print(f"Costs: comm={{COMMISSION_RATE:.3%}} stamp={{STAMP_TAX_RATE:.3%}}(sell) slip={{SLIPPAGE_RATE:.1%}}")
print(f"Capital: {{INITIAL_CAPITAL:,}}")''')

cd(f'''# ============================================================
# 2. Load Data
# ============================================================
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

STOCKS = {{
{stock_dict}
}}

# Change this to switch stocks (e.g. "002594.SZ" or "688981.SH")
CHOSEN = "{TARGET}"

df = pd.read_csv(STOCKS[CHOSEN], encoding="utf-8-sig", parse_dates=["trade_date"])
df = df.sort_values("trade_date").reset_index(drop=True)
df["price"] = df["close_qfq"]
print(f"Stock: {{CHOSEN}}")
print(f"Rows: {{len(df)}}, {{df.trade_date.min().date()}} ~ {{df.trade_date.max().date()}}")
print(f"Price range: {{df.price.min():.2f}} ~ {{df.price.max():.2f}}")
df[["trade_date","price","vol","amount"]].head()''')

cd('''# ============================================================
# 3. Moving Averages (per spec section 2)
# ============================================================
def calc_ma(s, n, t):
    return s.ewm(span=n, adjust=False).mean() if t == "ema" else s.rolling(n).mean()

df["ma_fast"] = calc_ma(df["price"], FAST_PERIOD, MA_TYPE)
df["ma_slow"] = calc_ma(df["price"], SLOW_PERIOD, MA_TYPE)
warmup = max(FAST_PERIOD, SLOW_PERIOD)
df_v = df.iloc[warmup:].copy().reset_index(drop=True)
fast_lbl = f"{MA_TYPE.upper()}({FAST_PERIOD})"
slow_lbl = f"{MA_TYPE.upper()}({SLOW_PERIOD})"
print(f"Warmup bars dropped: {warmup}")
print(f"Valid rows: {len(df_v)}")
df_v[["trade_date","price","ma_fast","ma_slow"]].head(10).round(2)''')

cd('''# ============================================================
# 4. Signal Generation
# ============================================================
df_v["signal"]   = (df_v["ma_fast"] > df_v["ma_slow"]).astype(int)
df_v["cross"]    = df_v["signal"].diff()
df_v["position"] = df_v["signal"].shift(1).fillna(0).astype(int)
n_g = int((df_v["cross"] == 1).sum())
n_d = int((df_v["cross"] == -1).sum())
print(f"Golden Cross: {n_g} | Death Cross: {n_d} | Pairs: {n_g}")
changes = df_v[df_v["cross"] != 0].copy()
changes["label"] = changes["cross"].map({1: "GOLDEN BUY", -1: "DEATH SELL"})
changes[["trade_date","price","label"]].head(10)''')

cd('''# ============================================================
# 5. Backtest (with all costs)
# ============================================================
df_v["price_ret"]  = df_v["price"].pct_change()
df_v["gross_ret"]  = df_v["position"] * df_v["price_ret"]
df_v["trade_act"]  = df_v["position"].diff().abs()
df_v["cost"]       = df_v["trade_act"] * (COMMISSION_RATE + SLIPPAGE_RATE)
df_v.loc[df_v["position"].diff() < 0, "cost"] += STAMP_TAX_RATE
df_v["strat_ret"]  = df_v["gross_ret"] - df_v["cost"]
df_v["equity"]     = (1 + df_v["strat_ret"].fillna(0)).cumprod()
df_v["bench_eq"]   = (1 + df_v["price_ret"].fillna(0)).cumprod()
pk_s = df_v["equity"].cummax(); pk_b = df_v["bench_eq"].cummax()
df_v["dd"]       = (df_v["equity"] - pk_s) / pk_s
df_v["bench_dd"] = (df_v["bench_eq"] - pk_b) / pk_b
print(f"Strategy equity: {df_v.equity.iloc[-1]:.4f} ({df_v.equity.iloc[-1]-1:+.2%})")
print(f"Benchmark eq:    {df_v.bench_eq.iloc[-1]:.4f} ({df_v.bench_eq.iloc[-1]-1:+.2%})")''')

cd('''# ============================================================
# 6. Performance Metrics
# ============================================================
ret = df_v["strat_ret"].dropna()
N = len(ret)
total_ret = df_v["equity"].iloc[-1] - 1
bench_ret = df_v["bench_eq"].iloc[-1] - 1
ann_ret   = (1+total_ret)**(252/N)-1
ann_bench = (1+bench_ret)**(252/N)-1
rc = ret[ret!=0] if len(ret[ret!=0])>0 else ret
sharpe = np.sqrt(252)*rc.mean()/rc.std() if rc.std()>0 else 0
max_dd = df_v["dd"].min()
max_dd_b = df_v["bench_dd"].min()
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
<tr style="background:#f8f9fa;font-weight:bold"><td style="padding:10px 16px">Metric</td><td style="padding:10px 16px;text-align:right">Strategy</td><td style="padding:10px 16px;text-align:right">Benchmark</td></tr>
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
ax.plot(dts,df_v["price"],"#B4B2A9",lw=0.8,alpha=0.7,label="Close")
ax.plot(dts,df_v["ma_fast"],"#D85A30",lw=1.2,label=fast_lbl)
ax.plot(dts,df_v["ma_slow"],"#534AB7",lw=1.2,label=slow_lbl)
g=df_v[df_v["cross"]==1]; d=df_v[df_v["cross"]==-1]
ax.scatter(g["trade_date"],g["price"],c="#1D9E75",marker="^",s=80,zorder=5,label="Golden")
ax.scatter(d["trade_date"],d["price"],c="#D85A30",marker="v",s=80,zorder=5,label="Death")
ax.set_title(f"{CHOSEN} - Price & MA & Signals",fontsize=14,fontweight="bold")
ax.legend(loc="upper left",fontsize=9); ax.set_ylabel("Price"); ax.grid(True,alpha=0.3)

ax=axes[1]
ax.fill_between(dts,1,df_v["equity"],alpha=0.1,color="#378ADD")
ax.plot(dts,df_v["equity"],"#378ADD",lw=1.8,label="Strategy")
ax.plot(dts,df_v["bench_eq"],"#B4B2A9",lw=1.2,ls="dashed",label="Benchmark")
ax.axhline(1,c="black",lw=0.5,ls="dotted")
ax.set_title("Equity Curve",fontsize=14,fontweight="bold")
ax.legend(loc="upper left",fontsize=9); ax.set_ylabel("Equity"); ax.grid(True,alpha=0.3)

ax=axes[2]
ax.fill_between(dts,df_v["dd"],0,alpha=0.3,color="#D85A30")
ax.fill_between(dts,df_v["bench_dd"],0,alpha=0.15,color="#B4B2A9")
ax.plot(dts,df_v["dd"],"#D85A30",lw=1,label="Strategy DD")
ax.plot(dts,df_v["bench_dd"],"#B4B2A9",lw=0.8,ls="dashed",label="Benchmark DD")
ax.axhline(0,c="black",lw=0.3); ax.axhline(-0.25,c="#D85A30",lw=0.5,ls="dotted",alpha=0.5)
ax.set_title("Drawdown",fontsize=14,fontweight="bold")
ax.legend(loc="lower left",fontsize=9); ax.set_ylabel("Drawdown"); ax.grid(True,alpha=0.3)
for a in axes:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(a.xaxis.get_majorticklabels(),rotation=45,ha="right",fontsize=8)
plt.tight_layout()
plt.show()''')

cd(f'''# ============================================================
# 8. Summary
# ============================================================
print(f"""
{{'='*60}}
  MA Crossover Backtest: {{CHOSEN}}
  {{MA_TYPE.upper()}}({{FAST_PERIOD}}) / {{MA_TYPE.upper()}}({{SLOW_PERIOD}})
  Return: {{total_ret:+.2%}} | Sharpe: {{sharpe:.2f}} | MaxDD: {{max_dd:+.2%}}
  WinRate: {{wr:.1%}} | Trades: {{len(trades_ret)}} | P/L Ratio: {{plr:.2f}}
{{'='*60}}

To switch stocks: change CHOSEN in Cell 2
To tune params: FAST_PERIOD / SLOW_PERIOD / MA_TYPE
Based on ma_crossover_strategy_spec.md
""")''')

nb.cells = cells
nb_path = os.path.join(OUT, 'ma_crossover_backtest.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook: {nb_path}')
print(f'Cells: {len(cells)}')

# Also generate dashboard data
df = pd.read_csv(data_file, encoding='utf-8-sig', parse_dates=['trade_date'])
df = df.sort_values('trade_date').reset_index(drop=True)
df['price'] = df['close_qfq']
FAST,SLOW,MT=10,60,'ema'

def cma(s,n,t):
    return s.ewm(span=n,adjust=False).mean() if t=='ema' else s.rolling(n).mean()

df['ma_fast']=cma(df['price'],FAST,MT)
df['ma_slow']=cma(df['price'],SLOW,MT)
df_v=df.iloc[max(FAST,SLOW):].copy().reset_index(drop=True)
df_v['signal']=(df_v['ma_fast']>df_v['ma_slow']).astype(int)
df_v['cross']=df_v['signal'].diff()
df_v['position']=df_v['signal'].shift(1).fillna(0).astype(int)
df_v['price_ret']=df_v['price'].pct_change()
df_v['gross_ret']=df_v['position']*df_v['price_ret']
df_v['trade_act']=df_v['position'].diff().abs()
df_v['cost']=df_v['trade_act']*(0.0003+0.001)
df_v.loc[df_v['position'].diff()<0,'cost']+=0.0005
df_v['strat_ret']=df_v['gross_ret']-df_v['cost']
df_v['equity']=(1+df_v['strat_ret'].fillna(0)).cumprod()
df_v['bench_eq']=(1+df_v['price_ret'].fillna(0)).cumprod()
pk_s=df_v['equity'].cummax();pk_b=df_v['bench_eq'].cummax()
df_v['dd']=(df_v['equity']-pk_s)/pk_s
df_v['bench_dd']=(df_v['bench_eq']-pk_b)/pk_b

ret=df_v['strat_ret'].dropna();N=len(ret)
total_ret=df_v['equity'].iloc[-1]-1;bench_ret=df_v['bench_eq'].iloc[-1]-1
ann_ret=(1+total_ret)**(252/N)-1;ann_bench=(1+bench_ret)**(252/N)-1
rc=ret[ret!=0] if len(ret[ret!=0])>0 else ret
sharpe=np.sqrt(252)*rc.mean()/rc.std() if rc.std()>0 else 0
max_dd=df_v['dd'].min();max_dd_b=df_v['bench_dd'].min()
calmar=ann_ret/abs(max_dd) if max_dd!=0 else 0

g_entry=df_v[df_v['cross']==1]
trades_ret=[]
for i in range(len(g_entry)):
    ei=df_v[df_v['trade_date']==g_entry['trade_date'].iloc[i]].index[0]
    xi=len(df_v)-1 if i+1>=len(g_entry) else df_v[df_v['trade_date']==g_entry['trade_date'].iloc[i+1]].index[0]-1
    trades_ret.append(df_v.loc[xi,'price']/df_v.loc[ei,'price']-1)
wins=[r for r in trades_ret if r>0];loses=[r for r in trades_ret if r<=0]
wr=len(wins)/len(trades_ret) if trades_ret else 0

dates_j=df_v['trade_date'].dt.strftime('%Y-%m-%d').tolist()
price_j=df_v['price'].round(2).tolist()
maf_j=df_v['ma_fast'].round(2).tolist()
mas_j=df_v['ma_slow'].round(2).tolist()
eq_j=df_v['equity'].round(6).tolist()
beq_j=df_v['bench_eq'].round(6).tolist()
dd_j=df_v['dd'].round(6).tolist()
bdd_j=df_v['bench_dd'].round(6).tolist()
gm=df_v['cross']==1;dm=df_v['cross']==-1
gd=[dates_j[i] for i in df_v[gm].index];gp=[price_j[i] for i in df_v[gm].index]
ddt_=[dates_j[i] for i in df_v[dm].index];dp_=[price_j[i] for i in df_v[dm].index]

trows=''
for dt,p,r in zip([str(x.date()) for x in g_entry['trade_date']],[round(x,2) for x in g_entry['price']],trades_ret):
    css_cls = '"up"' if r>0 else '"down"'
    b_cls = '"win"' if r>0 else '"loss"'
    lbl = '"Win"' if r>0 else '"Loss"'
    trows += f'<tr><td>{dt}</td><td>{p:.2f}</td><td><span class="badge {b_cls}">{lbl}</span></td><td class={css_cls}>{r:+.2%}</td></tr>'

TOT_up = 'up' if total_ret>0 else 'down'
BEN_up = 'up' if bench_ret>0 else 'down'

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MA Crossover Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root{{--bg:#fff;--bg2:#f8f9fa;--t:#1a1a2e;--t2:#555;--bd:#e0e0e0;--gn:#1D9E75;--rd:#D85A30;--pu:#534AB7;--bl:#378ADD;--ra:10px}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:var(--t);background:#f5f5f5;line-height:1.7}}
.hd{{background:#fff;border-bottom:1px solid var(--bd);padding:16px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}}
.hd h1{{font-size:20px;font-weight:600}}.hd .sub{{font-size:12px;color:#888}}
.container{{max-width:1200px;margin:0 auto;padding:18px 22px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(124px,1fr));gap:8px;margin-bottom:18px}}
.mt{{background:#fff;border:1px solid var(--bd);border-radius:var(--ra);padding:12px;text-align:center}}
.mt .lbl{{font-size:11px;color:#888;margin-bottom:2px}}.mt .val{{font-size:18px;font-weight:600}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:var(--ra);padding:14px;margin-bottom:14px}}
.card h3{{font-size:14px;font-weight:600;margin-bottom:10px}}
.chart{{width:100%;height:380px}}
.up{{color:var(--rd)}}.down{{color:var(--gn)}}.neutral{{color:var(--t)}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:500}}
.badge.win{{background:#E1F5EE;color:#0F6E56}}.badge.loss{{background:#FAECE7;color:#D85A30}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:var(--bg2);text-align:left;padding:8px 12px;font-weight:600;border-bottom:2px solid var(--bd)}}
td{{padding:6px 12px;border-bottom:1px solid var(--bd)}}tr:hover td{{background:#fafafa}}
.foot{{padding:10px 14px;font-size:12px;color:#888}}
</style></head><body>
<div class="hd">
<div><h1>MA Crossover Backtest Dashboard</h1><p class="sub">{TARGET} | EMA(10)/EMA(60) | {dates_j[0]} ~ {dates_j[-1]} | {len(df_v)} trading days</p></div>
</div>
<div class="container">
<div class="metrics">
<div class="mt"><div class="lbl">Total Return</div><div class="val {TOT_up}">{total_ret:+.2%}</div></div>
<div class="mt"><div class="lbl">Benchmark</div><div class="val {BEN_up}">{bench_ret:+.2%}</div></div>
<div class="mt"><div class="lbl">Annual Return</div><div class="val neutral">{ann_ret:+.2%}</div></div>
<div class="mt"><div class="lbl">Sharpe</div><div class="val neutral">{sharpe:.2f}</div></div>
<div class="mt"><div class="lbl">Max Drawdown</div><div class="val down">{max_dd:+.2%}</div></div>
<div class="mt"><div class="lbl">Calmar</div><div class="val neutral">{calmar:.2f}</div></div>
<div class="mt"><div class="lbl">Win Rate</div><div class="val neutral">{wr:.1%}</div></div>
<div class="mt"><div class="lbl">Trades</div><div class="val neutral">{len(trades_ret)}</div></div>
</div>
<div class="card"><h3>Price &amp; MA &amp; Signals</h3><div class="chart" id="pC"></div></div>
<div class="card"><h3>Equity Curve</h3><div class="chart" id="eC"></div></div>
<div class="card"><h3>Drawdown</h3><div class="chart" id="dC"></div></div>
<div class="card"><h3>Trade Log</h3><div style="overflow-x:auto"><table>
<tr><th>Date</th><th>Entry Price</th><th>Result</th><th>Return</th></tr>{trows}</table></div></div>
</div>
<script>
var D={json.dumps(dates_j)},P={json.dumps(price_j)},MF={json.dumps(maf_j)},MS={json.dumps(mas_j)},
 EQ={json.dumps(eq_j)},BQ={json.dumps(beq_j)},DD={json.dumps(dd_j)},BD={json.dumps(bdd_j)},
 GD={json.dumps(gd)},GP={json.dumps(gp)},DT={json.dumps(ddt_)},DP={json.dumps(dp_)};
function mk(i,o){{var c=echarts.init(document.getElementById(i));c.setOption(o);return c}}
var pc=mk("pC",{{tooltip:{{trigger:"axis"}},legend:{{data:["Close","EMA10","EMA60","Golden","Death"],bottom:0}},grid:{{left:70,right:30,top:14,bottom:36}},xAxis:{{type:"category",data:D,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},yAxis:{{type:"value"}},dataZoom:[{{type:"inside"}},{{type:"slider",bottom:2}}],series:[{{name:"Close",type:"line",data:P,lineStyle:{{color:"#B4B2A9",width:1}},symbol:"none",z:1}},{{name:"EMA10",type:"line",data:MF,lineStyle:{{color:"#D85A30",width:1.5}},symbol:"none",z:2}},{{name:"EMA60",type:"line",data:MS,lineStyle:{{color:"#534AB7",width:1.5}},symbol:"none",z:2}},{{name:"Golden",type:"scatter",data:GD.map(function(d,i){{return[d,GP[i]]}}),symbol:"triangle",symbolSize:13,itemStyle:{{color:"#1D9E75"}},z:3}},{{name:"Death",type:"scatter",data:DT.map(function(d,i){{return[d,DP[i]]}}),symbol:"triangle",symbolRotate:180,symbolSize:13,itemStyle:{{color:"#D85A30"}},z:3}}]}});
var ec=mk("eC",{{tooltip:{{trigger:"axis",formatter:function(p){{return p[0].axisValue+"<br/>"+p.map(function(x){{return x.marker+x.seriesName+": "+(x.value*100).toFixed(2)+"%"}}).join("<br/>")}}}},legend:{{data:["Strategy","Buy & Hold"],bottom:0}},grid:{{left:70,right:30,top:14,bottom:36}},xAxis:{{type:"category",data:D,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},yAxis:{{type:"value",axisLabel:{{formatter:function(v){{return(v*100).toFixed(0)+"%"}}}}}},dataZoom:[{{type:"inside"}},{{type:"slider",bottom:2}}],series:[{{name:"Strategy",type:"line",data:EQ,lineStyle:{{color:"#378ADD",width:2}},areaStyle:{{color:"rgba(55,138,221,0.06)"}},symbol:"none",z:2}},{{name:"Buy & Hold",type:"line",data:BQ,lineStyle:{{color:"#B4B2A9",width:1.2,type:"dashed"}},symbol:"none",z:1}}]}});
var dc=mk("dC",{{tooltip:{{trigger:"axis",formatter:function(p){{return p[0].axisValue+"<br/>"+p.map(function(x){{return x.marker+x.seriesName+": "+(x.value*100).toFixed(2)+"%"}}).join("<br/>")}}}},legend:{{data:["Strategy DD","Benchmark DD"],bottom:0}},grid:{{left:70,right:30,top:14,bottom:36}},xAxis:{{type:"category",data:D,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},yAxis:{{type:"value",axisLabel:{{formatter:function(v){{return(v*100).toFixed(0)+"%"}}}},max:0}},dataZoom:[{{type:"inside"}},{{type:"slider",bottom:2}}],series:[{{name:"Strategy DD",type:"line",data:DD,lineStyle:{{color:"#D85A30",width:1.5}},areaStyle:{{color:"rgba(216,90,48,0.1)"}},symbol:"none",z:2}},{{name:"Benchmark DD",type:"line",data:BD,lineStyle:{{color:"#B4B2A9",width:1,type:"dashed"}},symbol:"none",z:1}},{{name:"-25% Halt",type:"line",markLine:{{silent:true,symbol:"none",lineStyle:{{color:"#D85A30",type:"dotted",width:1}},data:[{{yAxis:-0.25,label:{{formatter:"-25% Halt",color:"#D85A30"}}}}]}},z:0}}]}});
[pc,ec,dc].forEach(function(c){{window.addEventListener("resize",function(){{c.resize()}})}})
</script>
<div class="foot">Commission(0.03%)+StampTax(0.05%sell)+Slippage(0.1%) | T+1 Execution | Past != Future | Based on ma_crossover_strategy_spec.md</div>
</body></html>'''

dash_path = os.path.join(OUT, 'ma_crossover_dashboard.html')
with open(dash_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Dashboard: {dash_path}')
print('All done!')
