"""
直接生成均线交叉策略 Notebook 和 HTML 看板
避免 f-string 嵌套问题
"""

import json, os, sys

PROJECT = "C:/Users/RYAN/Desktop/BA工作坊"
OUTPUT = os.path.join(PROJECT, "TASK3")
DATA   = os.path.join(PROJECT, "data", "603986.SH_兆易创新", "daily_adjusted.csv")

# ============================================================
# Cell 内容定义（纯字符串，不含 f-string）
# ============================================================

CELL_INTRO = """\
# 双均线交叉趋势跟踪策略 — 完整回测

> **策略 ID**: MA_CROSSOVER_V1 | **依据**: `ma_crossover_strategy_spec.md`  
> **类型**: 趋势跟踪 | **适用市场**: A 股（单向做多）

---

## 策略核心逻辑

```
价格 = 趋势 + 噪声
       |
       v
 均线取平均值 → 消除随机波动 → 暴露底层趋势
       |
       v
 快线穿越慢线 → 短期共识偏离中期成本 → 趋势转向信号
       |
       v
 金叉买入 / 死叉卖出 → 捕捉趋势惯性利润
```

## 本 Notebook 流程

1. 加载数据 → 2. 计算均线 → 3. 生成信号 → 4. 执行回测 → 5. 绩效评估 → 6. 可视化 → 7. 导出看板"""

CELL_PARAMS = """\
# ============================================================
# 1. 参数配置 (按 spec S2 定义)
# ============================================================
FAST_PERIOD   = 10        # 快线周期 (3~30, 默认10)
SLOW_PERIOD   = 60        # 慢线周期 (20~250, 默认60)
MA_TYPE       = "ema"     # 均线类型: "sma" / "ema"

INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0003   # 佣金万三
STAMP_TAX_RATE  = 0.0005   # 印花税万五(卖出)
SLIPPAGE_RATE   = 0.001    # 滑点千一

print(f"策略: {MA_TYPE.upper()}({FAST_PERIOD}) / {MA_TYPE.upper()}({SLOW_PERIOD})")
print(f"成本: 佣金{COMMISSION_RATE:.3%} + 印花税{STAMP_TAX_RATE:.3%}(卖) + 滑点{SLIPPAGE_RATE:.1%}")
print(f"初始资金: {INITIAL_CAPITAL:,.0f}")"""

CELL_DATA = """\
# ============================================================
# 2. 数据加载 — 支持三只标的切换
# ============================================================
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

PROJECT = "C:/Users/RYAN/Desktop/BA工作坊"
DATA_DIR = os.path.join(PROJECT, "data")

STOCKS = {
    "兆易创新 (603986.SH)": os.path.join(DATA_DIR, "603986.SH_兆易创新", "daily_adjusted.csv"),
    "比亚迪 (002594.SZ)":   os.path.join(DATA_DIR, "002594.SZ_比亚迪",   "daily_adjusted.csv"),
    "中芯国际 (688981.SH)": os.path.join(DATA_DIR, "688981.SH_中芯国际", "daily_adjusted.csv"),
}

# ★ 修改这里切换股票 ★
CHOSEN = "兆易创新 (603986.SH)"

df = pd.read_csv(STOCKS[CHOSEN], encoding="utf-8-sig", parse_dates=["trade_date"])
df = df.sort_values("trade_date").reset_index(drop=True)
df["price"] = df["close_qfq"]  # 前复权价格

print(f"股票: {CHOSEN}")
print(f"数据: {len(df)} 条, {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
print(f"价格: {df['price'].min():.2f} ~ {df['price'].max():.2f}")
df[["trade_date", "price", "vol", "amount"]].head()"""

CELL_MA = """\
# ============================================================
# 3. 均线计算 (按 spec S2)
# ============================================================
def calc_ma(s, n, t):
    return s.ewm(span=n, adjust=False).mean() if t == "ema" else s.rolling(n).mean()

df["ma_fast"] = calc_ma(df["price"], FAST_PERIOD, MA_TYPE)
df["ma_slow"] = calc_ma(df["price"], SLOW_PERIOD, MA_TYPE)

warmup = max(FAST_PERIOD, SLOW_PERIOD)
df_v = df.iloc[warmup:].copy().reset_index(drop=True)

fast_lbl = f"{MA_TYPE.upper()}({FAST_PERIOD})"
slow_lbl = f"{MA_TYPE.upper()}({SLOW_PERIOD})"
print(f"预热: {warmup} 根K线, 有效数据: {len(df_v)} 条")
df_v[["trade_date", "price", "ma_fast", "ma_slow"]].head(10).round(2)"""

CELL_SIGNAL = """\
# ============================================================
# 4. 信号生成 (按 spec S3 — 金叉/死叉 + T+1执行)
# ============================================================
df_v["signal"] = (df_v["ma_fast"] > df_v["ma_slow"]).astype(int)
df_v["cross"]  = df_v["signal"].diff()
df_v["position"] = df_v["signal"].shift(1).fillna(0).astype(int)

n_golden = int((df_v["cross"] == 1).sum())
n_death  = int((df_v["cross"] == -1).sum())
print(f"金叉: {n_golden} 次 | 死叉: {n_death} 次 | 交易: {n_golden} 次")

changes = df_v[df_v["cross"] != 0].copy()
changes["label"] = changes["cross"].map({1: "金叉买入", -1: "死叉卖出"})
changes[["trade_date", "price", "label"]].head(10)"""

CELL_BACKTEST = """\
# ============================================================
# 5. 执行回测 (按 spec S4 — 含全部交易成本)
# ============================================================
df_v["price_ret"] = df_v["price"].pct_change()
df_v["strat_ret_gross"] = df_v["position"] * df_v["price_ret"]

df_v["trade"] = df_v["position"].diff().abs()
df_v["cost"]  = df_v["trade"] * (COMMISSION_RATE + SLIPPAGE_RATE)
df_v.loc[df_v["position"].diff() < 0, "cost"] += STAMP_TAX_RATE

df_v["strat_ret"] = df_v["strat_ret_gross"] - df_v["cost"]

df_v["equity"]       = (1 + df_v["strat_ret"].fillna(0)).cumprod()
df_v["bench_equity"] = (1 + df_v["price_ret"].fillna(0)).cumprod()

pk_s = df_v["equity"].cummax()
pk_b = df_v["bench_equity"].cummax()
df_v["dd"]       = (df_v["equity"] - pk_s) / pk_s
df_v["bench_dd"] = (df_v["bench_equity"] - pk_b) / pk_b

print(f"策略净值: {df_v['equity'].iloc[-1]:.4f} ({df_v['equity'].iloc[-1]-1:+.2%})")
print(f"基准净值: {df_v['bench_equity'].iloc[-1]:.4f} ({df_v['bench_equity'].iloc[-1]-1:+.2%})")"""

CELL_METRICS = """\
# ============================================================
# 6. 绩效指标 (按 spec S8.4)
# ============================================================
ret = df_v["strat_ret"].dropna()
br  = df_v["price_ret"].dropna()
N   = len(ret)

total_ret  = df_v["equity"].iloc[-1] - 1
bench_ret  = df_v["bench_equity"].iloc[-1] - 1
ann_ret    = (1 + total_ret) ** (252 / N) - 1
ann_bench  = (1 + bench_ret) ** (252 / N) - 1

rc = ret[ret != 0] if len(ret[ret != 0]) > 0 else ret
sharpe = np.sqrt(252) * rc.mean() / rc.std() if rc.std() > 0 else 0

max_dd   = df_v["dd"].min()
max_dd_b = df_v["bench_dd"].min()
calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0

# 逐笔交易
g_entry = df_v[df_v["cross"] == 1]
trades_ret = []
for i in range(len(g_entry)):
    e_idx = df_v[df_v["trade_date"] == g_entry["trade_date"].iloc[i]].index[0]
    if i + 1 < len(g_entry):
        x_idx = df_v[df_v["trade_date"] == g_entry["trade_date"].iloc[i+1]].index[0] - 1
    else:
        x_idx = len(df_v) - 1
    trades_ret.append(df_v.loc[x_idx, "price"] / df_v.loc[e_idx, "price"] - 1)

wins  = [r for r in trades_ret if r > 0]
loses = [r for r in trades_ret if r <= 0]
wr    = len(wins) / len(trades_ret) if trades_ret else 0
aw    = np.mean(wins)  if wins  else 0
al    = np.mean([abs(r) for r in loses]) if loses else 0
plr   = aw / al if al > 0 else float("inf")

from IPython.display import display, HTML
display(HTML(f'''
<table style="border-collapse:collapse;width:100%;font-size:14px;">
<tr style="background:#f8f9fa;font-weight:bold;">
  <td style="padding:10px 16px;border-bottom:2px solid #e0e0e0;">指标</td>
  <td style="padding:10px 16px;border-bottom:2px solid #e0e0e0;text-align:right;">策略</td>
  <td style="padding:10px 16px;border-bottom:2px solid #e0e0e0;text-align:right;">基准</td>
</tr>
<tr><td>总收益率</td><td style="text-align:right;color:{"#D85A30" if total_ret>0 else "#1D9E75"};">{total_ret:+.2%}</td><td style="text-align:right;color:{"#D85A30" if bench_ret>0 else "#1D9E75"};">{bench_ret:+.2%}</td></tr>
<tr style="background:#f8f9fa;"><td>年化收益率</td><td style="text-align:right;">{ann_ret:+.2%}</td><td style="text-align:right;">{ann_bench:+.2%}</td></tr>
<tr><td>年化夏普</td><td style="text-align:right;font-weight:bold;">{sharpe:.2f}</td><td style="text-align:right;">-</td></tr>
<tr style="background:#f8f9fa;"><td>最大回撤</td><td style="text-align:right;color:#D85A30;">{max_dd:+.2%}</td><td style="text-align:right;color:#D85A30;">{max_dd_b:+.2%}</td></tr>
<tr><td>Calmar 比率</td><td style="text-align:right;font-weight:bold;">{calmar:.2f}</td><td style="text-align:right;">-</td></tr>
<tr style="background:#f8f9fa;"><td>胜率</td><td style="text-align:right;">{wr:.1%}</td><td style="text-align:right;">-</td></tr>
<tr><td>盈亏比</td><td style="text-align:right;font-weight:bold;">{plr:.2f}</td><td style="text-align:right;">-</td></tr>
<tr style="background:#f8f9fa;"><td>交易次数</td><td style="text-align:right;">{len(trades_ret)}</td><td style="text-align:right;">-</td></tr>
</table>
'''))"""

CELL_PLOT = """\
# ============================================================
# 7. 可视化
# ============================================================
import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 1.5, 1]})
dates = df_v["trade_date"]

ax = axes[0]
ax.plot(dates, df_v["price"],   "#B4B2A9", lw=0.8, alpha=0.7, label="close")
ax.plot(dates, df_v["ma_fast"], "#D85A30", lw=1.2, label=fast_lbl)
ax.plot(dates, df_v["ma_slow"], "#534AB7", lw=1.2, label=slow_lbl)
g = df_v[df_v["cross"] == 1]; d = df_v[df_v["cross"] == -1]
ax.scatter(g["trade_date"], g["price"], c="#1D9E75", marker="^", s=80, zorder=5, label="金叉")
ax.scatter(d["trade_date"], d["price"], c="#D85A30", marker="v", s=80, zorder=5, label="死叉")
ax.set_title(f"{CHOSEN} — 价格 & 均线 & 信号", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9); ax.set_ylabel("Price"); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(dates, 1, df_v["equity"], alpha=0.1, color="#378ADD")
ax.plot(dates, df_v["equity"], "#378ADD", lw=1.8, label="策略")
ax.plot(dates, df_v["bench_equity"], "#B4B2A9", lw=1.2, ls="dashed", label="基准")
ax.axhline(1, c="black", lw=0.5, ls="dotted")
ax.set_title("资金曲线", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9); ax.set_ylabel("净值"); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.fill_between(dates, df_v["dd"], 0, alpha=0.3, color="#D85A30")
ax.fill_between(dates, df_v["bench_dd"], 0, alpha=0.15, color="#B4B2A9")
ax.plot(dates, df_v["dd"], "#D85A30", lw=1, label="策略回撤")
ax.plot(dates, df_v["bench_dd"], "#B4B2A9", lw=0.8, ls="dashed", label="基准回撤")
ax.axhline(0, c="black", lw=0.3); ax.axhline(-0.25, c="#D85A30", lw=0.5, ls="dotted", alpha=0.5)
ax.set_title("回撤曲线", fontsize=14, fontweight="bold")
ax.legend(loc="lower left", fontsize=9); ax.set_ylabel("回撤"); ax.grid(True, alpha=0.3)

for a in axes:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()"""

CELL_FOOTER = """\
# ============================================================
# 8. 小结
# ============================================================
print(f\"\"\"
{'='*60}
  回测摘要: {CHOSEN}
  {MA_TYPE.upper()}({FAST_PERIOD}) / {MA_TYPE.upper()}({SLOW_PERIOD}) 双均线交叉
  总收益: {total_ret:+.2%} | 夏普: {sharpe:.2f} | 最大回撤: {max_dd:+.2%} | 胜率: {wr:.1%}
  交易次数: {len(trades_ret)} | 盈亏比: {plr:.2f}
{'='*60}

注意事项:
- 回测结果不代表实盘表现
- 震荡市中易产生鞭梢效应
- 修改 CHOSEN 变量可切换股票
- 修改 FAST_PERIOD / SLOW_PERIOD 可调参数

基于 BA 工作坊量化交易研究框架 · ma_crossover_strategy_spec.md
\"\"\")"""

# ============================================================
# 生成 Notebook
# ============================================================
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"}
}

pairs = [
    ("markdown", CELL_INTRO),
    ("code",     CELL_PARAMS),
    ("code",     CELL_DATA),
    ("code",     CELL_MA),
    ("code",     CELL_SIGNAL),
    ("code",     CELL_BACKTEST),
    ("code",     CELL_METRICS),
    ("code",     CELL_PLOT),
    ("code",     CELL_FOOTER),
]

for ct, src in pairs:
    if ct == "markdown":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nb_path = os.path.join(OUTPUT, "ma_crossover_backtest.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook saved: {nb_path}")

# ============================================================
# 执行 Notebook 获取数据 → 生成 HTML 看板
# ============================================================
import pandas as pd, numpy as np, json

df = pd.read_csv(DATA, encoding="utf-8-sig", parse_dates=["trade_date"])
df = df.sort_values("trade_date").reset_index(drop=True)
df["price"] = df["close_qfq"]

FAST, SLOW, MT = 10, 60, "ema"
def calc_ma(s, n, t):
    return s.ewm(span=n, adjust=False).mean() if t == "ema" else s.rolling(n).mean()

df["ma_fast"] = calc_ma(df["price"], FAST, MT)
df["ma_slow"] = calc_ma(df["price"], SLOW, MT)
warmup = max(FAST, SLOW)
df_v = df.iloc[warmup:].copy().reset_index(drop=True)

df_v["signal"]   = (df_v["ma_fast"] > df_v["ma_slow"]).astype(int)
df_v["cross"]    = df_v["signal"].diff()
df_v["position"] = df_v["signal"].shift(1).fillna(0).astype(int)

df_v["price_ret"] = df_v["price"].pct_change()
df_v["strat_ret_gross"] = df_v["position"] * df_v["price_ret"]
df_v["trade"] = df_v["position"].diff().abs()
df_v["cost"]  = df_v["trade"] * (0.0003 + 0.001)
df_v.loc[df_v["position"].diff() < 0, "cost"] += 0.0005
df_v["strat_ret"] = df_v["strat_ret_gross"] - df_v["cost"]
df_v["equity"]       = (1 + df_v["strat_ret"].fillna(0)).cumprod()
df_v["bench_equity"] = (1 + df_v["price_ret"].fillna(0)).cumprod()
pk_s = df_v["equity"].cummax(); pk_b = df_v["bench_equity"].cummax()
df_v["dd"] = (df_v["equity"] - pk_s) / pk_s
df_v["bench_dd"] = (df_v["bench_equity"] - pk_b) / pk_b

ret = df_v["strat_ret"].dropna(); br = df_v["price_ret"].dropna(); N = len(ret)
total_ret = df_v["equity"].iloc[-1] - 1
bench_ret = df_v["bench_equity"].iloc[-1] - 1
ann_ret   = (1 + total_ret) ** (252 / N) - 1
ann_bench = (1 + bench_ret) ** (252 / N) - 1
rc = ret[ret != 0] if len(ret[ret != 0]) > 0 else ret
sharpe = np.sqrt(252) * rc.mean() / rc.std() if rc.std() > 0 else 0
max_dd = df_v["dd"].min(); max_dd_b = df_v["bench_dd"].min()
calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

g_entry = df_v[df_v["cross"] == 1]
trades_ret = []
for i in range(len(g_entry)):
    e_idx = df_v[df_v["trade_date"] == g_entry["trade_date"].iloc[i]].index[0]
    x_idx = len(df_v) - 1 if i + 1 >= len(g_entry) else df_v[df_v["trade_date"] == g_entry["trade_date"].iloc[i+1]].index[0] - 1
    trades_ret.append(df_v.loc[x_idx, "price"] / df_v.loc[e_idx, "price"] - 1)

wins = [r for r in trades_ret if r > 0]; loses = [r for r in trades_ret if r <= 0]
wr = len(wins) / len(trades_ret) if trades_ret else 0

# JSON 数据
dates_j    = df_v["trade_date"].dt.strftime("%Y-%m-%d").tolist()
price_j    = df_v["price"].round(2).tolist()
ma_fast_j  = df_v["ma_fast"].round(2).tolist()
ma_slow_j  = df_v["ma_slow"].round(2).tolist()
equity_j   = df_v["equity"].round(6).tolist()
bench_eq_j = df_v["bench_equity"].round(6).tolist()
dd_j       = df_v["dd"].round(6).tolist()
bench_dd_j = df_v["bench_dd"].round(6).tolist()

gm = df_v["cross"] == 1; dm = df_v["cross"] == -1
gd = [dates_j[i] for i in df_v[gm].index]; gp = [price_j[i] for i in df_v[gm].index]
ddt = [dates_j[i] for i in df_v[dm].index]; dp = [price_j[i] for i in df_v[dm].index]

trade_rows = ""
for i, (d, p, r) in enumerate(zip(
    [str(x.date()) for x in g_entry["trade_date"]],
    [round(x, 2) for x in g_entry["price"]],
    trades_ret
)):
    trade_rows += f'<tr><td>{d}</td><td>¥{p:.2f}</td><td><span class="badge {"win" if r>0 else "loss"}">{"盈利" if r>0 else "亏损"}</span></td><td class="{"up" if r>0 else "down"}">{r:+.2%}</td></tr>'

# HTML
html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>均线交叉策略回测看板</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root{--bg:#fff;--bg2:#f8f9fa;--text:#1a1a2e;--t2:#555;--bd:#e0e0e0;--g:#1D9E75;--r:#D85A30;--p:#534AB7;--b:#378ADD;--rd:10px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:var(--text);background:#f5f5f5;line-height:1.7}
.header{background:#fff;border-bottom:1px solid var(--bd);padding:16px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.header h1{font-size:20px;font-weight:600}.header .sub{font-size:12px;color:#888}
.ctrls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.ctrls select,.ctrls input{padding:5px 10px;border:1px solid var(--bd);border-radius:6px;font-size:12px;background:#fff}
.ctrls input[type=number]{width:56px;text-align:center}
.ctrls button{padding:5px 16px;background:var(--p);color:#fff;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500}
.ctrls button:hover{opacity:.85}
.ctrls .hint{font-size:11px;color:#aaa}
.container{max-width:1200px;margin:0 auto;padding:18px 22px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(124px,1fr));gap:8px;margin-bottom:18px}
.metric{background:#fff;border:1px solid var(--bd);border-radius:var(--rd);padding:12px;text-align:center}
.metric .lbl{font-size:11px;color:#888;margin-bottom:2px}
.metric .val{font-size:18px;font-weight:600}
.card{background:#fff;border:1px solid var(--bd);border-radius:var(--rd);padding:14px;margin-bottom:14px}
.card h3{font-size:14px;font-weight:600;margin-bottom:10px}
.chart{width:100%;height:380px}
.up{color:var(--r)}.down{color:var(--g)}.neutral{color:var(--text)}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:500}
.badge.win{background:#E1F5EE;color:#0F6E56}.badge.loss{background:#FAECE7;color:#D85A30}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:var(--bg2);text-align:left;padding:8px 12px;font-weight:600;border-bottom:2px solid var(--bd)}
td{padding:6px 12px;border-bottom:1px solid var(--bd)}
tr:hover td{background:#fafafa}
.foot{padding:10px 14px;font-size:12px;color:#888}
""" + "</style></head><body>" + f"""
<div class="header">
  <div><h1>均线交叉策略回测看板</h1><p class="sub">兆易创新 (603986.SH) · EMA(10)/EMA(60) · {dates_j[0]} ~ {dates_j[-1]} · {len(df_v)} 交易日</p></div>
  <div class="ctrls">
    <span style="font-size:12px;color:#888;">切换股票需重新运行 Notebook</span>
  </div>
</div>
<div class="container">
<div class="metrics">
  <div class="metric"><div class="lbl">策略总收益</div><div class="val {'up' if total_ret>0 else 'down'}">{total_ret:+.2%}</div></div>
  <div class="metric"><div class="lbl">基准收益</div><div class="val {'up' if bench_ret>0 else 'down'}">{bench_ret:+.2%}</div></div>
  <div class="metric"><div class="lbl">年化收益</div><div class="val neutral">{ann_ret:+.2%}</div></div>
  <div class="metric"><div class="lbl">年化夏普</div><div class="val neutral">{sharpe:.2f}</div></div>
  <div class="metric"><div class="lbl">最大回撤</div><div class="val down">{max_dd:+.2%}</div></div>
  <div class="metric"><div class="lbl">Calmar</div><div class="val neutral">{calmar:.2f}</div></div>
  <div class="metric"><div class="lbl">胜率</div><div class="val neutral">{wr:.1%}</div></div>
  <div class="metric"><div class="lbl">交易次数</div><div class="val neutral">{len(trades_ret)}</div></div>
</div>

<div class="card"><h3>价格 & 均线 & 交易信号</h3><div class="chart" id="pChart"></div></div>
<div class="card"><h3>资金曲线</h3><div class="chart" id="eChart"></div></div>
<div class="card"><h3>回撤曲线</h3><div class="chart" id="dChart"></div></div>
<div class="card"><h3>交易记录</h3><div style="overflow-x:auto"><table>
<tr><th>日期</th><th>入场价</th><th>结果</th><th>收益率</th></tr>
""" + trade_rows + """
</table></div></div>
</div>
<script>
var D=""" + json.dumps(dates_j) + """;
var P=""" + json.dumps(price_j) + """;
var MF=""" + json.dumps(ma_fast_j) + """;
var MS=""" + json.dumps(ma_slow_j) + """;
var EQ=""" + json.dumps(equity_j) + """;
var BQ=""" + json.dumps(bench_eq_j) + """;
var DD=""" + json.dumps(dd_j) + """;
var BD=""" + json.dumps(bench_dd_j) + """;
var GD=""" + json.dumps(gd) + """;
var GP=""" + json.dumps(gp) + """;
var DD2=""" + json.dumps(ddt) + """;
var DP=""" + json.dumps(dp) + """;

function mk(id,o){var c=echarts.init(document.getElementById(id));c.setOption(o);return c}
var pC=mk("pChart",{tooltip:{trigger:"axis"},legend:{data:["close","EMA(10)","EMA(60)","金叉","死叉"],bottom:0},grid:{left:70,right:30,top:14,bottom:36},xAxis:{type:"category",data:D,axisLabel:{formatter:function(v){return v.slice(5)}}},yAxis:{type:"value",axisLabel:{formatter:function(v){return"¥"+v.toFixed(0)}}},dataZoom:[{type:"inside"},{type:"slider",bottom:2}],series:[{name:"close",type:"line",data:P,lineStyle:{color:"#B4B2A9",width:1},symbol:"none",z:1},{name:"EMA(10)",type:"line",data:MF,lineStyle:{color:"#D85A30",width:1.5},symbol:"none",z:2},{name:"EMA(60)",type:"line",data:MS,lineStyle:{color:"#534AB7",width:1.5},symbol:"none",z:2},{name:"金叉",type:"scatter",data:GD.map(function(d,i){return[d,GP[i]]}),symbol:"triangle",symbolSize:13,itemStyle:{color:"#1D9E75"},z:3},{name:"死叉",type:"scatter",data:DD2.map(function(d,i){return[d,DP[i]]}),symbol:"triangle",symbolRotate:180,symbolSize:13,itemStyle:{color:"#D85A30"},z:3}]});
var eC=mk("eChart",{tooltip:{trigger:"axis",formatter:function(p){return p[0].axisValue+"<br/>"+p.map(function(x){return x.marker+x.seriesName+": "+(x.value*100).toFixed(2)+"%"}).join("<br/>")}},legend:{data:["策略净值","基准 (Buy & Hold)"],bottom:0},grid:{left:70,right:30,top:14,bottom:36},xAxis:{type:"category",data:D,axisLabel:{formatter:function(v){return v.slice(5)}}},yAxis:{type:"value",axisLabel:{formatter:function(v){return(v*100).toFixed(0)+"%"}}},dataZoom:[{type:"inside"},{type:"slider",bottom:2}],series:[{name:"策略净值",type:"line",data:EQ,lineStyle:{color:"#378ADD",width:2},areaStyle:{color:"rgba(55,138,221,0.06)"},symbol:"none",z:2},{name:"基准 (Buy & Hold)",type:"line",data:BQ,lineStyle:{color:"#B4B2A9",width:1.2,type:"dashed"},symbol:"none",z:1}]});
var dC=mk("dChart",{tooltip:{trigger:"axis",formatter:function(p){return p[0].axisValue+"<br/>"+p.map(function(x){return x.marker+x.seriesName+": "+(x.value*100).toFixed(2)+"%"}).join("<br/>")}},legend:{data:["策略回撤","基准回撤"],bottom:0},grid:{left:70,right:30,top:14,bottom:36},xAxis:{type:"category",data:D,axisLabel:{formatter:function(v){return v.slice(5)}}},yAxis:{type:"value",axisLabel:{formatter:function(v){return(v*100).toFixed(0)+"%"}},max:0},dataZoom:[{type:"inside"},{type:"slider",bottom:2}],series:[{name:"策略回撤",type:"line",data:DD,lineStyle:{color:"#D85A30",width:1.5},areaStyle:{color:"rgba(216,90,48,0.1)"},symbol:"none",z:2},{name:"基准回撤",type:"line",data:BD,lineStyle:{color:"#B4B2A9",width:1,type:"dashed"},symbol:"none",z:1},{name:"熔断线",type:"line",markLine:{silent:true,symbol:"none",lineStyle:{color:"#D85A30",type:"dotted",width:1},data:[{yAxis:-0.25,label:{formatter:"-25% 熔断",color:"#D85A30"}}]},z:0}]});
[window].forEach(function(w){w.addEventListener("resize",function(){pC.resize();eC.resize();dC.resize()})});
</script>
<div class="foot">回测计入佣金(万三)+印花税(万五,卖出)+滑点(千一) | 信号T+1执行 | 回测结果不代表实盘表现</div>
</body></html>"""

html_path = os.path.join(OUTPUT, "ma_crossover_dashboard.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved: {html_path}")
print("Done!")
