#!/usr/bin/env python
"""Generate single-stock Turtle Strategy deep-dive analysis notebook"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"}
}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ====== Cell 0 ======
md("""# 海龟策略 · 单股票深度分析

> 选择一只股票，查看海龟趋势跟踪策略的完整运行过程——从原始指标（TR/ATR/唐奇安通道）到交易信号，再到持仓演化与绩效评估。

**数据来源**：A股 (tushare, 前复权 `daily_adjusted.csv`) + H股 (akshare, `daily_hk.csv`)

**策略参数**：系统1(20/10) + 系统2(55/20) | ATR(20) | 风险1% | 止损2N | 金字塔加仓4U/0.5N""")

# ====== Cell 1: Stock selector ======
md("""## 1. 选择股票""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载全部5只股票
BASE = Path.cwd().parent
STOCKS = [
    {"code":"688981.SH","name":"中芯国际(A)","file":"data/688981.SH_中芯国际/daily_adjusted.csv","cols":{"d":"trade_date","o":"open_qfq","h":"high_qfq","l":"low_qfq","c":"close_qfq"},"lot":100},
    {"code":"00981.HK","name":"中芯国际(H)","file":"data/688981.SH_中芯国际/daily_hk.csv","cols":{"d":"trade_date","o":"open","h":"high","l":"low","c":"close"},"lot":500},
    {"code":"002594.SZ","name":"比亚迪(A)","file":"data/002594.SZ_比亚迪/daily_adjusted.csv","cols":{"d":"trade_date","o":"open_qfq","h":"high_qfq","l":"low_qfq","c":"close_qfq"},"lot":100},
    {"code":"01211.HK","name":"比亚迪(H)","file":"data/002594.SZ_比亚迪/daily_hk.csv","cols":{"d":"trade_date","o":"open","h":"high","l":"low","c":"close"},"lot":500},
    {"code":"603986.SH","name":"兆易创新","file":"data/603986.SH_兆易创新/daily_adjusted.csv","cols":{"d":"trade_date","o":"open_qfq","h":"high_qfq","l":"low_qfq","c":"close_qfq"},"lot":100},
]

all_data = {}
for s in STOCKS:
    path = BASE / s["file"]
    df = pd.read_csv(path, encoding="utf-8-sig")
    cc = s["cols"]
    df = df[[cc[k] for k in ["d","o","h","l","c"]]].copy()
    df.columns = ["date","open","high","low","close"]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    s["df"] = df
    s["n_rows"] = len(df)
    s["date_range"] = f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}"
    all_data[s["code"]] = s

print("已加载股票：")
for s in STOCKS:
    print(f"  [{s['code']}] {s['name']:10s}  {s['n_rows']:4d} 行  {s['date_range']}  (手数: {s['lot']})")
print(f"\\n请在下方输入股票代码以选择分析对象（例如 603986.SH）：")""")

# ====== Cell 2: Select stock ======
md("""## 2. 设定分析标的""")

code("""# ---- 修改这里的代码来选择股票 ----
SELECTED = "603986.SH"   # 可选: 688981.SH / 00981.HK / 002594.SZ / 01211.HK / 603986.SH
# ---------------------------------

stock = all_data[SELECTED]
df = stock["df"].copy()
LOT = stock["lot"]
print(f"已选择: [{stock['code']}] {stock['name']}")
print(f"数据: {stock['n_rows']} 行, {stock['date_range']}")
print(f"手数: {LOT} 股/手, 数据列: date, open, high, low, close")
df.head(3)""")

# ====== Cell 3: Compute indicators ======
md("""## 3. 计算海龟策略指标""")

code("""# ---- 策略参数 ----
S1_ENTRY, S1_EXIT = 20, 10
S2_ENTRY, S2_EXIT = 55, 20
ATR_P = 20
RISK_PCT = 0.01
STOP_N = 2.0
MAX_U = 4
ADD_N = 0.5
SLIPPAGE = 0.001
ACCOUNT = 1_000_000

n = len(df)
h, l, c, o_vals = df["high"].values, df["low"].values, df["close"].values, df["open"].values

# ---------- TR ----------
tr = np.zeros(n)
tr[0] = h[0] - l[0]
for i in range(1, n):
    tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
df["tr"] = tr

# ---------- ATR(20) ----------
atr = np.full(n, np.nan)
atr[ATR_P-1] = tr[:ATR_P].mean()
for i in range(ATR_P, n):
    atr[i] = (atr[i-1]*(ATR_P-1) + tr[i]) / ATR_P
df["N"] = atr

# ---------- Donchian Channels ----------
s1u = np.full(n, np.nan); s1l = np.full(n, np.nan)
s2u = np.full(n, np.nan); s2l = np.full(n, np.nan)
for i in range(S1_ENTRY, n):
    s1u[i] = h[i-S1_ENTRY:i].max()
for i in range(S1_EXIT, n):
    s1l[i] = l[i-S1_EXIT:i].min()
for i in range(S2_ENTRY, n):
    s2u[i] = h[i-S2_ENTRY:i].max()
for i in range(S2_EXIT, n):
    s2l[i] = l[i-S2_EXIT:i].min()
df["s1_upper"] = s1u; df["s1_lower"] = s1l
df["s2_upper"] = s2u; df["s2_lower"] = s2l

print("✓ 指标计算完成")
print(f"  TR 范围: {tr.min():.2f} ~ {tr.max():.2f}")
print(f"  ATR(20) 最后值: {atr[-1]:.2f}")
print(f"  唐奇安通道已计算 (系统1: {S1_ENTRY}/{S1_EXIT}日, 系统2: {S2_ENTRY}/{S2_EXIT}日)")""")

# ====== Cell 4: Run backtest ======
md("""## 4. 运行海龟策略回测""")

code("""# ---- 回测引擎 ----
cash = ACCOUNT
holdings = []       # {sys, ep, en, u, shares, stop, addP}
positions = []      # 完整交易记录
signals = []        # 信号日志
equity_curve = []   # 每日净值
last_s1_pnl = 0     # 系统1重入过滤

def unit_shares(nv):
    raw = int(cash * RISK_PCT / (nv * LOT))
    return max(raw * LOT, LOT)

start = max(S2_ENTRY, ATR_P) + 1

for i in range(start, n):
    today = df.iloc[i]
    nv = today["N"]
    if pd.isna(nv) or nv <= 0:
        equity_curve.append({"date": today["date"], "equity": cash, "units": 0})
        continue

    # --- 止损 ---
    rm = []
    for hh in holdings:
        if today["low"] < hh["stop"]:
            px = hh["stop"]
            cash += px * hh["shares"]
            positions.append({"sys": hh["sys"], "entry_date": hh["entry_date"], "ep": hh["ep"], "en": hh["en"],
                              "exit_date": today["date"], "exp": px, "xr": "STOP_LOSS",
                              "u": hh["u"], "shares": hh["shares"], "pnl": (px - hh["ep"]) * hh["shares"]})
            if hh["sys"] == "S1": last_s1_pnl = (px - hh["ep"]) * hh["shares"]
            rm.append(hh)
    for hh in rm: holdings.remove(hh)

    # --- S2 出场 ---
    if not pd.isna(today["s2_lower"]) and today["close"] < today["s2_lower"]:
        for hh in list(holdings):
            if hh["sys"] == "S2":
                px = o_vals[min(i+1, n-1)] * (1 - SLIPPAGE)
                cash += px * hh["shares"]
                positions.append({"sys": "S2", "entry_date": hh["entry_date"], "ep": hh["ep"], "en": hh["en"],
                                  "exit_date": today["date"], "exp": px, "xr": "TRAILING_S2",
                                  "u": hh["u"], "shares": hh["shares"], "pnl": (px - hh["ep"]) * hh["shares"]})
                holdings.remove(hh)

    # --- S1 出场 ---
    if not pd.isna(today["s1_lower"]) and today["close"] < today["s1_lower"]:
        for hh in list(holdings):
            if hh["sys"] == "S1":
                px = o_vals[min(i+1, n-1)] * (1 - SLIPPAGE)
                cash += px * hh["shares"]
                positions.append({"sys": "S1", "entry_date": hh["entry_date"], "ep": hh["ep"], "en": hh["en"],
                                  "exit_date": today["date"], "exp": px, "xr": "TRAILING_S1",
                                  "u": hh["u"], "shares": hh["shares"], "pnl": (px - hh["ep"]) * hh["shares"]})
                last_s1_pnl = (px - hh["ep"]) * hh["shares"]
                holdings.remove(hh)

    # --- S2 入场 ---
    if not pd.isna(today["s2_upper"]) and today["close"] > today["s2_upper"]:
        if not any(hh["sys"] == "S2" for hh in holdings) and i + 1 < n:
            px = o_vals[i+1] * (1 + SLIPPAGE)
            us = unit_shares(nv)
            if us > 0 and px * us <= cash:
                cash -= px * us
                holdings.append({"sys": "S2", "entry_date": df.iloc[i+1]["date"], "ep": px, "en": nv,
                                 "u": 1, "shares": us, "stop": px - STOP_N * nv, "addP": [px]})
                signals.append({"date": today["date"], "system": "S2", "type": "BUY", "price": px, "units": 1})

    # --- S1 入场 ---
    if not pd.isna(today["s1_upper"]) and today["close"] > today["s1_upper"]:
        if not any(hh["sys"] == "S1" for hh in holdings) and last_s1_pnl >= 0 and i + 1 < n:
            px = o_vals[i+1] * (1 + SLIPPAGE)
            us = unit_shares(nv)
            if us > 0 and px * us <= cash:
                cash -= px * us
                holdings.append({"sys": "S1", "entry_date": df.iloc[i+1]["date"], "ep": px, "en": nv,
                                 "u": 1, "shares": us, "stop": px - STOP_N * nv, "addP": [px]})
                signals.append({"date": today["date"], "system": "S1", "type": "BUY", "price": px, "units": 1})

    # --- 加仓 ---
    for hh in holdings:
        if hh["u"] >= MAX_U or i + 1 >= n:
            continue
        needed = hh["addP"][-1] + ADD_N * hh["en"]
        if today["close"] >= needed:
            px = o_vals[i+1] * (1 + SLIPPAGE)
            us = unit_shares(nv)
            if us > 0 and px * us <= cash:
                cash -= px * us
                hh["u"] += 1; hh["shares"] += us; hh["addP"].append(px)
                hh["stop"] = max(hh["stop"], px - STOP_N * nv)
                signals.append({"date": today["date"], "system": hh["sys"], "type": "ADD", "price": px, "units": hh["u"]})

    # --- 日终净值 ---
    mkt = sum(hh["shares"] * today["close"] for hh in holdings)
    equity_curve.append({"date": today["date"], "equity": cash + mkt, "cash": cash,
                         "units": sum(hh["u"] for hh in holdings)})

# 强制平仓
for hh in list(holdings):
    px = df.iloc[-1]["close"]
    cash += px * hh["shares"]
    positions.append({"sys": hh["sys"], "entry_date": hh["entry_date"], "ep": hh["ep"], "en": hh["en"],
                      "exit_date": df.iloc[-1]["date"], "exp": px, "xr": "END_OF_DATA",
                      "u": hh["u"], "shares": hh["shares"], "pnl": (px - hh["ep"]) * hh["shares"]})
    holdings = []

eq_df = pd.DataFrame(equity_curve)
pos_df = pd.DataFrame(positions)
sig_df = pd.DataFrame(signals)

# 绩效指标
eq_vals = eq_df["equity"].values
init_eq = eq_vals[0]
total_ret = eq_vals[-1] / init_eq - 1
ann_ret = (1 + total_ret) ** (252 / len(eq_vals)) - 1
peak = np.maximum.accumulate(eq_vals)
dd = (eq_vals - peak) / peak
max_dd = dd.min()
rets = np.diff(eq_vals) / eq_vals[:-1]
daily_rf = 0.02 / 252
sharpe = (rets.mean() - daily_rf) / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

winners = pos_df[pos_df["pnl"] > 0] if len(pos_df) > 0 else pd.DataFrame()
losers = pos_df[pos_df["pnl"] <= 0] if len(pos_df) > 0 else pd.DataFrame()
total_win = winners["pnl"].sum() if len(winners) > 0 else 0
total_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 0

print("=" * 50)
print(f"  {stock['name']} ({stock['code']}) 回测完成")
print("=" * 50)
print(f"  年化收益率: {ann_ret:+.1%}")
print(f"  最大回撤:   {max_dd:.1%}")
print(f"  夏普比率:   {sharpe:.2f}")
print(f"  胜率:       {len(winners)/len(pos_df):.0%}" if len(pos_df) > 0 else "  胜率:       N/A")
print(f"  盈亏因子:   {total_win/total_loss:.2f}" if total_loss > 0 else "  盈亏因子:   N/A")
print(f"  总交易笔数: {len(pos_df)}")
print(f"  持仓天数均值: {(pos_df['exit_date'] - pos_df['entry_date']).dt.days.mean():.0f}" if len(pos_df) > 0 else "  N/A")""")

# ====== Cell 5: K-line + channels ======
md("""## 5. K线图 + 唐奇安通道""")

code("""fig, ax = plt.subplots(figsize=(18, 7))

# K线 — 用收盘价折线代替（日线级别折线更清晰）
ax.plot(df["date"], df["close"], color="#1a1a2e", linewidth=0.8, alpha=0.7, label="收盘价")

# 唐奇安通道
mask = df["s2_upper"].notna()
ax.plot(df.loc[mask, "date"], df.loc[mask, "s2_upper"], color="#3B82F6", linewidth=1.2, linestyle="--", label=f"系统2上轨 ({S2_ENTRY}日)")
ax.plot(df.loc[mask, "date"], df.loc[mask, "s2_lower"], color="#3B82F6", linewidth=1.2, linestyle="--", label=f"系统2下轨 ({S2_EXIT}日)")
ax.fill_between(df.loc[mask, "date"], df.loc[mask, "s2_upper"], df.loc[mask, "s2_lower"], color="#DBEAFE", alpha=0.15)

mask1 = df["s1_upper"].notna()
ax.plot(df.loc[mask1, "date"], df.loc[mask1, "s1_upper"], color="#22C55E", linewidth=0.8, linestyle=":", label=f"系统1上轨 ({S1_ENTRY}日)")
ax.plot(df.loc[mask1, "date"], df.loc[mask1, "s1_lower"], color="#EF4444", linewidth=0.8, linestyle=":", label=f"系统1下轨 ({S1_EXIT}日)")

# 标注信号
for _, s in sig_df.iterrows():
    if s["type"] == "BUY":
        ax.scatter(s["date"], s["price"], color="#EF4444", marker="^", s=80, zorder=5, edgecolors="white", linewidths=0.5)
    elif s["type"] == "SELL" or s["type"] == "STOP":
        ax.scatter(s["date"], s["price"], color="#22C55E", marker="v", s=80, zorder=5, edgecolors="white", linewidths=0.5)
    elif s["type"] == "ADD":
        ax.scatter(s["date"], s["price"], color="#F97316", marker="+", s=60, zorder=5, linewidths=1.5)

ax.set_title(f"{stock['name']} ({stock['code']}) — 海龟策略唐奇安通道与交易信号", fontsize=15, fontweight="bold")
ax.set_ylabel("价格 (元)", fontsize=12)
ax.legend(loc="upper left", fontsize=9, ncol=3)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""")

# ====== Cell 6: TR and ATR ======
md("""## 6. 波动率指标 — TR 与 ATR(N)""")

code("""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

# TR
ax1.fill_between(df["date"], df["tr"], alpha=0.4, color="#3B82F6", label="True Range")
ax1.plot(df["date"], df["tr"], color="#3B82F6", linewidth=0.8)
ax1.axhline(y=df["tr"].mean(), color="#EF4444", linestyle="--", linewidth=1, label=f"均值 {df['tr'].mean():.2f}")
ax1.set_ylabel("TR (元)", fontsize=12)
ax1.set_title("True Range — 每日真实波幅", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# ATR(N)
ax2.plot(df["date"], df["N"], color="#8B5CF6", linewidth=1.5, label=f"ATR({ATR_P}) = N")
ax2.fill_between(df["date"], df["N"], alpha=0.15, color="#8B5CF6")
ax2.set_ylabel("N (元)", fontsize=12)
ax2.set_title(f"ATR({ATR_P}) — 平均真实波幅（指数平滑）", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

fig.autofmt_xdate()
plt.tight_layout()
plt.show()""")

# ====== Cell 7: Equity + Drawdown ======
md("""## 7. 净值曲线与回撤""")

code("""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

# 净值
ax1.plot(eq_df["date"], eq_df["equity"] / init_eq * 100, color="#3B82F6", linewidth=2, label="海龟策略净值")
ax1.axhline(y=100, color="#94a3b8", linestyle="--", linewidth=1, label="初始净值 100")
# 买入持有
bh = df["close"] / df["close"].iloc[0] * 100
ax1.plot(df["date"], bh, color="#F97316", linewidth=1, alpha=0.6, linestyle=":", label="买入持有")
ax1.fill_between(eq_df["date"], eq_df["equity"] / init_eq * 100, 100, alpha=0.08, color="#3B82F6")
ax1.set_ylabel("净值 (初始=100)", fontsize=12)
ax1.set_title("账户净值曲线 — 海龟策略 vs 买入持有", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# 回撤
ax2.fill_between(eq_df["date"], dd * 100, 0, color="#EF4444", alpha=0.3)
ax2.plot(eq_df["date"], dd * 100, color="#EF4444", linewidth=1)
ax2.axhline(y=dd.min() * 100, color="#991B1B", linestyle="--", linewidth=1, label=f"最大回撤 {max_dd:.1%}")
ax2.set_ylabel("回撤 (%)", fontsize=12)
ax2.set_title("回撤曲线", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

fig.autofmt_xdate()
plt.tight_layout()
plt.show()""")

# ====== Cell 8: Unit evolution ======
md("""## 8. 持仓 Unit 数演变""")

code("""fig, ax = plt.subplots(figsize=(18, 4))

ax.fill_between(eq_df["date"], eq_df["units"], alpha=0.4, color="#3B82F6")
ax.plot(eq_df["date"], eq_df["units"], color="#3B82F6", linewidth=1)
ax.axhline(y=4, color="#EF4444", linestyle="--", linewidth=0.8, alpha=0.7, label="单股上限 4U")
ax.set_ylabel("持仓 Unit 数", fontsize=12)
ax.set_title("每日持仓 Unit 数（系统1+系统2合计）", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(5, eq_df["units"].max() + 1))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""")

# ====== Cell 9: Trade P&L ======
md("""## 9. 逐笔交易盈亏分析""")

code("""if len(pos_df) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # 逐笔盈亏
    colors = ["#EF4444" if p < 0 else "#22C55E" for p in pos_df["pnl"]]
    ax1.bar(range(len(pos_df)), pos_df["pnl"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=0, color="#1a1a2e", linewidth=0.8)
    total = pos_df["pnl"].sum()
    ax1.set_title(f"逐笔盈亏 (总: ¥{total:,.0f})", fontsize=13, fontweight="bold")
    ax1.set_xlabel("交易序号")
    ax1.set_ylabel("盈亏 (¥)")
    ax1.grid(True, alpha=0.2, axis="y")

    # 系统对比
    for sys in ["S1", "S2"]:
        sub = pos_df[pos_df["sys"] == sys]
        if len(sub) == 0: continue
        ax2.bar([sys], [sub["pnl"].sum()], color="#3B82F6" if sys == "S1" else "#F97316", alpha=0.85, width=0.4)
        ax2.text(sys, sub["pnl"].sum(), f"¥{sub['pnl'].sum():,.0f}\\n{len(sub)}笔", ha="center", va="bottom" if sub["pnl"].sum() > 0 else "top", fontsize=10)
    ax2.axhline(y=0, color="#1a1a2e", linewidth=0.8)
    ax2.set_title("系统1 vs 系统2 累计盈亏", fontsize=13, fontweight="bold")
    ax2.set_ylabel("累计盈亏 (¥)")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.show()
else:
    print("该股票在回测期内无交易信号")""")

# ====== Cell 10: Exit reason pie ======
md("""## 10. 出场原因分布""")

code("""if len(pos_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    reason_counts = pos_df["xr"].value_counts()
    colors_map = {"STOP_LOSS": "#EF4444", "TRAILING_S1": "#22C55E", "TRAILING_S2": "#3B82F6", "END_OF_DATA": "#94a3b8"}
    wedges, texts, autotexts = ax.pie(
        reason_counts.values, labels=reason_counts.index, autopct="%1.1f%%",
        colors=[colors_map.get(k, "#888") for k in reason_counts.index],
        startangle=90, pctdistance=0.75
    )
    for t in autotexts: t.set_fontsize(11); t.set_fontweight("bold")
    ax.set_title("出场原因分布", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ---- 指标汇总表 ----
metrics = {
    "指标": ["年化收益率", "最大回撤", "夏普比率", "胜率", "盈亏因子", "总交易笔数",
             "平均持仓天数", "平均盈利(¥)", "平均亏损(¥)", "TR 均值", "ATR(N) 终值"],
    "数值": [
        f"{ann_ret:+.2%}", f"{max_dd:.2%}", f"{sharpe:.2f}",
        f"{len(winners)/len(pos_df):.0%}" if len(pos_df) > 0 else "N/A",
        f"{total_win/total_loss:.2f}" if total_loss > 0 else "N/A",
        len(pos_df),
        f"{(pos_df['exit_date'] - pos_df['entry_date']).dt.days.mean():.0f}" if len(pos_df) > 0 else "N/A",
        f"¥{winners['pnl'].mean():,.0f}" if len(winners) > 0 else "N/A",
        f"¥{losers['pnl'].mean():,.0f}" if len(losers) > 0 else "N/A",
        f"{df['tr'].mean():.2f}", f"{atr[-1]:.2f}",
    ]
}
pd.DataFrame(metrics).style.set_caption(f"{stock['name']} ({stock['code']}) 策略绩效汇总")""")

# ====== Cell 11: Change stock guide ======
md("""## 11. 切换股票

要分析另一只股票，回到 **Cell 2**，修改 `SELECTED` 变量为以下任一代码，然后依次运行后续所有 Cell：

| 代码 | 名称 | 市场 |
|------|------|------|
| `688981.SH` | 中芯国际(A) | A股·科创板 |
| `00981.HK` | 中芯国际(H) | H股·港股 |
| `002594.SZ` | 比亚迪(A) | A股·深市 |
| `01211.HK` | 比亚迪(H) | H股·港股 |
| `603986.SH` | 兆易创新 | A股·沪市 |""")

nb.cells = cells
with open("turtle_single_stock.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated: TASK4/turtle_single_stock.ipynb")
