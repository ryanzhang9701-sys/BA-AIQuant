#!/usr/bin/env python
"""Generate Jupyter Notebook for Turtle Strategy Backtest"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13.0"
    }
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ====== Cell 0: Title ======
md("""# 海龟策略回测 · 跨股票对比

> **Spec v1.0.0** | 双系统（20/10 + 55/20）| 金字塔加仓 | ATR仓位管理 | 三层风控

本 Notebook 基于 `turtle_strategy_spec.md` 规范，对三只 A 股（中芯国际、比亚迪、兆易创新）进行海龟策略独立模式回测，并生成对比看板。

**核心问题**：海龟趋势跟踪策略对这三只股票中的哪一只最有效？""")

# ====== Cell 1: Setup ======
md("""## 1. 环境准备""")

code("""import sys
sys.path.insert(0, '.')
from turtle_backtest import run_all, make_dashboard_data, STOCKS, PARAMS, OUTPUT_DIR
import pandas as pd
import json
import webbrowser

print("✓ 回测引擎加载完成")
print(f"  策略参数:")
print(f"    系统1: {PARAMS['s1_entry']}日入场 / {PARAMS['s1_exit']}日出场")
print(f"    系统2: {PARAMS['s2_entry']}日入场 / {PARAMS['s2_exit']}日出场")
print(f"    ATR周期: {PARAMS['atr_period']}")
print(f"    单笔风险: {PARAMS['risk_pct']*100}%")
print(f"    最大加仓: {PARAMS['max_units']} Unit")
print(f"    止损距离: {PARAMS['stop_n']} N")
print(f"    初始资金: ¥{PARAMS['account_init']:,}")
print(f"  目标股票:")
for s in STOCKS:
    print(f"    {s['name']} ({s['ts_code']}) — {s.get('group','')}")""")

# ====== Cell 2: Run Backtest ======
md("""## 2. 运行回测""")

code("""%%time
all_results = run_all()""")

# ====== Cell 3: Performance Summary ======
md("""## 3. 绩效总览""")

code("""# 生成对比表
rows = []
for r in all_results:
    p = r['performance']
    rows.append({
        '股票': r['name'],
        '代码': r['ts_code'],
        '年化收益': f\"{p.get('annual_return', 0):.1%}\",
        '最大回撤': f\"{p.get('max_drawdown', 0):.1%}\",
        '夏普比率': f\"{p.get('sharpe', 0):.2f}\",
        '胜率': f\"{p.get('win_rate', 0):.0%}\",
        '盈亏因子': f\"{p.get('profit_factor', 0):.2f}\",
        '交易笔数': p.get('total_trades', 0),
        '平均持仓天数': f\"{p.get('avg_hold_days', 0):.0f}\",
    })

summary_df = pd.DataFrame(rows)
summary_df.style \
    .background_gradient(subset=['年化收益'], cmap='RdYlGn') \
    .background_gradient(subset=['最大回撤'], cmap='RdYlGn_r') \
    .background_gradient(subset=['夏普比率'], cmap='Blues')""")

# ====== Cell 4: Equity Curve ======
md("""## 4. 净值曲线对比""")

code("""import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 6))
colors = {'中芯国际': '#3B82F6', '比亚迪': '#F97316', '兆易创新': '#8B5CF6'}

for r in all_results:
    eq = r['equity']
    init = eq['equity'].iloc[0]
    norm = eq['equity'] / init * 100
    ax.plot(eq['date'], norm, label=r['name'], color=colors.get(r['name'], '#333'), linewidth=2)

ax.axhline(y=100, color='#94a3b8', linestyle='--', linewidth=1, label='基准线 (100)')
ax.set_title('海龟策略净值曲线 — 独立模式', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('净值 (初始=100)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()""")

# ====== Cell 5: Trade Analysis ======
md("""## 5. 交易分布分析""")

code("""# 盈亏分布
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, r in enumerate(all_results):
    ax = axes[idx]
    pos = r['positions']
    if pos.empty:
        ax.text(0.5, 0.5, '无交易', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(r['name'])
        continue

    colors_bar = ['#EF4444' if p < 0 else '#22C55E' for p in pos['pnl']]
    ax.bar(range(len(pos)), pos['pnl'], color=colors_bar, alpha=0.8)
    ax.axhline(y=0, color='#333', linewidth=0.5)
    ax.set_title(f\"{r['name']} — {len(pos)} 笔交易\", fontsize=13, fontweight='bold')
    ax.set_xlabel('交易序号')
    ax.set_ylabel('盈亏 (¥)')
    total_pnl = pos['pnl'].sum()
    ax.text(0.02, 0.95, f'总盈亏: ¥{total_pnl:,.0f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            color='#EF4444' if total_pnl < 0 else '#22C55E')

plt.tight_layout()
plt.show()""")

# ====== Cell 6: Generate Dashboard ======
md("""## 6. 生成对比看板""")

code("""# 生成 dashboard JSON 数据
dash_data = make_dashboard_data(all_results)
with open(OUTPUT_DIR / 'dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(dash_data, f, ensure_ascii=False, indent=2, default=str)

print("✓ Dashboard 数据已生成")

# 打开 HTML 看板
import os
html_path = os.path.abspath('turtle_comparison.html')
print(f"  看板路径: {html_path}")

# 启动本地服务器以正确加载 fetch 数据
import http.server
import socketserver
import threading

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.', **kwargs)

def start_server():
    with socketserver.TCPServer(("", 8765), Handler) as httpd:
        httpd.serve_forever()

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
print(f"  本地服务器: http://localhost:8765/turtle_comparison.html")
print(f"  (请手动在浏览器中打开上方链接)")""")

# ====== Cell 7: Conclusion ======
md("""## 7. 结论

回测完成后，观察以下维度判断"海龟策略对哪只股票更有效"：

| 维度 | 好信号 | 坏信号 |
|------|--------|--------|
| 净值曲线形态 | 台阶式稳步上升 | 剧烈震荡后回归起点 |
| 盈亏因子 | > 1.5 | < 1.0 |
| 最大回撤 | < 20% | > 35% |
| 平均持仓天数 | 与趋势持续性匹配 | 过短（频繁假突破） |
| 月度热力图 | 大部分月份红盘 | 集中在少数月份盈利 |

> ⚠ **样本量提醒**：三只股票的回测结果不足以得出统计意义上可靠的结论。更准确的判断需要 10+ 只股票、多行业、多周期数据的验证。""")

nb.cells = cells

with open('turtle_backtest.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 已生成: TASK4/turtle_backtest.ipynb")
