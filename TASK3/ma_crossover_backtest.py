"""
均线交叉策略 — 完整回测演示
=============================
支持 SMA / EMA 双均线金叉死叉信号，含交易成本、资金曲线、绩效报告。
用法: python ma_crossover_backtest.py
输出: TASK3/ma_crossover_report.html
"""

import sys
import os
import pandas as pd
import numpy as np

# ── 路径配置 ────────────────────────────────────
PROJECT = r"C:\Users\RYAN\Desktop\BA工作坊"
DATA_DIR = os.path.join(PROJECT, "data")
OUTPUT_DIR = os.path.join(PROJECT, "TASK3")

# 可选的股票数据文件
STOCK_FILES = {
    "比亚迪 002594.SZ": os.path.join(DATA_DIR, "002594.SZ_比亚迪", "daily_adjusted.csv"),
    "中芯国际 688981.SH": os.path.join(DATA_DIR, "688981.SH_中芯国际", "daily_adjusted.csv"),
    "兆易创新 603986.SH": os.path.join(DATA_DIR, "603986.SH_兆易创新", "daily_adjusted.csv"),
}


def load_data(path):
    """读取复权后的日线数据"""
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["price"] = df["close_qfq"]  # 使用前复权收盘价
    return df[["trade_date", "price", "vol", "amount"]].copy()


def calc_signals(df, fast=10, slow=60, ma_type="ema"):
    """
    计算均线交叉信号。

    参数
    ----
    fast: int, 快线周期
    slow: int, 慢线周期
    ma_type: 'sma' 或 'ema'

    返回
    ----
    df 中增加列: ma_fast, ma_slow, signal, position
    """
    if ma_type == "ema":
        df["ma_fast"] = df["price"].ewm(span=fast, adjust=False).mean()
        df["ma_slow"] = df["price"].ewm(span=slow, adjust=False).mean()
    else:
        df["ma_fast"] = df["price"].rolling(fast).mean()
        df["ma_slow"] = df["price"].rolling(slow).mean()

    # 信号: 快线 > 慢线 = 1 (持有多头), 否则 = 0 (空仓)
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    # 交叉点: diff() 产生 +1(金叉) / -1(死叉)
    df["cross"] = df["signal"].diff()

    # 仓位: 金叉日买入, 死叉日卖出 (实际交易发生在次日, 避免未来函数)
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    return df


def run_backtest(df, commission=0.0005, slippage=0.001):
    """
    执行回测。

    参数
    ----
    commission: 单边佣金率 (默认万五)
    slippage:  滑点比例 (默认千一)

    返回
    ----
    df 中增加: strategy_return, equity, benchmark_return, benchmark_equity
    """
    df["price_return"] = df["price"].pct_change()

    # 策略日收益率 = 仓位 × 当日涨跌幅
    df["strategy_return_raw"] = df["position"] * df["price_return"]

    # 交易成本: 只在换仓日收取
    df["trade"] = df["position"].diff().abs()
    df["cost"] = df["trade"] * (commission + slippage)

    df["strategy_return"] = df["strategy_return_raw"] - df["cost"]

    # 资金曲线 (初始资金 1.0)
    df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    df["benchmark_equity"] = (1 + df["price_return"].fillna(0)).cumprod()

    return df


def compute_metrics(df):
    """计算绩效指标"""
    ret = df["strategy_return"].dropna()
    bench_ret = df["price_return"].dropna()

    total_days = len(ret)

    # 剔除 NaN 后计算
    ret_clean = ret[ret != 0] if len(ret[ret != 0]) > 0 else ret
    bench_clean = bench_ret[bench_ret != 0] if len(bench_ret[bench_ret != 0]) > 0 else bench_ret

    total_return = df["equity"].iloc[-1] - 1
    bench_return = df["benchmark_equity"].iloc[-1] - 1

    annual_return = (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0
    annual_bench = (1 + bench_return) ** (252 / total_days) - 1 if total_days > 0 else 0

    sharpe = np.sqrt(252) * ret_clean.mean() / ret_clean.std() if ret_clean.std() > 0 else 0

    # 最大回撤
    peak = df["equity"].cummax()
    drawdown = (df["equity"] - peak) / peak
    max_dd = drawdown.min()

    # 卡玛比率
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # 胜率
    trade_signals = df["cross"].dropna()
    trades = df[df["cross"] != 0].copy()
    if len(trades) >= 2:
        # 配对金叉和死叉, 计算每次交易的盈亏
        trade_returns = []
        for i in range(0, len(trades) - 1, 2):
            buy_idx = trades.index[i]
            sell_idx = trades.index[i + 1] if i + 1 < len(trades.index) else trades.index[-1]
            r = (df.loc[sell_idx, "price"] / df.loc[buy_idx, "price"] - 1) * df.loc[buy_idx, "signal"]
            trade_returns.append(r)
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
    else:
        win_rate = 0

    return {
        "total_return": total_return,
        "bench_return": bench_return,
        "annual_return": annual_return,
        "annual_bench": annual_bench,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "total_trades": len(trades) // 2,
    }


def generate_html_report(df, metrics, params, stock_name):
    """生成包含 Plotly 图表的 HTML 报告"""
    import json

    # 准备 JSON 数据
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    price = df["price"].round(2).tolist()
    ma_fast = df["ma_fast"].round(2).tolist()
    ma_slow = df["ma_slow"].round(2).tolist()
    equity = df["equity"].round(4).tolist()
    bench_equity = df["benchmark_equity"].round(4).tolist()

    # 标记金叉和死叉
    golden_idx = df[df["cross"] == 1].index.tolist()
    death_idx = df[df["cross"] == -1].index.tolist()

    golden_dates = [dates[i] for i in golden_idx if i < len(dates)]
    golden_prices = [price[i] for i in golden_idx if i < len(price)]
    death_dates = [dates[i] for i in death_idx if i < len(dates)]
    death_prices = [price[i] for i in death_idx if i < len(price)]

    fast_label = f"{params['ma_type'].upper()}{params['fast']}"
    slow_label = f"{params['ma_type'].upper()}{params['slow']}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>均线交叉策略回测 — {stock_name}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root {{
    --bg: #ffffff; --bg2: #f8f9fa; --text: #1a1a2e; --text2: #555;
    --border: #e0e0e0; --green: #1D9E75; --red: #D85A30; --purple: #534AB7;
    --radius: 10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--text); background: var(--bg); line-height: 1.7;
    max-width: 1040px; margin: 0 auto; padding: 32px 20px 60px;
  }}
  h1 {{ font-size: 24px; font-weight: 600; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 32px 0 12px; }}
  .subtitle {{ color: #888; font-size: 14px; margin: 4px 0 24px; }}
  .chart {{ width: 100%; height: 420px; border: 1px solid var(--border); border-radius: var(--radius); margin: 16px 0; }}
  .metrics {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0;
  }}
  .metric {{
    background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px; text-align: center;
  }}
  .metric .label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
  .metric .value {{ font-size: 22px; font-weight: 600; }}
  .metric .value.up {{ color: var(--red); }}
  .metric .value.down {{ color: var(--green); }}
  .metric .value.neutral {{ color: var(--text); }}
  .params {{
    display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0;
  }}
  .param {{
    background: #EEEDFE; border-radius: 16px; padding: 4px 14px;
    font-size: 13px; color: var(--purple); font-weight: 500;
  }}
  .footnote {{ margin-top: 32px; padding: 16px; background: #FFFDE7; border: 1px solid #FFE082; border-radius: 8px; font-size: 13px; color: #555; }}
  @media (max-width: 768px) {{ .metrics {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>

<h1>均线交叉策略回测报告</h1>
<p class="subtitle">
  {stock_name} · {fast_label} / {slow_label} 双均线 ·
  回测区间 {dates[0]} ~ {dates[-1]} · 共 {len(df)} 个交易日
</p>

<div class="params">
  <span class="param">快线 = {fast_label}</span>
  <span class="param">慢线 = {slow_label}</span>
  <span class="param">佣金 = {params['commission']:.4%}</span>
  <span class="param">滑点 = {params['slippage']:.2%}</span>
  <span class="param">总交易 = {metrics['total_trades']} 次</span>
</div>

<h2>绩效指标</h2>
<div class="metrics">
  <div class="metric">
    <div class="label">策略总收益</div>
    <div class="value {'up' if metrics['total_return'] > 0 else 'down'}">{metrics['total_return']:+.2%}</div>
  </div>
  <div class="metric">
    <div class="label">基准收益 (Buy & Hold)</div>
    <div class="value {'up' if metrics['bench_return'] > 0 else 'down'}">{metrics['bench_return']:+.2%}</div>
  </div>
  <div class="metric">
    <div class="label">年化收益率</div>
    <div class="value {'up' if metrics['annual_return'] > 0 else 'down'}">{metrics['annual_return']:+.2%}</div>
  </div>
  <div class="metric">
    <div class="label">年化夏普比率</div>
    <div class="value neutral">{metrics['sharpe']:.2f}</div>
  </div>
  <div class="metric">
    <div class="label">最大回撤</div>
    <div class="value down">{metrics['max_drawdown']:+.2%}</div>
  </div>
  <div class="metric">
    <div class="label">Calmar 比率</div>
    <div class="value neutral">{metrics['calmar']:.2f}</div>
  </div>
  <div class="metric">
    <div class="label">胜率</div>
    <div class="value neutral">{metrics['win_rate']:.1%}</div>
  </div>
  <div class="metric">
    <div class="label">基准年化</div>
    <div class="value {'up' if metrics['annual_bench'] > 0 else 'down'}">{metrics['annual_bench']:+.2%}</div>
  </div>
</div>

<h2>价格 & 均线 & 信号</h2>
<div class="chart" id="priceChart"></div>

<h2>资金曲线</h2>
<div class="chart" id="equityChart"></div>

<div class="footnote">
  <strong>策略逻辑：</strong>快线（{fast_label}）上穿慢线（{slow_label}）= 金叉买入；快线下穿慢线 = 死叉卖出。
  信号产生后<strong>次日</strong>以开盘价执行，避免未来函数。
  已计入单边佣金 {params['commission']:.3%} 和滑点 {params['slippage']:.2%}。
  <br><br>
  <strong>免责声明：</strong>本报告仅供学习和研究使用，不构成任何投资建议。过去表现不代表未来收益。
</div>

<script>
var dates = {json.dumps(dates)};
var price = {json.dumps(price)};
var maFast = {json.dumps(ma_fast)};
var maSlow = {json.dumps(ma_slow)};
var equity = {json.dumps(equity)};
var benchEq = {json.dumps(bench_equity)};
var goldenDates = {json.dumps(golden_dates)};
var goldenPrices = {json.dumps(golden_prices)};
var deathDates = {json.dumps(death_dates)};
var deathPrices = {json.dumps(death_prices)};

// ── 价格 + 均线 + 信号图表 ──
var priceChart = echarts.init(document.getElementById('priceChart'));
priceChart.setOption({{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ data: ['收盘价', '{fast_label}', '{slow_label}', '金叉', '死叉'], bottom: 0 }},
  grid: {{ left: 60, right: 40, top: 20, bottom: 40 }},
  xAxis: {{ type: 'category', data: dates, axisLabel: {{ formatter: v => v.slice(5) }} }},
  yAxis: {{ type: 'value', axisLabel: {{ formatter: v => '¥' + v.toFixed(0) }} }},
  dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', bottom: 4 }}],
  series: [
    {{
      name: '收盘价', type: 'line', data: price,
      lineStyle: {{ color: '#B4B2A9', width: 1 }}, symbol: 'none', z: 1
    }},
    {{
      name: '{fast_label}', type: 'line', data: maFast,
      lineStyle: {{ color: '#D85A30', width: 1.5 }}, symbol: 'none', z: 2
    }},
    {{
      name: '{slow_label}', type: 'line', data: maSlow,
      lineStyle: {{ color: '#534AB7', width: 1.5 }}, symbol: 'none', z: 2
    }},
    {{
      name: '金叉', type: 'scatter', data: goldenDates.map((d, i) => [d, goldenPrices[i]]),
      symbol: 'triangle', symbolSize: 14, itemStyle: {{ color: '#1D9E75' }}, z: 3
    }},
    {{
      name: '死叉', type: 'scatter', data: deathDates.map((d, i) => [d, deathPrices[i]]),
      symbol: 'triangle', symbolRotate: 180, symbolSize: 14, itemStyle: {{ color: '#D85A30' }}, z: 3
    }}
  ]
}});

// ── 资金曲线图表 ──
var equityChart = echarts.init(document.getElementById('equityChart'));
equityChart.setOption({{
  tooltip: {{ trigger: 'axis', formatter: p => p[0].axisValue + '<br/>' + p.map(x => x.marker + x.seriesName + ': ' + (x.value*100).toFixed(2) + '%').join('<br/>') }},
  legend: {{ data: ['策略净值', '基准净值 (Buy & Hold)'], bottom: 0 }},
  grid: {{ left: 60, right: 40, top: 20, bottom: 40 }},
  xAxis: {{ type: 'category', data: dates, axisLabel: {{ formatter: v => v.slice(5) }} }},
  yAxis: {{ type: 'value', axisLabel: {{ formatter: v => (v*100).toFixed(0) + '%' }} }},
  dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', bottom: 4 }}],
  series: [
    {{
      name: '策略净值', type: 'line', data: equity,
      lineStyle: {{ color: '#378ADD', width: 2 }},
      areaStyle: {{ color: 'rgba(55,138,221,0.08)' }}, symbol: 'none', z: 2
    }},
    {{
      name: '基准净值 (Buy & Hold)', type: 'line', data: benchEq,
      lineStyle: {{ color: '#B4B2A9', width: 1.5, type: 'dashed' }}, symbol: 'none', z: 1
    }}
  ]
}});

window.addEventListener('resize', () => {{ priceChart.resize(); equityChart.resize(); }});
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("  均线交叉策略回测系统")
    print("=" * 60)
    print()

    # ── 选择股票 ──
    print("可选股票:")
    for i, name in enumerate(STOCK_FILES, 1):
        print(f"  {i}. {name}")
    print()

    choice = input("请选择 (1/2/3, 默认 1 比亚迪): ").strip()
    if not choice:
        choice = "1"
    idx = int(choice) - 1
    stock_name = list(STOCK_FILES.keys())[idx]
    data_path = list(STOCK_FILES.values())[idx]

    # ── 选择参数 ──
    fast = input("快线周期 (默认 10): ").strip()
    fast = int(fast) if fast else 10
    slow = input("慢线周期 (默认 60): ").strip()
    slow = int(slow) if slow else 60
    ma_choice = input("均线类型 sma/ema (默认 ema): ").strip().lower()
    ma_type = ma_choice if ma_choice in ("sma", "ema") else "ema"

    if fast >= slow:
        print("错误: 快线周期必须小于慢线周期")
        sys.exit(1)

    params = {"fast": fast, "slow": slow, "ma_type": ma_type, "commission": 0.0005, "slippage": 0.001}

    print(f"\n加载 {stock_name} 数据...")
    df = load_data(data_path)
    print(f"  共 {len(df)} 条日线数据, 区间 {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")

    print(f"计算 {params['ma_type'].upper()}({fast}) / {params['ma_type'].upper()}({slow}) 信号...")
    df = calc_signals(df, fast=fast, slow=slow, ma_type=ma_type)

    signal_count = (df["cross"] == 1).sum()
    print(f"  金叉信号: {signal_count} 次, 死叉信号: {(df['cross'] == -1).sum()} 次")

    print("执行回测...")
    df = run_backtest(df, commission=params["commission"], slippage=params["slippage"])

    print("计算绩效指标...")
    metrics = compute_metrics(df)

    print("\n" + "─" * 40)
    print("  回测结果")
    print("─" * 40)
    print(f"  策略总收益:   {metrics['total_return']:+.2%}")
    print(f"  基准收益:     {metrics['bench_return']:+.2%}")
    print(f"  年化收益率:   {metrics['annual_return']:+.2%}")
    print(f"  年化夏普:     {metrics['sharpe']:.2f}")
    print(f"  最大回撤:     {metrics['max_drawdown']:+.2%}")
    print(f"  Calmar 比率:  {metrics['calmar']:.2f}")
    print(f"  胜率:         {metrics['win_rate']:.1%}")
    print(f"  交易次数:     {metrics['total_trades']}")
    print("─" * 40)

    # ── 生成报告 ──
    html = generate_html_report(df, metrics, params, stock_name)
    output_path = os.path.join(OUTPUT_DIR, "ma_crossover_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n报告已生成: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
