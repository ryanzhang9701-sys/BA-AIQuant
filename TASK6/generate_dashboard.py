#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate HTML dashboard from backtest results."""

import json, os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def generate():
    df_results = pd.read_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'))
    df_summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
    df_metrics = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'))
    with open(os.path.join(MODEL_DIR, 'best_model_meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Build chart data
    windows = sorted(df_results['Window'].unique())
    models = sorted(df_results['Model'].unique())
    colors = {'Logistic': '#7F77DD', 'RandomForest': '#378ADD', 'XGBoost': '#639922', 'LightGBM': '#1D9E75'}

    # Cumulative return series
    cum_ret_series = {}
    for m in models:
        mdata = df_results[df_results['Model'] == m].sort_values('Window')
        cum = (1 + mdata['Portfolio_Return']).cumprod().values.tolist()
        cum_ret_series[m] = [1.0] + cum

    # Model comparison table rows
    comp_rows = ''
    for _, row in df_summary.iterrows():
        comp_rows += f'''
        <tr>
            <td style="font-weight:500;">{row['Model']}</td>
            <td>{row['Cumulative_Return']:.4f}</td>
            <td>{row['Mean_Quarterly_Return']:.4f}</td>
            <td>{row['Sharpe']:.2f}</td>
            <td>{row['MaxDD']:.2%}</td>
            <td>{row['Calmar']:.2f}</td>
            <td>{row['Win_Rate']:.0%}</td>
            <td>{row['Avg_Holdings']:.0f}</td>
        </tr>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化选股 ML 回测看板</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6f8;color:#333}}
.header{{background:#fff;border-bottom:1px solid #e8e8e8;padding:16px 24px;display:flex;align-items:center;justify-content:space-between}}
.header h1{{font-size:18px;font-weight:600;color:#1a1a2e}}
.best-badge{{background:#eaf3de;color:#3b6d11;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:500}}
.container{{max-width:1200px;margin:0 auto;padding:20px 24px}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
.kpi-card{{background:#fff;border:1px solid #e8e8e8;border-radius:10px;padding:16px}}
.kpi-card .label{{font-size:12px;color:#999}}
.kpi-card .value{{font-size:24px;font-weight:700;color:#1a1a2e;margin-top:4px}}
.kpi-card .sub{{font-size:11px;color:#bbb;margin-top:2px}}
.panel{{background:#fff;border:1px solid #e8e8e8;border-radius:10px;padding:20px;margin-bottom:16px}}
.panel h2{{font-size:15px;font-weight:600;color:#1a1a2e;margin-bottom:16px}}
.chart-box{{width:100%;height:380px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#fafbfc;padding:10px 12px;border-bottom:2px solid #e8e8e8;text-align:left;color:#888;font-weight:500}}
td{{padding:10px 12px;border-bottom:1px solid #f0f0f0;color:#555}}
tr:hover td{{background:#fafbfc}}
.footer{{text-align:center;color:#bbb;font-size:11px;padding:20px}}
.strategy-tag{{display:inline-block;background:#eef2ff;color:#4a7dff;padding:2px 8px;border-radius:8px;font-size:11px;margin:2px 4px}}
</style>
</head>
<body>
<div class="header">
<h1>量化选股 ML 回测看板</h1>
<span class="best-badge">最佳模型: {meta['best_model']} (Sharpe {meta['best_sharpe']:.2f})</span>
</div>
<div class="container">

<div class="kpi-row">
<div class="kpi-card"><div class="label">最佳模型</div><div class="value">{meta['best_model']}</div><div class="sub">Sharpe {meta['best_sharpe']:.2f}</div></div>
<div class="kpi-card"><div class="label">累计收益</div><div class="value" style="color:#639922">{meta['best_cumret']:.2%}</div><div class="sub">5期Walk-Forward</div></div>
<div class="kpi-card"><div class="label">回测窗口</div><div class="value">5</div><div class="sub">2021Q2 ~ 2022Q2</div></div>
<div class="kpi-card"><div class="label">对比模型</div><div class="value">4</div><div class="sub">LR / RF / XGB / LGB</div></div>
</div>

<div class="panel">
<h2>策略参数</h2>
<div style="font-size:13px;">
{''.join(f'<span class="strategy-tag">{k}: <b>{v}</b></span>' for k, v in meta['strategy_params'].items())}
</div>
</div>

<div class="panel">
<h2>净值曲线</h2>
<div class="chart-box" id="navChart"></div>
</div>

<div class="panel">
<h2>模型收益对比</h2>
<table>
<thead><tr>
<th>模型</th><th>累计收益</th><th>季度均值</th><th>Sharpe</th><th>最大回撤</th><th>Calmar</th><th>胜率</th><th>平均持仓</th>
</tr></thead>
<tbody>{comp_rows}</tbody>
</table>
</div>

<div class="panel">
<h2>各窗口收益分布</h2>
<div class="chart-box" id="perfChart"></div>
</div>

<div class="panel">
<h2>模型AUC对比 (验证集)</h2>
<div class="chart-box" id="aucChart"></div>
</div>

<div class="footer">量化选股 ML 建模 · Walk-Forward Backtest · {meta['best_model']}</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<script>
var dfResults = {json.dumps(df_results.to_dict('records'), ensure_ascii=False)};
var dfMetrics = {json.dumps(df_metrics.to_dict('records'), ensure_ascii=False)};
var windows = {json.dumps(list(windows))};
var models = {json.dumps(list(models))};
var colors = {json.dumps(colors)};

// Nav chart
(function(){{
var chart = echarts.init(document.getElementById('navChart'));
var series = models.map(function(m) {{
var vals = [];
var cum = 1;
vals.push(1);
for (var i = 0; i < windows.length; i++) {{
var r = dfResults.filter(function(d) {{ return d.Model === m && d.Window === windows[i]; }});
if (r.length > 0) cum *= (1 + r[0].Portfolio_Return);
vals.push(cum);
}}
return {{ name: m, type: 'line', data: vals, smooth: true,
  lineStyle: {{ color: colors[m], width: m === '{meta['best_model']}' ? 3 : 1.5 }},
  itemStyle: {{ color: colors[m] }}, symbol: 'circle', symbolSize: 6 }};
}});
chart.setOption({{
tooltip: {{ trigger: 'axis' }},
xAxis: {{ type: 'category', data: ['Start'].concat(windows) }},
yAxis: {{ type: 'value', name: '净值', axisLabel: {{ formatter: function(v){{return v.toFixed(2)}} }} }},
series: series,
grid: {{ left: 60, right: 30, top: 20, bottom: 40 }}
}});
}})();

// Performance chart
(function(){{
var chart = echarts.init(document.getElementById('perfChart'));
var series = models.map(function(m) {{
var data = windows.map(function(w) {{
var r = dfResults.filter(function(d) {{ return d.Model === m && d.Window === w; }});
return r.length > 0 ? (r[0].Portfolio_Return * 100).toFixed(2) : 0;
}});
return {{ name: m, type: 'bar', data: data,
  itemStyle: {{ color: colors[m] }},
  barGap: '20%' }};
}});
chart.setOption({{
tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }}, valueFormatter: function(v){{return v + '%'}} }},
xAxis: {{ type: 'category', data: windows }},
yAxis: {{ type: 'value', name: '季度收益 (%)' }},
series: series,
legend: {{ top: 0 }},
grid: {{ left: 60, right: 20, top: 40, bottom: 40 }}
}});
}})();

// AUC chart
(function(){{
var chart = echarts.init(document.getElementById('aucChart'));
var series = models.map(function(m) {{
var data = windows.map(function(w) {{
var r = dfMetrics.filter(function(d) {{ return d.Model === m && d.Window === w; }});
return r.length > 0 ? r[0].AUC_Val.toFixed(4) : null;
}});
return {{ name: m, type: 'line', data: data,
  lineStyle: {{ color: colors[m] }},
  itemStyle: {{ color: colors[m] }},
  symbol: 'circle', symbolSize: 7,
  markLine: {{ data: [{{ type: 'average', name: '均值' }}], silent: true,
    lineStyle: {{ type: 'dashed', color: '#999' }},
    label: {{ formatter: 'avg: {{c}}' }} }}
}};
}});
chart.setOption({{
tooltip: {{ trigger: 'axis' }},
xAxis: {{ type: 'category', data: windows }},
yAxis: {{ type: 'value', name: 'AUC-ROC', min: 0.45, max: 0.65 }},
series: series,
legend: {{ top: 0 }},
grid: {{ left: 60, right: 30, top: 40, bottom: 40 }}
}});
}})();
</script>
</body>
</html>'''

    out_path = os.path.join(OUTPUT_DIR, 'backtest_dashboard.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Dashboard saved to {out_path}")
    return out_path

if __name__ == '__main__':
    generate()
