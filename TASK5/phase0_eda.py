# -*- coding: utf-8 -*-
"""
Phase 0 — 探索性数据分析 (EDA) v3
修复: 热力图数值显示、图表重叠、图幅过大等问题
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pointbiserialr
import json, os

DATA_FILE = "model_data_stock.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feat_cols = [c for c in feat_cols if c != 'Code']

print(f"加载: {df.shape[0]} 行 × {df.shape[1]} 列, {len(feat_cols)} 特征")

# ============================================================
# HTML 模板
# ============================================================
PAGE_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f6f8;}}
.header{{text-align:center;padding:16px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);position:sticky;top:0;z-index:10;}}
.header h1{{font-size:17px;color:#1a1a2e;margin-bottom:4px;}}
.header p{{font-size:11px;color:#888;}}
.grid{{display:grid;grid-template-columns:repeat({cols},1fr);gap:10px;padding:10px;max-width:1500px;margin:0 auto;}}
.card{{background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.08);padding:6px;}}
.card-title{{font-size:10px;color:#555;text-align:center;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.chart{{width:100%;height:{chart_height}px;}}
</style>
</head>
<body>
<div class="header"><h1>{title}</h1><p>{subtitle}</p></div>
<div class="grid">
"""

PAGE_FOOT = """</div>
<script>
(function(){{
var charts=[];
document.querySelectorAll('.chart').forEach(function(el){{
var c=echarts.init(el);
var opt=JSON.parse(el.getAttribute('data-option'));
c.setOption(opt);
charts.push(c);
el._chart=c;
}});
window.addEventListener('resize',function(){{charts.forEach(function(c){{c.resize();}});}});
}})();
</script>
</body>
</html>"""

# ============================================================
# 辅助: 生成 option JSON + 单独注入 JS 函数 formatter
# ============================================================
def raw_js_page(title, subtitle, js_code, height="calc(100vh - 80px)"):
    """页面: 在 setOption 中直接写 JS 函数（绕过 json.dumps 的限制）"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f6f8;}}
.header{{text-align:center;padding:16px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
.header h1{{font-size:17px;color:#1a1a2e;margin-bottom:4px;}}
.header p{{font-size:11px;color:#888;}}
#main{{width:100%;height:{height};}}
</style>
</head>
<body>
<div class="header"><h1>{title}</h1><p>{subtitle}</p></div>
<div id="main"></div>
<script>
var c=echarts.init(document.getElementById('main'));
{js_code}
window.addEventListener('resize',function(){{c.resize();}});
</script>
</body>
</html>"""

# ============================================================
# 0a — 描述性统计表
# ============================================================
print("\n[0a] 描述性统计表...")
stats_data = {}
for col in feat_cols:
    s = df[col].dropna()
    stats_data[col] = {
        'count': len(s), 'mean': round(s.mean(), 4), 'std': round(s.std(), 4),
        'min': round(s.min(), 4),
        'p1': round(s.quantile(0.01), 4), 'p5': round(s.quantile(0.05), 4),
        'p25': round(s.quantile(0.25), 4), 'p50': round(s.median(), 4),
        'p75': round(s.quantile(0.75), 4), 'p95': round(s.quantile(0.95), 4),
        'p99': round(s.quantile(0.99), 4), 'max': round(s.max(), 4),
        'skew': round(s.skew(), 4), 'kurtosis': round(s.kurtosis(), 4),
        'iqr': round(s.quantile(0.75) - s.quantile(0.25), 4),
        'cv': round(s.std() / s.mean(), 4) if s.mean() != 0 else None,
    }
pd.DataFrame(stats_data).T.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_stats.csv'), encoding='utf-8-sig')
print("  -> descriptive_stats.csv")

# ============================================================
# 0b — 目标变量 (修复: 缩小图幅)
# ============================================================
print("\n[0b] 目标变量...")
y_counts = df['Y'].value_counts().to_dict()
y_by_date = df.groupby('Date')['Y'].apply(lambda x: (x == True).sum() / len(x) * 100)

y_vals = [round(v, 2) for v in y_by_date.values]
y_min, y_max = min(y_vals), max(y_vals)
y_pad = max((y_max - y_min) * 0.15, 3)  # 15% padding

opt_y = {
    'tooltip': {'trigger': 'item'},
    'grid': [
        {'left': '3%', 'top': '5%', 'width': '42%', 'height': '45%'},
        {'left': '52%', 'top': '18%', 'width': '45%', 'height': '60%'}
    ],
    'xAxis': [
        {'show': False},
        {'type': 'category', 'gridIndex': 1, 'data': list(y_by_date.index),
         'axisLabel': {'rotate': 30, 'fontSize': 11, 'interval': 0}}
    ],
    'yAxis': [
        {'show': False},
        {'type': 'value', 'gridIndex': 1, 'name': '正例占比(%)',
         'min': round(y_min - y_pad, 1), 'max': round(y_max + y_pad, 1),
         'nameTextStyle': {'fontSize': 11}}
    ],
    'series': [
        {'type': 'pie', 'radius': ['30%', '55%'], 'center': ['23%', '35%'],
         'label': {'formatter': '{b}\n{d}%', 'fontSize': 12},
         'data': [
             {'value': y_counts.get(True, 0), 'name': 'True (正例)',
              'itemStyle': {'color': '#e74c3c'}},
             {'value': y_counts.get(False, 0), 'name': 'False (负例)',
              'itemStyle': {'color': '#2ecc71'}}
         ]},
        {'type': 'bar', 'xAxisIndex': 1, 'yAxisIndex': 1,
         'data': y_vals,
         'barWidth': '45%',
         'itemStyle': {'color': '#e74c3c', 'borderRadius': [4, 4, 0, 0]},
         'label': {'show': True, 'position': 'top', 'formatter': '{c}%', 'fontSize': 12},
         'markLine': {'silent': True, 'symbol': 'none',
             'data': [{'type': 'average', 'label': {'formatter': '均值 {c}%', 'fontSize': 10},
                       'lineStyle': {'color': '#f39c12', 'type': 'dashed', 'width': 2}}]}}
    ]
}
with open(os.path.join(OUTPUT_DIR, 'eda_target_distribution.html'), 'w', encoding='utf-8') as f:
    f.write(raw_js_page('目标变量 Y 分布分析', '左:总体分布 | 右:各截面正例占比趋势',
        f'c.setOption({json.dumps(opt_y, ensure_ascii=False, default=str)});', '700px'))
print("  -> eda_target_distribution.html")

# ============================================================
# 0c-1 — 直方图+KDE (修复: 中位数标注与标题重叠)
# ============================================================
print("\n[0c-1] 直方图+KDE...")

def make_hist_opt(col):
    s = df[col].dropna()
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s_c = s[(s >= lo) & (s <= hi)]
    mu, med = s_c.mean(), s_c.median()
    skew_v, kurt_v = round(s.skew(), 2), round(s.kurtosis(), 2)
    hist, bin_edges = np.histogram(s_c, bins=40)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        kde = stats.gaussian_kde(s_c)
        x_kde = np.linspace(lo, hi, 150)
        y_kde = kde(x_kde)
        y_kde = y_kde / y_kde.max() * max(hist) * 0.85
    except Exception:
        x_kde, y_kde = [], []

    return {
        'tooltip': {'trigger': 'axis'},
        'grid': {'left': '6%', 'right': '5%', 'top': '12%', 'bottom': '12%'},
        'xAxis': {'type': 'value', 'axisLabel': {'fontSize': 9, 'rotate': 15}},
        'yAxis': {'type': 'value', 'show': False},
        'series': [
            {'type': 'bar', 'barWidth': '90%',
             'data': [[round(float(x), 2), round(float(h), 2)] for x, h in zip(bc, hist)],
             'itemStyle': {'color': 'rgba(52,152,219,0.4)'}},
            {'type': 'line', 'symbol': 'none',
             'data': [[round(float(x), 2), round(float(y), 4)] for x, y in zip(x_kde, y_kde)],
             'lineStyle': {'color': '#e74c3c', 'width': 2.5}},
            {'type': 'line', 'data': [],  # 均值标线
             'markLine': {'silent': True, 'symbol': 'none',
                'data': [{'xAxis': round(float(mu), 2), 'label': {'formatter': f'μ={mu:.1f}',
                          'position': 'start', 'fontSize': 10, 'color': '#3498db'},
                          'lineStyle': {'color': '#3498db', 'type': 'dashed', 'width': 2}}]}},
            {'type': 'line', 'data': [],  # 中位数标线 — 放在下面避免与标题重叠
             'markLine': {'silent': True, 'symbol': 'none',
                'data': [{'xAxis': round(float(med), 2), 'label': {'formatter': f'M={med:.1f}',
                          'position': 'insideEndTop', 'fontSize': 10, 'color': '#e67e22'},
                          'lineStyle': {'color': '#e67e22', 'type': 'dashed', 'width': 2}}]}}
        ],
        'title': {'text': f'skew={skew_v}  kurt={kurt_v}',
                  'textStyle': {'fontSize': 10, 'color': '#999'}, 'left': 'center', 'top': 2}
    }

blocks_hist = [(col[:28], make_hist_opt(col)) for col in feat_cols]
with open(os.path.join(OUTPUT_DIR, 'eda_dist_histograms.html'), 'w', encoding='utf-8') as f:
    f.write(PAGE_HEAD.format(title='特征分布 — 直方图 + KDE 密度曲线',
            subtitle='蓝色直方图 | 红色KDE | 蓝色虚线=均值 | 橙色虚线=中位数',
            cols=4, chart_height=280) +
            "\n".join(f'<div class="card"><div class="card-title">{t}</div>'
                      f'<div class="chart" data-option=\'{json.dumps(o, ensure_ascii=False, default=str).replace(chr(39), "&#39;")}\'></div></div>'
                      for t, o in blocks_hist) + PAGE_FOOT)
print("  -> eda_dist_histograms.html")

# ============================================================
# 0c-2 — 跨期箱线图
# ============================================================
print("\n[0c-2] 跨期箱线图...")
dates = sorted(df['Date'].unique())

def make_box_opt(col):
    s = df[col]
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    box_data, med_data = [], []
    for d in dates:
        sd = df[df['Date'] == d][col].dropna().clip(lo, hi)
        if len(sd) > 5:
            q1, q2, q3 = sd.quantile([0.25, 0.5, 0.75])
            iqr_v = q3 - q1
            box_data.append([round(max(float(sd.min()), q1 - 1.5 * iqr_v), 3),
                             round(float(q1), 3), round(float(q2), 3),
                             round(float(q3), 3), round(min(float(sd.max()), q3 + 1.5 * iqr_v), 3)])
            med_data.append(round(float(q2), 3))
        else:
            box_data.append(None)
            med_data.append(None)
    valid_med = [(i, v) for i, v in enumerate(med_data) if v is not None]
    return {
        'tooltip': {'trigger': 'item'},
        'grid': {'left': '10%', 'right': '5%', 'top': '6%', 'bottom': '15%'},
        'xAxis': {'type': 'category', 'data': [str(d) for d in dates],
                  'axisLabel': {'fontSize': 7, 'rotate': 35, 'interval': 0}},
        'yAxis': {'type': 'value', 'axisLabel': {'fontSize': 7}},
        'series': [
            {'type': 'boxplot',
             'data': [{'value': bd, 'itemStyle': {'color': 'rgba(52,152,219,0.5)',
                      'borderColor': '#2c3e50'}} if bd else {'value': [0,0,0,0,0]}
                      for bd in box_data]},
            {'type': 'line', 'data': [[str(dates[i]), v] for i, v in valid_med],
             'lineStyle': {'color': '#e74c3c', 'width': 1.5, 'type': 'dashed'},
             'symbol': 'circle', 'symbolSize': 4, 'itemStyle': {'color': '#e74c3c'}}
        ]
    }

blocks_box = [(col[:28], make_box_opt(col)) for col in feat_cols]
with open(os.path.join(OUTPUT_DIR, 'eda_boxplot_by_date.html'), 'w', encoding='utf-8') as f:
    f.write(PAGE_HEAD.format(title='特征跨期分布 — 分组箱线图',
            subtitle='5个季度分组 | Y轴截断至P1-P99 | 红色虚线=中位数趋势',
            cols=4, chart_height=280) +
            "\n".join(f'<div class="card"><div class="card-title">{t}</div>'
                      f'<div class="chart" data-option=\'{json.dumps(o, ensure_ascii=False, default=str).replace(chr(39), "&#39;")}\'></div></div>'
                      for t, o in blocks_box) + PAGE_FOOT)
print("  -> eda_boxplot_by_date.html")

# ============================================================
# 0c-3 — 极端值热力图 (修复: 数值显示)
# ============================================================
print("\n[0c-3] 极端值热力图...")
heat_header = ['P0.1', 'P1', 'P25', 'P50', 'P75', 'P99', 'P99.9', '超P1-P99%']
heat_y, heat_vals = [], []
for col in feat_cols:
    s = df[col].dropna()
    p = np.percentile(s, [0.1, 1, 25, 50, 75, 99, 99.9])
    beyond = round(((s < p[1]) | (s > p[5])).sum() / len(s) * 100, 1)
    heat_y.append(col[:24])
    heat_vals.append([round(float(v), 2) for v in p] + [beyond])

# 直接用 JS 构造选项，label formatter 用函数
heat_data_json = json.dumps([[xi, yi, heat_vals[yi][xi]]
                              for yi in range(len(heat_y))
                              for xi in range(len(heat_header))],
                             ensure_ascii=False)
y_names_json = json.dumps(heat_y, ensure_ascii=False)
x_names_json = json.dumps(heat_header, ensure_ascii=False)
all_vals = [v for row in heat_vals for v in row[:-1]]  # 排除最后一列(百分比)
vmin, vmax = min(all_vals), max(all_vals)

heat_js = f"""c.setOption({{
    tooltip: {{trigger: 'item',
        formatter: function(p) {{ return p.name + ': ' + p.data[2]; }}
    }},
    grid: {{left: '26%', right: '8%', top: '5%', bottom: '12%'}},
    xAxis: {{type: 'category', data: {x_names_json}, position: 'top',
             axisLabel: {{fontSize: 10}}}},
    yAxis: {{type: 'category', data: {y_names_json},
             axisLabel: {{fontSize: 9, width: 180, overflow: 'truncate'}}}},
    visualMap: {{min: {vmin}, max: {vmax}, calculable: true,
                orient: 'horizontal', left: 'center', bottom: '0%',
                inRange: {{color: ['#2ecc71', '#f9f9f9', '#f39c12', '#e74c3c']}}}},
    series: [{{type: 'heatmap', data: {heat_data_json},
        label: {{show: true, fontSize: 8,
            formatter: function(p) {{ return String(p.data[2]); }}}},
        emphasis: {{itemStyle: {{shadowBlur: 10}}}}
    }}]
}});"""

with open(os.path.join(OUTPUT_DIR, 'eda_outlier_heatmap.html'), 'w', encoding='utf-8') as f:
    f.write(raw_js_page('极端值分析 — 分位数热力图',
            '颜色越深越极端 | 最后一列:超出P1-P99的样本占比(%)', heat_js, '750px'))
print("  -> eda_outlier_heatmap.html")

# ============================================================
# 0d-1 — 相关性热力图 (修复: 数值显示)
# ============================================================
print("\n[0d-1] 相关性热力图...")
pearson_mat = df[feat_cols].corr('pearson')
spearman_mat = df[feat_cols].corr('spearman')
short_names = [c[:15] for c in feat_cols]

# 数据格式: [x, y, spearman_value] (3元素, visualMap只能处理3元素)
# Pearson 值存储在独立查找表中供 label formatter 使用
corr_data = []
pearson_map = {}  # "x,y" -> pearson_value
for i in range(len(feat_cols)):
    for j in range(len(feat_cols)):
        if i > j:
            sp = round(spearman_mat.iloc[i, j], 4)
            ps = round(pearson_mat.iloc[i, j], 4)
            corr_data.append([j, i, sp])
            pearson_map[f"{j},{i}"] = ps

names_json = json.dumps(short_names, ensure_ascii=False)
data_json = json.dumps(corr_data, ensure_ascii=False)
pearson_json = json.dumps(pearson_map, ensure_ascii=False)

corr_js = f"""c.setOption({{
    tooltip: {{trigger: 'item',
        formatter: function(p) {{ return p.name + '<br/>Spearman: ' + p.data[2].toFixed(4) + '<br/>Pearson: ' + (PEARSON_MAP[p.data[0]+','+p.data[1]] || 0).toFixed(4); }}
    }},
    grid: {{left: '18%', right: '5%', top: '5%', bottom: '12%'}},
    xAxis: {{type: 'category', data: {names_json}, position: 'top',
             axisLabel: {{fontSize: 9, rotate: 45, interval: 0}}}},
    yAxis: {{type: 'category', data: {names_json},
             axisLabel: {{fontSize: 9, interval: 0}}}},
    visualMap: {{min: -1, max: 1, calculable: true, orient: 'horizontal',
                left: 'center', bottom: '0%',
                inRange: {{color: ['#2980b9', '#f5f5f5', '#c0392b']}}}},
    series: [{{type: 'heatmap', data: {data_json},
        label: {{show: true, fontSize: 7,
            formatter: function(p) {{
                var key = p.data[0] + ',' + p.data[1];
                var sp = p.data[2];
                var ps = PEARSON_MAP[key] || 0;
                return 'S:' + sp.toFixed(2) + '\\nP:' + ps.toFixed(2);
            }}}},
        emphasis: {{itemStyle: {{shadowBlur: 10}}}}
    }}]
}});"""
# 将 Pearson 查找表注入到 JS 中
corr_js = f"var PEARSON_MAP = {pearson_json};\n" + corr_js

with open(os.path.join(OUTPUT_DIR, 'eda_correlation_heatmap.html'), 'w', encoding='utf-8') as f:
    f.write(raw_js_page('相关性矩阵 — Spearman & Pearson',
            '颜色:Spearman秩相关 | 数值:上行Spearman, 下行Pearson | 仅显示下三角',
            corr_js, '850px'))
print("  -> eda_correlation_heatmap.html")

# ============================================================
# 0d-2 — 散点图矩阵 (修复: 相关系数标题与Y轴重叠)
# ============================================================
print("\n[0d-2] 散点图矩阵...")
high_corr_pairs = []
for i in range(len(feat_cols)):
    for j in range(i + 1, len(feat_cols)):
        sp = spearman_mat.iloc[i, j]
        if abs(sp) > 0.7:
            high_corr_pairs.append((feat_cols[i], feat_cols[j], sp, pearson_mat.iloc[i, j]))
high_corr_pairs.sort(key=lambda x: -abs(x[2]))

def make_scatter_opt(c1, c2, sp_val, ps_val):
    s1, s2 = df[c1], df[c2]
    x_lo, x_hi = s1.quantile(0.01), s1.quantile(0.99)
    y_lo, y_hi = s2.quantile(0.01), s2.quantile(0.99)
    mask_t, mask_f = df['Y'] == True, df['Y'] == False
    pts_t = [[round(float(x), 3), round(float(y), 3)] for x, y in
             zip(s1[mask_t], s2[mask_t]) if x_lo <= x <= x_hi and y_lo <= y <= y_hi]
    pts_f = [[round(float(x), 3), round(float(y), 3)] for x, y in
             zip(s1[mask_f], s2[mask_f]) if x_lo <= x <= x_hi and y_lo <= y <= y_hi]
    if len(pts_t) > 1500:
        np.random.seed(42)
        pts_t = [pts_t[i] for i in np.random.choice(len(pts_t), 1500, replace=False)]
    if len(pts_f) > 1500:
        np.random.seed(42)
        pts_f = [pts_f[i] for i in np.random.choice(len(pts_f), 1500, replace=False)]

    return {
        'tooltip': {'trigger': 'item'},
        'grid': {'left': '10%', 'right': '8%', 'top': '12%', 'bottom': '12%'},
        'xAxis': {'type': 'value', 'name': c1[:12], 'nameTextStyle': {'fontSize': 9},
                  'axisLabel': {'fontSize': 8}},
        'yAxis': {'type': 'value', 'name': c2[:12], 'nameTextStyle': {'fontSize': 9},
                  'axisLabel': {'fontSize': 8}},
        # 相关系数信息放在 tooltip 和注释，而非 title（避免与Y轴名重叠）
        'series': [
            {'type': 'scatter', 'data': pts_t, 'symbolSize': 3,
             'itemStyle': {'color': 'rgba(231,76,60,0.35)'}, 'name': 'Y=True'},
            {'type': 'scatter', 'data': pts_f, 'symbolSize': 2,
             'itemStyle': {'color': 'rgba(46,204,113,0.35)'}, 'name': 'Y=False'}
        ],
        'title': {'text': f'S={sp_val:.3f} P={ps_val:.3f}',
                  'textStyle': {'fontSize': 10, 'color': '#666'},
                  'left': 'right', 'top': 2, 'padding': [0, 8, 0, 0]}
    }

blocks_sc = [(f'{c1[:10]} vs {c2[:10]}', make_scatter_opt(c1, c2, sp_v, ps_v))
             for c1, c2, sp_v, ps_v in high_corr_pairs]
sc_cols = min(3, len(high_corr_pairs))
with open(os.path.join(OUTPUT_DIR, 'eda_scatter_pairs.html'), 'w', encoding='utf-8') as f:
    f.write(PAGE_HEAD.format(title='高相关特征对散点图矩阵',
            subtitle=f'{len(high_corr_pairs)} 对 Spearman |r|>0.7 | 红=Y:True | 绿=Y:False',
            cols=sc_cols, chart_height=380) +
            "\n".join(f'<div class="card"><div class="card-title">{t}</div>'
                      f'<div class="chart" data-option=\'{json.dumps(o, ensure_ascii=False, default=str).replace(chr(39), "&#39;")}\'></div></div>'
                      for t, o in blocks_sc) + PAGE_FOOT)
print(f"  -> eda_scatter_pairs.html ({len(high_corr_pairs)} 对)")

# ============================================================
# 0d-3 — VIF
# ============================================================
print("\n[0d-3] VIF...")
def calc_vif(X_arr):
    n, k = X_arr.shape
    v = {}
    for j in range(k):
        y = X_arr[:, j]
        Xo = np.delete(X_arr, j, axis=1)
        XtX = Xo.T @ Xo + np.eye(Xo.shape[1]) * 1e-8
        beta = np.linalg.solve(XtX, Xo.T @ y)
        yp = Xo @ beta
        r2 = 1 - np.sum((y - yp)**2) / np.sum((y - np.mean(y))**2)
        v[feat_cols[j]] = round(1/(1-r2) if r2 < 0.9999 else 100, 2)
    return v

X_raw = df[feat_cols].values.astype(np.float64)
vif_before = calc_vif(X_raw)
X_win = np.zeros_like(X_raw)
for j in range(X_raw.shape[1]):
    s = pd.Series(X_raw[:, j])
    X_win[:, j] = np.clip(X_raw[:, j], s.quantile(0.01), s.quantile(0.99))
vif_after = calc_vif(X_win)

vif_items = sorted(vif_after.items(), key=lambda x: -x[1])
vif_names = [f[0][:24] for f in vif_items]
vif_b = [round(vif_before[f[0]], 2) for f in vif_items]
vif_a = [round(f[1], 2) for f in vif_items]

opt_vif = {
    'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
    'legend': {'data': ['缩尾前 VIF', '缩尾后 VIF'], 'top': 5},
    'grid': {'left': '3%', 'right': '14%', 'top': '14%', 'bottom': '5%', 'containLabel': True},
    'xAxis': {'type': 'value', 'name': 'VIF 值'},
    'yAxis': {'type': 'category', 'data': vif_names, 'inverse': True, 'axisLabel': {'fontSize': 10}},
    'series': [
        {'name': '缩尾前 VIF', 'type': 'bar', 'data': vif_b,
         'itemStyle': {'color': 'rgba(149,165,166,0.85)'},
         'label': {'show': True, 'position': 'right', 'fontSize': 9, 'formatter': '{c}'}},
        {'name': '缩尾后 VIF', 'type': 'bar', 'data': vif_a,
         'itemStyle': {'color': 'rgba(52,152,219,0.85)'},
         'label': {'show': True, 'position': 'right', 'fontSize': 9, 'formatter': '{c}'},
         'markLine': {'silent': True, 'symbol': 'none',
             'data': [
                 {'xAxis': 5, 'label': {'formatter': 'VIF=5'}, 'lineStyle': {'color': '#f39c12', 'type': 'dashed', 'width': 2}},
                 {'xAxis': 10, 'label': {'formatter': 'VIF=10'}, 'lineStyle': {'color': '#e74c3c', 'type': 'dashed', 'width': 2}}
             ]}}
    ]
}
with open(os.path.join(OUTPUT_DIR, 'eda_vif_barchart.html'), 'w', encoding='utf-8') as f:
    f.write(raw_js_page('多重共线性诊断 — VIF 缩尾前后对比',
            '灰色:缩尾前(极端值掩盖) | 蓝色:缩尾后(共线性暴露) | 虚线:VIF=5/10',
            f'c.setOption({json.dumps(opt_vif, ensure_ascii=False, default=str)});', '700px'))
print("  -> eda_vif_barchart.html")

# ============================================================
# 0e-1 — 分组KDE
# ============================================================
print("\n[0e-1] 分组KDE...")

def make_kde_opt(col):
    s = df[col]
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    xr = np.linspace(lo, hi, 150)
    s_t = s[df['Y'] == True].dropna().clip(lo, hi)
    s_f = s[df['Y'] == False].dropna().clip(lo, hi)
    ks_v = round(stats.ks_2samp(s_t, s_f).statistic, 4) if len(s_t) > 10 and len(s_f) > 10 else 0
    ser = []
    if len(s_t) > 10:
        kde_t = stats.gaussian_kde(s_t)
        yt = kde_t(xr)
        yt = yt / yt.max() if yt.max() > 0 else yt
        ser.append({'type': 'line', 'symbol': 'none',
            'data': [[round(float(x), 3), round(float(y), 5)] for x, y in zip(xr, yt)],
            'lineStyle': {'color': '#e74c3c', 'width': 2},
            'areaStyle': {'color': 'rgba(231,76,60,0.1)'}, 'name': 'Y=True'})
    if len(s_f) > 10:
        kde_f = stats.gaussian_kde(s_f)
        yf = kde_f(xr)
        yf = yf / yf.max() if yf.max() > 0 else yf
        ser.append({'type': 'line', 'symbol': 'none',
            'data': [[round(float(x), 3), round(float(y), 5)] for x, y in zip(xr, yf)],
            'lineStyle': {'color': '#2ecc71', 'width': 2},
            'areaStyle': {'color': 'rgba(46,204,113,0.1)'}, 'name': 'Y=False'})
    return {
        'tooltip': {'trigger': 'axis'},
        'grid': {'left': '6%', 'right': '5%', 'top': '12%', 'bottom': '10%'},
        'xAxis': {'type': 'value', 'axisLabel': {'fontSize': 8, 'rotate': 15}},
        'yAxis': {'type': 'value', 'show': False},
        'title': {'text': f'KS={ks_v:.4f}', 'textStyle': {'fontSize': 9,
                  'color': '#e74c3c' if ks_v > 0.05 else '#999'}, 'left': 'center', 'top': 2},
        'series': ser
    }

blocks_kde = [(col[:28], make_kde_opt(col)) for col in feat_cols]
with open(os.path.join(OUTPUT_DIR, 'eda_feature_vs_target.html'), 'w', encoding='utf-8') as f:
    f.write(PAGE_HEAD.format(title='特征与目标关系 — 分组KDE密度曲线',
            subtitle='红=Y:True | 绿=Y:False | KS值越大区分力越强',
            cols=4, chart_height=280) +
            "\n".join(f'<div class="card"><div class="card-title">{t}</div>'
                      f'<div class="chart" data-option=\'{json.dumps(o, ensure_ascii=False, default=str).replace(chr(39), "&#39;")}\'></div></div>'
                      for t, o in blocks_kde) + PAGE_FOOT)
print("  -> eda_feature_vs_target.html")

# ============================================================
# 0e-2 — 点双列相关
# ============================================================
print("\n[0e-2] 点双列相关...")
pbc_results = []
y_n = df['Y'].astype(int).values
for col in feat_cols:
    s = df[col].dropna()
    mask = ~pd.isna(s)
    if mask.sum() > 10:
        r, p = pointbiserialr(y_n[mask], s[mask])
        pbc_results.append({'col': col, 'r': round(r, 5), 'p': round(p, 6), 'abs_r': abs(r)})
pbc_results.sort(key=lambda x: -x['abs_r'])

opt_pbc = {
    'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
    'grid': {'left': '3%', 'right': '12%', 'top': '5%', 'bottom': '5%', 'containLabel': True},
    'xAxis': {'type': 'value', 'name': '点双列相关系数 r'},
    'yAxis': {'type': 'category', 'data': [r['col'][:24] for r in pbc_results],
              'inverse': True, 'axisLabel': {'fontSize': 10}},
    'series': [{'type': 'bar',
        'data': [{'value': r['r'],
                  'itemStyle': {'color': '#e74c3c' if r['r'] > 0 else '#2ecc71'}}
                 for r in pbc_results],
        'label': {'show': True, 'position': 'right', 'fontSize': 9, 'formatter': '{c}'}
    }]
}
with open(os.path.join(OUTPUT_DIR, 'eda_feature_target_corr.html'), 'w', encoding='utf-8') as f:
    f.write(raw_js_page('特征与目标 Y 的相关系数排名',
            '点双列相关系数 | 红=正相关 | 绿=负相关',
            f'c.setOption({json.dumps(opt_pbc, ensure_ascii=False, default=str)});', '650px'))
print("  -> eda_feature_target_corr.html")

print(f"\n{'='*60}")
print("Phase 0 EDA v3 完成!")
print(f"输出: {os.path.abspath(OUTPUT_DIR)}/")
print(f"{'='*60}")
