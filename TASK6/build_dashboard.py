#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-compute backtest results for strategy presets and embed into HTML."""
import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

PRESETS = {
    '保守': {'T_buy': 0.65, 'T_sell': 0.50, 'alpha': 0.25, 'cap_stock': 0.03, 'N_max': 10, 'beta': 0.4},
    '中性': {'T_buy': 0.60, 'T_sell': 0.50, 'alpha': 0.50, 'cap_stock': 0.05, 'N_max': 20, 'beta': 0.6},
    '激进': {'T_buy': 0.55, 'T_sell': 0.45, 'alpha': 1.00, 'cap_stock': 0.08, 'N_max': 30, 'beta': 1.0},
}
MODELS = ['Logistic','RandomForest','XGBoost','LightGBM']
WINDOWS = ['W1','W2','W3','W4','W5']

def compute(predictions, strategy):
    """Compute backtest results for a given strategy."""
    results = []
    grouped = {}
    for d in predictions:
        key = d['window']+'|'+d['model']
        grouped.setdefault(key, []).append(d)

    for key, stocks in grouped.items():
        w, m = key.split('|')
        b = stocks[0]['b']
        candidates = [s for s in stocks if s['p_cal'] > strategy['T_buy']]
        candidates.sort(key=lambda x: x['p_cal'], reverse=True)
        candidates = candidates[:strategy['N_max']]

        if not candidates:
            results.append({'window':w,'model':m,'ret':0,'n':0,'avg_p':0})
            continue

        for c in candidates:
            c['f_kelly'] = max(0, (b*c['p_cal']-(1-c['p_cal']))/b)
            c['f_raw'] = min(c['f_kelly']*strategy['alpha'], strategy['cap_stock'])
        F_raw = sum(c['f_raw'] for c in candidates)
        scale = strategy['beta']/F_raw if F_raw > strategy['beta'] else 1
        port_ret = sum(c['f_raw']*scale*c['next_ret'] for c in candidates)
        avg_p = sum(c['p_cal'] for c in candidates)/len(candidates)
        results.append({'window':w,'model':m,'ret':round(port_ret,6),'n':len(candidates),'avg_p':round(avg_p,4)})
    return results

def summarize(all_results):
    """Summarize per-model metrics from window-level results."""
    by_model = {}
    by_window = {}
    for r in all_results:
        by_window.setdefault(r['window'], []).append(r)
        bm = by_model.setdefault(r['model'], {'rets':[], 'ns':[], 'cum':1, 'wins':0, 'maxdd':0})
        bm['rets'].append(r['ret'])
        bm['ns'].append(r['n'])
        bm['cum'] *= (1+r['ret'])
        if r['ret'] > 0: bm['wins'] += 1

    rows = []
    for m, d in by_model.items():
        cum_ret = d['cum']-1
        mean_ret = sum(d['rets'])/len(d['rets'])
        std_ret = (sum((r-mean_ret)**2 for r in d['rets'])/len(d['rets']))**0.5
        sharpe = mean_ret/std_ret*2 if std_ret>0 else 0
        cum_arr = [1]
        for r in d['rets']: cum_arr.append(cum_arr[-1]*(1+r))
        peak, maxdd = cum_arr[0], 0
        for v in cum_arr[1:]:
            if v>peak: peak=v
            dd = (v-peak)/peak
            if dd<maxdd: maxdd=dd
        avg_n = sum(d['ns'])/len(d['ns'])
        rows.append({
            'model':m,'cumRet':round(cum_ret,4),'meanRet':round(mean_ret,4),
            'sharpe':round(sharpe,2),'maxdd':round(abs(maxdd),4),
            'winRate':f"{d['wins']}/{len(d['rets'])}",'avgN':round(avg_n,0),
            'nav_series':[round(v,4) for v in cum_arr],
        })
    rows.sort(key=lambda x: x['sharpe'], reverse=True)
    return rows, by_window

def main():
    with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'r') as f:
        preds = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'model_info.json'), 'r', encoding='utf-8') as f:
        model_info = json.load(f)

    all_data = {}
    for pname, strategy in PRESETS.items():
        results = compute(preds, strategy)
        summary, by_window = summarize(results)
        all_data[pname] = {'summary': summary, 'by_window': by_window, 'strategy': strategy}

    # Generate HTML
    html = generate_html(all_data, model_info)
    out_path = os.path.join(OUTPUT_DIR, 'backtest_dashboard.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Dashboard saved: {out_path}")

def generate_html(all_data, model_info):
    js_data = json.dumps(all_data, ensure_ascii=False)
    js_model_info = json.dumps(model_info, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化选股 ML 回测看板</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6f8;color:#333}}
.header{{background:#fff;border-bottom:1px solid #e8e8e8;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
.header h1{{font-size:16px;font-weight:600;color:#1a1a2e}}
.header-right{{display:flex;gap:12px;align-items:center}}
.header select{{padding:6px 10px;border:1px solid #d9d9d9;border-radius:6px;font-size:13px;outline:none;background:#fff;cursor:pointer}}
.header select:focus{{border-color:#4a7dff}}
.header .preset-tabs{{display:flex;gap:0}}
.header .preset-tab{{padding:6px 16px;border:1px solid #d9d9d9;background:#fff;font-size:12px;cursor:pointer;color:#666;transition:all 0.2s}}
.header .preset-tab:first-child{{border-radius:6px 0 0 6px}}
.header .preset-tab:last-child{{border-radius:0 6px 6px 0}}
.header .preset-tab.active{{background:#eef2ff;border-color:#4a7dff;color:#4a7dff;font-weight:500}}
.container{{max-width:1200px;margin:0 auto;padding:16px 20px}}
.kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}}
.kpi-card{{background:#fff;border:1px solid #e8e8e8;border-radius:8px;padding:12px}}
.kpi-card .label{{font-size:11px;color:#999}}
.kpi-card .value{{font-size:22px;font-weight:700;color:#1a1a2e;margin-top:2px}}
.kpi-card .sub{{font-size:10px;color:#bbb}}

.panel{{background:#fff;border:1px solid #e8e8e8;border-radius:10px;padding:16px;margin-bottom:14px}}
.panel h2{{font-size:14px;font-weight:600;color:#1a1a2e;margin-bottom:12px}}
.chart{{width:100%;height:350px}}

table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#fafbfc;padding:8px 10px;border-bottom:2px solid #e8e8e8;text-align:left;color:#888;font-weight:500;white-space:nowrap}}
td{{padding:8px 10px;border-bottom:1px solid #f0f0f0;color:#555}}
tr:hover td{{background:#fafbfc}}
tr.best td{{background:#f0fbe8;font-weight:500}}

.coef-section{{margin-bottom:20px}}
.coef-section h3{{font-size:13px;font-weight:600;color:#333;margin-bottom:8px}}
.coef-bar{{display:flex;align-items:center;gap:8px;padding:2px 0;font-size:11px}}
.coef-bar .fname{{width:140px;text-align:right;color:#666;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.coef-bar .bar-wrap{{flex:1;height:14px;background:#f0f0f0;border-radius:3px;overflow:hidden}}
.coef-bar .bar-fill{{height:100%;border-radius:3px}}
.coef-bar .bar-val{{width:50px;font-size:11px;text-align:right;font-family:monospace}}

.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}.kpi-row{{grid-template-columns:repeat(3,1fr)}}}}
</style>
</head>
<body>
<div class="header">
<h1>量化选股 ML 回测看板</h1>
<div class="header-right">
<div class="preset-tabs" id="presetTabs"></div>
<select id="modelSelect"></select>
</div>
</div>
<div class="container">

<div class="panel" id="strategyPanel" style="padding:12px 16px;font-size:12px;color:#666"></div>
<div class="kpi-row" id="kpiRow"></div>

<div class="grid2">
<div class="panel"><h2>累计净值</h2><div class="chart" id="navChart"></div></div>
<div class="panel"><h2>季度收益</h2><div class="chart" id="perfChart"></div></div>
</div>

<div class="panel"><h2>模型对比</h2><div id="compTable"></div></div>

<div class="panel"><h2>模型详情 — <span id="detailModelName"></span></h2>
<div id="modelDetail"></div></div>
</div>

</div>

<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<script>
var ALL_DATA = {js_data};
var MODEL_INFO = {js_model_info};
var colors = {{'Logistic':'#7F77DD','RandomForest':'#378ADD','XGBoost':'#639922','LightGBM':'#1D9E75'}};
var WINDOWS = ['W1','W2','W3','W4','W5'];
var currentPreset = '{list(all_data.keys())[0]}';
var currentModel = 'Logistic';

function initUI(){{
  var presetTabs = document.getElementById('presetTabs');
  Object.keys(ALL_DATA).forEach(function(p){{
    var tab = document.createElement('div');
    tab.className = 'preset-tab' + (p===currentPreset?' active':'');
    tab.textContent = p;
    tab.onclick = function(){{ switchPreset(p); }};
    presetTabs.appendChild(tab);
  }});

  var sel = document.getElementById('modelSelect');
  ['Logistic','RandomForest','XGBoost','LightGBM'].forEach(function(m){{
    var o = document.createElement('option');
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  }});
  sel.value = currentModel;
  sel.onchange = function(){{ switchModel(this.value); }};
}}

function switchPreset(p){{
  currentPreset = p;
  document.querySelectorAll('.preset-tab').forEach(function(t,i,a){{ t.classList.toggle('active', t.textContent===p); }});
  updateAll();
}}

function switchModel(m){{
  currentModel = m;
  document.getElementById('modelSelect').value = m;
  updateAll();
}}

function updateAll(){{
  var data = ALL_DATA[currentPreset];
  var s = data.summary;
  var sel = s.find(function(r){{return r.model===currentModel;}}) || s[0];
  document.getElementById('strategyPanel').innerHTML = '策略: <b>'+currentPreset+'</b> | T_buy='+data.strategy.T_buy+' T_sell='+data.strategy.T_sell+' alpha='+data.strategy.alpha+' cap='+data.strategy.cap_stock+' N='+data.strategy.N_max+' beta='+data.strategy.beta;
  updateKPIs(sel);
  updateNavChart(s, sel);
  updatePerfChart(data.by_window);
  updateCompTable(s, sel);
  updateModelDetail(sel);
}}

function updateKPIs(sel){{
  document.getElementById('kpiRow').innerHTML = [
    {{l:'累计收益', v:(sel.cumRet*100).toFixed(2)+'%'}},
    {{l:'季度均值', v:(sel.meanRet*100).toFixed(2)+'%'}},
    {{l:'Sharpe', v:sel.sharpe.toFixed(2)}},
    {{l:'最大回撤', v:(sel.maxdd*100).toFixed(2)+'%'}},
    {{l:'胜率', v:sel.winRate}},
  ].map(function(k){{return '<div class="kpi-card"><div class="label">'+k.l+'</div><div class="value">'+k.v+'</div></div>';}}).join('');
}}

function updateNavChart(summary, sel){{
  var allVals = [];
  var series = summary.map(function(r){{
    var data = r.nav_series.map(function(v){{ return Math.round(v*10000); }});
    allVals = allVals.concat(data);
    return {{name:r.model, type:'line', data:data, smooth:true,
      lineStyle:{{color:colors[r.model],width:r.model===currentModel?3:1.5}},
      itemStyle:{{color:colors[r.model]}}, symbolSize:r.model===currentModel?7:4,
      symbol:r.model===currentModel?'circle':'none'}};
  }});
  var yMin = Math.floor(Math.min.apply(null, allVals)/200)*200;
  var yMax = Math.ceil(Math.max.apply(null, allVals)/200)*200;
  var chart = echarts.init(document.getElementById('navChart'));
  chart.setOption({{
    tooltip:{{trigger:'axis',valueFormatter:function(v){{return '¥'+v.toLocaleString();}}}},\n    legend:{{top:0,textStyle:{{fontSize:11}}}},
    xAxis:{{type:'category',data:['Start'].concat(WINDOWS)}},
    yAxis:{{type:'value',name:'净值',min:yMin,max:yMax,axisLabel:{{formatter:function(v){{return '¥'+(v/1000).toFixed(0)+'k';}}}}}},
    series:series, grid:{{left:70,right:20,top:35,bottom:35}}
  }});
}}

function updatePerfChart(byWindow){{
  var series = ['Logistic','RandomForest','XGBoost','LightGBM'].map(function(m){{
    var data = WINDOWS.map(function(w){{
      var wr = byWindow[w] || [];
      var r = wr.filter(function(d){{return d.model===m;}});
      return r.length>0 ? parseFloat((r[0].ret*100).toFixed(2)) : 0;
    }});
    return {{name:m, type:'bar', data:data, itemStyle:{{color:colors[m]}}, barGap:'20%'}};
  }});
  var chart = echarts.init(document.getElementById('perfChart'));
  chart.setOption({{
    tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}},valueFormatter:function(v){{return v+'%'}}}},
    legend:{{top:0,textStyle:{{fontSize:11}}}},
    xAxis:{{type:'category',data:WINDOWS}},
    yAxis:{{type:'value',name:'季度收益 (%)'}},
    series:series, grid:{{left:55,right:20,top:35,bottom:35}}
  }});
}}

function updateCompTable(summary, sel){{
  var html = '<table><thead><tr><th></th><th>累计收益</th><th>均值/季</th><th>Sharpe</th><th>最大回撤</th><th>胜率</th><th>平均持仓</th><th>解读</th></tr></thead><tbody>';
  var comments = {{
    'Logistic':'线性模型最稳，系数可解释，胜率高回撤低，适合保守策略。简单因子线性组合即有不错表现。',
    'RandomForest':'Bagging集成，对异常值鲁棒。GARP信号贡献最高，但波动较大，部分窗口过度集中。',
    'XGBoost':'梯度提升主力，整体均衡。GARP与质量因子并重，但概率偏保守导致某些窗口选股不足。',
    'LightGBM':'速度最快但回撤偏大。极度依赖MV和成长因子，可能过度暴露于小盘成长风格。',
  }};
  summary.forEach(function(r){{
    var isSel = r.model === sel.model;
    html += '<tr'+(isSel?' class="best"':'')+'><td style="font-weight:500">'+(isSel?'* ':'')+r.model+'</td>'+
      '<td>'+(r.cumRet*100).toFixed(2)+'%</td><td>'+(r.meanRet*100).toFixed(2)+'%</td>'+
      '<td>'+r.sharpe.toFixed(2)+'</td><td>'+(r.maxdd*100).toFixed(2)+'%</td>'+
      '<td>'+r.winRate+'</td><td>'+r.avgN+'</td>'+
      '<td style="font-size:11px;color:#888;max-width:280px">'+(comments[r.model]||'')+'</td></tr>';
  }});
  html += '</tbody></table>';
  document.getElementById('compTable').innerHTML = html;
}}

function renderCoefBars(containerId, data, type){{
  if(!data || data.length===0){{ document.getElementById(containerId).innerHTML='<p style="color:#999;font-size:11px">无数据</p>'; return; }}
  var maxV = Math.max.apply(null, data.map(function(d){{ return type==='coef'?Math.abs(d.coef):d.importance; }}));
  var top = data.slice(0,12);
  var html = '';
  top.forEach(function(d){{
    var v = type==='coef'?d.coef:d.importance;
    var pct = maxV>0?(Math.abs(v)/maxV*100):0;
    var c = type==='coef'?(v>=0?'#639922':'#e24b4a'):'#4a7dff';
    html += '<div class="coef-bar"><span class="fname" title="'+d.feature+'">'+d.feature+'</span>'+
      '<div class="bar-wrap"><div class="bar-fill" style="width:'+pct+'%;background:'+c+'"></div></div>'+
      '<span class="bar-val" style="color:'+c+'">'+v.toFixed(4)+'</span></div>';
  }});
  document.getElementById(containerId).innerHTML = html;
}}

function updateModelDetail(sel){{
  var m = sel.model;
  document.getElementById('detailModelName').textContent = m;
  var info = MODEL_INFO.models[m];
  if(!info){{ document.getElementById('modelDetail').innerHTML='<p style="color:#999">无详情</p>'; return; }}
  var title = info.type==='coefficients'?'回归系数':'特征重要性';
  var html = '<div class="coef-section"><h3>'+title+' Top 12</h3><div id="coefBars"></div></div>';
  document.getElementById('modelDetail').innerHTML = html;
  renderCoefBars('coefBars', info.data, info.type==='coefficients'?'coef':'imp');
  document.getElementById('modelDetail').classList.add('show');
}}

initUI();
updateAll();

window.addEventListener('resize', function(){{
  ['navChart','perfChart'].forEach(function(id){{
    var inst = echarts.getInstanceByDom(document.getElementById(id));
    if(inst) inst.resize();
  }});
}});
</script>
</body>
</html>'''

if __name__ == '__main__':
    main()
