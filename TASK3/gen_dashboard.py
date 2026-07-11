"""Generate MA Crossover Dashboard v3 — split charts, cost toggles, stock names, auto MA axis"""
import os, json, pandas as pd, numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, 'data')
OUT  = os.path.join(PROJ, 'TASK3')
os.makedirs(OUT, exist_ok=True)

# Stock short name map (hardcoded to avoid encoding issues)
NAME_MAP = {
    '002594.SZ': 'BYD',
    '002594.HK': 'BYD(HK)',
    '603986.SH': 'GigaDevice',
    '688981.SH': 'SMIC',
    '688981.HK': 'SMIC(HK)',
}

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
        stocks.append(dict(id=sid, label=f'{sid} {NAME_MAP.get(sid, "")}',
            market='A', data=df[['trade_date','close_qfq']].copy(), price_col='close_qfq'))

    if os.path.exists(csv_hk):
        df = pd.read_csv(csv_hk, encoding='utf-8-sig', parse_dates=['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        sid = code.split('.')[0] + '.HK'
        stocks.append(dict(id=sid, label=f'{sid} {NAME_MAP.get(sid, "")}',
            market='HK', data=df[['trade_date','close']].copy(), price_col='close'))

print(f'Loaded {len(stocks)} stocks')
for s in stocks:
    print(f'  {s["label"]}: {len(s["data"])} rows')

# Build data JSON
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
# HTML
# ============================================================
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MA Crossover Dashboard v3</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root{{--bg:#fff;--bg2:#f8f9fa;--t:#1a1a2e;--t2:#555;--bd:#e0e0e0;--gn:#1D9E75;--rd:#D85A30;--pu:#534AB7;--bl:#378ADD;--ra:10px}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:var(--t);background:#f5f5f5;line-height:1.7}}
.hd{{background:#fff;border-bottom:1px solid var(--bd);padding:12px 24px}}
.hd h1{{font-size:18px;font-weight:600;margin-bottom:6px}}
.ctrls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:12px}}
.ctrls select{{padding:4px 10px;border:1px solid var(--bd);border-radius:6px;font-size:12px;background:#fff;min-width:140px}}
.ctrls input[type=number]{{padding:4px 6px;border:1px solid var(--bd);border-radius:6px;font-size:12px;width:54px;text-align:center;background:#fff}}
.ctrls label{{color:#888;font-size:11px;white-space:nowrap}}
.ctrls .sep{{width:1px;height:20px;background:var(--bd);margin:0 4px}}
.presets{{display:flex;gap:4px}}
.presets button{{padding:4px 10px;border:1px solid var(--bd);border-radius:14px;font-size:11px;cursor:pointer;background:#fff;transition:all .12s;white-space:nowrap}}
.presets button:hover{{border-color:var(--pu);color:var(--pu)}}
.presets button.active{{background:var(--pu);color:#fff;border-color:var(--pu)}}
.toggle-row{{display:flex;gap:16px;align-items:center;font-size:12px}}
.toggle-row label{{display:flex;align-items:center;gap:4px;cursor:pointer;color:var(--t2)}}
.toggle-row input[type=checkbox]{{accent-color:var(--pu)}}
.btn-go{{padding:4px 18px;background:var(--pu);color:#fff;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500}}
.btn-go:hover{{opacity:.85}}
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
  <h1>MA Crossover Live Backtest</h1>
  <div class="ctrls">
    <label>Stock</label>
    <select id="stockSelect" onchange="runBacktest()">{stock_options}</select>
    <span class="sep"></span>
    <label>MA</label>
    <select id="maType" onchange="runBacktest()">
      <option value="ema" selected>EMA</option><option value="sma">SMA</option>
    </select>
    <label>Fast</label>
    <input type="number" id="fastP" value="10" min="3" max="50" onchange="validateParams()">
    <label>Slow</label>
    <input type="number" id="slowP" value="60" min="10" max="300" onchange="validateParams()">
    <div class="presets" id="pGroup">
      <button onclick="setPreset(5,20)" data-p="agg">5/20</button>
      <button class="active" onclick="setPreset(10,60)" data-p="bal">10/60</button>
      <button onclick="setPreset(20,120)" data-p="con">20/120</button>
    </div>
    <span class="sep"></span>
    <div class="toggle-row">
      <label><input type="checkbox" id="chkCommission" checked onchange="runBacktest()">Commission</label>
      <label><input type="checkbox" id="chkStamp" checked onchange="runBacktest()">Stamp Tax</label>
      <label><input type="checkbox" id="chkSlippage" checked onchange="runBacktest()">Slippage</label>
    </div>
    <button class="btn-go" onclick="runBacktest()">Run</button>
  </div>
</div>

<div class="container">
  <div class="metrics" id="metrics"></div>

  <div class="grid-2">
    <div class="card"><h3>Price &amp; Signals</h3><div class="chart-sm" id="pC"></div></div>
    <div class="card"><h3>MA Comparison (auto Y range)</h3><div class="chart-sm" id="maC"></div></div>
  </div>

  <div class="grid-2">
    <div class="card"><h3>Equity Curve</h3><div class="chart-sm" id="eC"></div></div>
    <div class="card"><h3>Drawdown</h3><div class="chart-sm" id="dC"></div></div>
  </div>

  <div class="card"><h3>Trade Log</h3><div style="overflow-x:auto"><table id="tradeTable"><tr><th colspan="4">Click Run</th></tr></table></div></div>
</div>

<div class="foot">Based on ma_crossover_strategy_spec.md | T+1 execution | Past != Future</div>

<script>
var DATA = {DATA_JSON};
var DEFAULT = "{default_id}";
var charts = {{}};

function initChart(id) {{
  charts[id] = echarts.init(document.getElementById(id));
}}
["pC","maC","eC","dC"].forEach(initChart);
window.addEventListener("resize", function(){{
  for (var k in charts) charts[k].resize();
}});

function calcEMA(v, n) {{
  var a = 2/(n+1), r = [v[0]], e = v[0];
  for (var i = 1; i < v.length; i++) {{ e = a*v[i] + (1-a)*e; r.push(e); }}
  return r;
}}
function calcSMA(v, n) {{
  var r = [], s = 0;
  for (var i = 0; i < v.length; i++) {{
    s += v[i]; if (i >= n) s -= v[i-n];
    r.push(i >= n-1 ? s/n : null);
  }}
  return r;
}}
function calcMA(v, n, t) {{ return t === "ema" ? calcEMA(v, n) : calcSMA(v, n); }}

function runBacktest() {{
  var sid = document.getElementById("stockSelect").value;
  var mt = document.getElementById("maType").value;
  var fp = parseInt(document.getElementById("fastP").value) || 10;
  var sp = parseInt(document.getElementById("slowP").value) || 60;
  if (fp >= sp) {{ alert("Fast < Slow required!"); return; }}

  var useComm = document.getElementById("chkCommission").checked;
  var useStamp = document.getElementById("chkStamp").checked;
  var useSlip = document.getElementById("chkSlippage").checked;
  var COMM = useComm ? 0.0003 : 0;
  var STAMP = useStamp ? 0.0005 : 0;
  var SLIP = useSlip ? 0.001 : 0;

  var sd = DATA[sid];
  var dates = sd.dates, price = sd.price, N = dates.length;
  var maF = calcMA(price, fp, mt), maS = calcMA(price, sp, mt);

  var w = Math.max(fp, sp);
  var signal = new Array(N).fill(0), cross = new Array(N).fill(0), pos = new Array(N).fill(0);
  for (var i = w; i < N; i++) {{
    if (maF[i] !== null && maS[i] !== null) {{
      signal[i] = maF[i] > maS[i] ? 1 : 0;
      if (i > w) cross[i] = signal[i] - signal[i-1];
      pos[i] = i > w ? signal[i-1] : 0;
    }}
  }}

  var equity = new Array(N).fill(1), bench = new Array(N).fill(1);
  var dd = new Array(N).fill(0), bdd = new Array(N).fill(0);
  var peak = 1, bpeak = 1;
  for (var i = w+1; i < N; i++) {{
    var pr = price[i]/price[i-1] - 1;
    bench[i] = bench[i-1] * (1+pr);
    var act = Math.abs(pos[i]-pos[i-1]);
    var cost = act*(COMM+SLIP);
    if (pos[i] < pos[i-1]) cost += STAMP;
    var sr = pos[i]*pr - cost;
    equity[i] = equity[i-1] * (1+sr);
    if (equity[i] > peak) peak = equity[i];
    dd[i] = (equity[i]-peak)/peak;
    if (bench[i] > bpeak) bpeak = bench[i];
    bdd[i] = (bench[i]-bpeak)/bpeak;
  }}

  var Nv = N - w - 1;
  var tRet = equity[N-1]-1, bRet = bench[N-1]-1;
  var aRet = Nv>0 ? Math.pow(1+tRet, 252/Nv)-1 : 0;
  var aBen = Nv>0 ? Math.pow(1+bRet, 252/Nv)-1 : 0;

  var rets = [];
  for (var i = w+1; i < N; i++) {{
    var pr = price[i]/price[i-1]-1;
    var act = Math.abs(pos[i]-pos[i-1]);
    var cost = act*(COMM+SLIP);
    if (pos[i] < pos[i-1]) cost += STAMP;
    rets.push(pos[i]*pr - cost);
  }}
  var mn = rets.reduce(function(a,b){{return a+b}},0)/rets.length;
  var vr = rets.reduce(function(a,b){{return a+(b-mn)*(b-mn)}},0)/rets.length;
  var sharpe = vr>0 ? Math.sqrt(252)*mn/Math.sqrt(vr) : 0;

  var maxDd = Math.min.apply(null, dd.slice(w));
  var calmar = maxDd!==0 ? aRet/Math.abs(maxDd) : 0;

  // Trades
  var trades = [];
  for (var i = w+1; i < N; i++) {{
    if (cross[i] === 1) {{
      var exP = price[N-1];
      for (var j = i+1; j < N; j++) {{
        if (cross[j] === 1) {{ exP = price[j-1]; break; }}
      }}
      trades.push({{date:dates[i], entry:price[i], exit:exP, ret:exP/price[i]-1}});
    }}
  }}
  var wins = trades.filter(function(t){{return t.ret>0}});
  var wr = trades.length>0 ? wins.length/trades.length : 0;

  // Metrics
  document.getElementById("metrics").innerHTML =
    '<div class="mt"><div class="lbl">Total Return</div><div class="val '+(tRet>0?"up":"down")+'">'+(tRet*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Benchmark</div><div class="val '+(bRet>0?"up":"down")+'">'+(bRet*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Annual</div><div class="val neutral">'+(aRet*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Sharpe</div><div class="val neutral">'+sharpe.toFixed(2)+'</div></div>'+
    '<div class="mt"><div class="lbl">Max DD</div><div class="val down">'+(maxDd*100).toFixed(2)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Calmar</div><div class="val neutral">'+calmar.toFixed(2)+'</div></div>'+
    '<div class="mt"><div class="lbl">Win Rate</div><div class="val neutral">'+(wr*100).toFixed(1)+'%</div></div>'+
    '<div class="mt"><div class="lbl">Trades</div><div class="val neutral">'+trades.length+'</div></div>';

  // Trade log
  document.getElementById("tradeTable").innerHTML='<tr><th>Date</th><th>Entry</th><th>Result</th><th>Return</th></tr>'+
    trades.map(function(t){{var wl=t.ret>0?"win":"loss";return'<tr><td>'+t.date+'</td><td>'+t.entry.toFixed(2)+'</td><td><span class="badge '+wl+'">'+(t.ret>0?'Win':'Loss')+'</span></td><td class="'+(t.ret>0?"up":"down")+'">'+(t.ret*100).toFixed(2)+'%</td></tr>';}}).join('');

  // Golden/Death markers
  var gScat=[], dScat=[];
  for (var i = w+1; i < N; i++) {{
    if (cross[i]===1) gScat.push([dates[i], price[i]]);
    if (cross[i]===-1) dScat.push([dates[i], price[i]]);
  }}

  var ft = mt.toUpperCase(), fl = ft+"("+fp+")", sl = ft+"("+sp+")";

  // Compute MA Y range (auto-fit to MA lines)
  var maMin = Infinity, maMax = -Infinity;
  for (var i = w; i < N; i++) {{
    if (maF[i] !== null) {{ if (maF[i] < maMin) maMin = maF[i]; if (maF[i] > maMax) maMax = maF[i]; }}
    if (maS[i] !== null) {{ if (maS[i] < maMin) maMin = maS[i]; if (maS[i] > maMax) maMax = maS[i]; }}
  }}
  var pad = (maMax - maMin) * 0.08;
  maMin -= pad; maMax += pad;

  // Chart 1: Price + Signals only
  charts.pC.setOption({{
    tooltip:{{trigger:"axis"}},
    legend:{{data:["Close","Golden","Death"],top:2}},
    grid:{{left:60,right:20,top:38,bottom:50}},
    xAxis:{{type:"category",data:dates,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},
    yAxis:{{type:"value",axisLabel:{{formatter:function(v){{return v.toFixed(0)}}}}}},
    dataZoom:[{{type:"inside"}},{{type:"slider",bottom:4,height:22}}],
    series:[
      {{name:"Close",type:"line",data:price,lineStyle:{{color:"#B4B2A9",width:1}},symbol:"none",z:1}},
      {{name:"Golden",type:"scatter",data:gScat,symbol:"triangle",symbolSize:12,itemStyle:{{color:"#1D9E75"}},z:3}},
      {{name:"Death",type:"scatter",data:dScat,symbol:"triangle",symbolRotate:180,symbolSize:12,itemStyle:{{color:"#D85A30"}},z:3}}
    ]
  }});

  // Chart 2: MA comparison (auto Y range)
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
}}

function validateParams() {{
  var f = parseInt(document.getElementById("fastP").value)||10;
  var s = parseInt(document.getElementById("slowP").value)||60;
  if (f >= s) document.getElementById("slowP").value = f+1;
}}
function setPreset(f,s) {{
  document.getElementById("fastP").value = f;
  document.getElementById("slowP").value = s;
  var bs = document.querySelectorAll("#pGroup button");
  bs.forEach(function(b){{b.classList.remove("active")}});
  var n = f===5?"agg":f===10?"bal":"con";
  var t = document.querySelector('[data-p="'+n+'"]');
  if(t) t.classList.add("active");
  runBacktest();
}}

runBacktest();
</script>
</body>
</html>'''

out_path = os.path.join(OUT, 'ma_crossover_dashboard.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'\nDashboard: {out_path}')
print(f'Size: {os.path.getsize(out_path)/1024:.0f} KB')
print('Done!')
