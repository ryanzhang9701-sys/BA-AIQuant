"""Generate self-contained Single-Stock Turtle Strategy Deep-Dive HTML Dashboard v2"""
import json, pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "turtle_single_stock.html"

STOCKS = [
    {"code":"688981.SH","name":"中芯国际(A)","file":"data/688981.SH_中芯国际/daily_adjusted.csv","cols":{"d":"trade_date","o":"open_qfq","h":"high_qfq","l":"low_qfq","c":"close_qfq"},"lot":100},
    {"code":"00981.HK","name":"中芯国际(H)","file":"data/688981.SH_中芯国际/daily_hk.csv","cols":{"d":"trade_date","o":"open","h":"high","l":"low","c":"close"},"lot":500},
    {"code":"002594.SZ","name":"比亚迪(A)","file":"data/002594.SZ_比亚迪/daily_adjusted.csv","cols":{"d":"trade_date","o":"open_qfq","h":"high_qfq","l":"low_qfq","c":"close_qfq"},"lot":100},
    {"code":"01211.HK","name":"比亚迪(H)","file":"data/002594.SZ_比亚迪/daily_hk.csv","cols":{"d":"trade_date","o":"open","h":"high","l":"low","c":"close"},"lot":500},
    {"code":"603986.SH","name":"兆易创新","file":"data/603986.SH_兆易创新/daily_adjusted.csv","cols":{"d":"trade_date","o":"open_qfq","h":"high_qfq","l":"low_qfq","c":"close_qfq"},"lot":100},
]

raw_data = {}
for s in STOCKS:
    df = pd.read_csv(BASE / s["file"], encoding="utf-8-sig")
    cc = s["cols"]
    df = df[[cc[k] for k in ["d","o","h","l","c"]]].copy()
    df.columns = ["date","open","high","low","close"]
    df["date"] = df["date"].astype(str)
    records = [{"d":r["date"],"o":round(float(r["open"]),2),"h":round(float(r["high"]),2),"l":round(float(r["low"]),2),"c":round(float(r["close"]),2)} for _,r in df.iterrows()]
    raw_data[s["code"]] = {"name":s["name"],"lot":s["lot"],"data":records}

data_json = json.dumps(raw_data, ensure_ascii=False)

HTML = fr'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>海龟策略 · 单股票深度分析</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>
:root{{--bg:#F7F8FA;--card:#fff;--text:#1a1a2e;--muted:#7b8794;--border:#E4E7EB;--blue:#3B82F6;--coral:#F97316;--purple:#8B5CF6;--green:#22C55E;--red:#EF4444;--r:12px}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text)}}
.header{{background:linear-gradient(135deg,#1e293b,#0f172a);padding:16px 24px;color:#fff;display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
.header h1{{font-size:19px;font-weight:600}}
.header .sub{{font-size:11px;color:#94a3b8}}
select{{padding:7px 12px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;font-size:13px;cursor:pointer;min-width:160px}}
select:focus{{outline:none;border-color:var(--blue)}}
.container{{max-width:1500px;margin:0 auto;padding:12px 16px}}
.controls{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:10px 14px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap}}
.cg label{{font-size:10px;color:var(--muted);display:block;margin-bottom:2px}}
.cg input[type=range]{{width:90px}}
.cg .val{{font-size:11px;font-weight:600;display:inline-block;min-width:24px;text-align:center}}
.btn{{background:var(--blue);color:#fff;border:none;padding:6px 14px;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500}}
.btn:hover{{opacity:.9}}
.preset-btn{{background:#f1f5f9;border:1px solid var(--border);padding:5px 10px;border-radius:6px;font-size:11px;cursor:pointer}}
.preset-btn.active{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.kpi-grid{{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:10px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:12px;text-align:center}}
.kpi .l{{font-size:10px;color:var(--muted)}}.kpi .v{{font-size:20px;font-weight:700;margin-top:3px}}
.kpi .v.up{{color:var(--red)}}.kpi .v.dn{{color:var(--green)}}
.panel{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px;margin-bottom:10px}}
.panel h3{{font-size:13px;font-weight:600;margin-bottom:8px;color:var(--text)}}
.row2{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px}}
.chart{{width:100%;overflow:hidden;min-height:200px}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}}
th{{background:#f8fafc;font-weight:600;color:var(--muted);font-size:10px}}
td{{color:var(--text)}}
footer{{text-align:center;padding:12px;color:var(--muted);font-size:10px}}
#loading{{display:none;text-align:center;padding:40px;color:var(--muted);font-size:13px}}
#loading.show{{display:block}}
#error{{display:none;background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:12px;margin:8px 0;color:#991B1B;font-size:12px;font-family:monospace}}
</style>
</head>
<body>
<div class="header">
  <h1>海龟策略 · 单股票深度分析</h1>
  <select id="stockSelect" onchange="switchStock()"></select>
  <span class="sub">K线+通道 | TR/ATR | 净值回撤 | 持仓 | 逐笔盈亏 | 出场分析</span>
</div>
<div class="container">
<div id="error"></div>

<div class="controls">
  <div class="cg"><label>系统1 入场</label><input type="range" id="s1e" min="5" max="60" value="20" oninput="$$('s1eV').textContent=value;debounceRun()"><span class="val" id="s1eV">20</span>日</div>
  <div class="cg"><label>系统1 出场</label><input type="range" id="s1x" min="3" max="30" value="10" oninput="$$('s1xV').textContent=value;debounceRun()"><span class="val" id="s1xV">10</span>日</div>
  <div class="cg"><label>系统2 入场</label><input type="range" id="s2e" min="20" max="120" value="55" oninput="$$('s2eV').textContent=value;debounceRun()"><span class="val" id="s2eV">55</span>日</div>
  <div class="cg"><label>系统2 出场</label><input type="range" id="s2x" min="5" max="60" value="20" oninput="$$('s2xV').textContent=value;debounceRun()"><span class="val" id="s2xV">20</span>日</div>
  <button class="btn" onclick="runBacktest()">重新计算</button>
  <button class="preset-btn active" onclick="setPreset('classic')">经典</button>
  <button class="preset-btn" onclick="setPreset('fast')">灵敏</button>
  <button class="preset-btn" onclick="setPreset('slow')">迟钝</button>
</div>

<div id="loading" class="show">正在加载数据并计算...</div>

<div id="dashboard" style="display:none">
  <div class="kpi-grid" id="kpi"></div>
  <div class="panel"><h3>K线图 + 唐奇安通道 + 交易信号</h3><div class="chart" id="ch1" style="height:460px"></div></div>
  <div class="row2">
    <div class="panel"><h3>True Range & ATR(N) — 波动率</h3><div class="chart" id="ch2" style="height:260px"></div></div>
    <div class="panel"><h3>账户净值 & 回撤</h3><div class="chart" id="ch3" style="height:260px"></div></div>
  </div>
  <div class="row2">
    <div class="panel"><h3>持仓 Unit 数演变</h3><div class="chart" id="ch4" style="height:240px"></div></div>
    <div class="panel"><h3>逐笔交易盈亏</h3><div class="chart" id="ch5" style="height:240px"></div></div>
  </div>
  <div class="row2">
    <div class="panel"><h3>系统1 vs 系统2</h3><div class="chart" id="ch6" style="height:240px"></div></div>
    <div class="panel"><h3>出场原因分布</h3><div class="chart" id="ch7" style="height:240px"></div></div>
  </div>
  <div class="panel"><h3>绩效指标汇总</h3><div id="metric-table"></div></div>
  <div class="panel"><h3>交易明细 — 出入场日期</h3><div id="trade-log"></div></div>
</div>
</div>
<footer>数据: tushare(前复权A股) + akshare(H股) · AT(20) · Risk 1% · Stop 2N · Pyramiding 4U/0.5N</footer>

<script>
const RAW={data_json};
const $$=id=>document.getElementById(id);
const toDate=s=>s.slice(0,4)+'-'+s.slice(4,6)+'-'+s.slice(6,8);

// Init stock selector
const sel=$$('stockSelect');
Object.entries(RAW).forEach(([code,s])=>{{const o=document.createElement('option');o.value=code;o.textContent=s.name+' ('+code+')';sel.appendChild(o)}});
sel.value=Object.keys(RAW)[0];

let results=null;
const chartCache={{}};

function getChart(id){{
  if(!chartCache[id]){{const el=$$(id);if(el)chartCache[id]=echarts.init(el);}}
  return chartCache[id];
}}

function resizeAll(){{
  Object.values(chartCache).forEach(c=>{{try{{c.resize()}}catch(e){{}}}});
}}

// ---- Turtle Engine ----
function calcTR(d){{
  const t=new Array(d.length);t[0]=d[0].h-d[0].l;
  for(let i=1;i<d.length;i++){{const a=d[i].h-d[i].l,b=Math.abs(d[i].h-d[i-1].c),c=Math.abs(d[i].l-d[i-1].c);t[i]=Math.max(a,b,c)}}
  return t;
}}
function calcATR(tr,p){{
  const a=new Array(tr.length).fill(NaN);let s=0;for(let i=0;i<p;i++)s+=tr[i];a[p-1]=s/p;
  for(let i=p;i<tr.length;i++)a[i]=(a[i-1]*(p-1)+tr[i])/p;return a;
}}
function calcDC(d,ep,xp){{
  const u=new Array(d.length).fill(NaN),l=new Array(d.length).fill(NaN);
  for(let i=ep;i<d.length;i++){{let mx=-Infinity;for(let j=i-ep;j<i;j++)mx=Math.max(mx,d[j].h);u[i]=mx}}
  for(let i=xp;i<d.length;i++){{let mn=Infinity;for(let j=i-xp;j<i;j++)mn=Math.min(mn,d[j].l);l[i]=mn}}
  return{{upper:u,lower:l}};
}}

function runTurtle(data,lot,s1e,s1x,s2e,s2x){{
  const n=data.length,tr=calcTR(data),atr=calcATR(tr,20);
  const s1=calcDC(data,s1e,s1x),s2=calcDC(data,s2e,s2x);
  const pos=[],sigs=[],eq=[];
  let cash=1e6,lastS1pnl=0,holdings=[];
  const si=Math.max(s2e,20)+1;

  function us(nv){{const r=Math.floor(cash*0.01/(nv*lot));return Math.max(r*lot,lot)}}

  for(let i=si;i<n;i++){{
    const td=data[i],nv=atr[i];if(isNaN(nv)||nv<=0){{eq.push([td.d,cash,0]);continue}}
    // stops
    const rm=[];for(const h of holdings){{if(td.l<h.stop){{cash+=h.stop*h.shares;pos.push({{sys:h.sys,ed:h.entry_date,ep:h.ep,exd:td.d,exp:h.stop,xr:'STOP_LOSS',u:h.u,shares:h.shares,pnl:(h.stop-h.ep)*h.shares}});sigs.push({{d:td.d,sys:h.sys,tp:'STOP',px:h.stop}});if(h.sys==='S1')lastS1pnl=(h.stop-h.ep)*h.shares;rm.push(h)}}}}holdings=holdings.filter(h=>!rm.includes(h));
    // trailing exits
    [{{sys:'S2',low:s2.lower,label:'TRAILING_S2'}},{{sys:'S1',low:s1.lower,label:'TRAILING_S1'}}].forEach(({{sys,low,label}})=>{{
      if(!isNaN(low[i])&&td.c<low[i]){{for(const h of [...holdings]){{if(h.sys!==sys)continue;const px=data[Math.min(i+1,n-1)].o*0.999;cash+=px*h.shares;pos.push({{sys,ed:h.entry_date,ep:h.ep,exd:td.d,exp:px,xr:label,u:h.u,shares:h.shares,pnl:(px-h.ep)*h.shares}});sigs.push({{d:td.d,sys,tp:'SELL',px}});if(sys==='S1')lastS1pnl=(px-h.ep)*h.shares;holdings=holdings.filter(x=>x!==h)}}}}
    }});
    // S2 entry
    if(!isNaN(s2.upper[i])&&td.c>s2.upper[i]&&!holdings.some(h=>h.sys==='S2')&&i+1<n){{const px=data[i+1].o*1.001,sz=us(nv);if(sz>0&&px*sz<=cash){{cash-=px*sz;holdings.push({{sys:'S2',entry_date:data[i+1].d,ep:px,en:nv,u:1,shares:sz,stop:px-2*nv,addP:[px]}});sigs.push({{d:td.d,sys:'S2',tp:'BUY',px,u:1}})}}}}
    // S1 entry
    if(!isNaN(s1.upper[i])&&td.c>s1.upper[i]&&!holdings.some(h=>h.sys==='S1')&&lastS1pnl>=0&&i+1<n){{const px=data[i+1].o*1.001,sz=us(nv);if(sz>0&&px*sz<=cash){{cash-=px*sz;holdings.push({{sys:'S1',entry_date:data[i+1].d,ep:px,en:nv,u:1,shares:sz,stop:px-2*nv,addP:[px]}});sigs.push({{d:td.d,sys:'S1',tp:'BUY',px,u:1}})}}}}
    // pyramid
    for(const h of holdings){{if(h.u>=4||i+1>=n)continue;const nd=h.addP[h.addP.length-1]+0.5*h.en;if(td.c>=nd){{const px=data[i+1].o*1.001,sz=us(nv);if(sz>0&&px*sz<=cash){{cash-=px*sz;h.u++;h.shares+=sz;h.addP.push(px);h.stop=Math.max(h.stop,px-2*nv);sigs.push({{d:td.d,sys:h.sys,tp:'ADD',px,u:h.u}})}}}}}}
    // equity
    let mkt=0;for(const h of holdings)mkt+=h.shares*td.c;eq.push([td.d,cash+mkt,holdings.reduce((s,h)=>s+h.u,0)]);
  }}
  // force exit
  for(const h of holdings){{const px=data[n-1].c;cash+=px*h.shares;pos.push({{sys:h.sys,ed:h.entry_date,ep:h.ep,exd:data[n-1].d,exp:px,xr:'END_OF_DATA',u:h.u,shares:h.shares,pnl:(px-h.ep)*h.shares}})}}
  return{{tr,atr,s1u:s1.upper,s1l:s1.lower,s2u:s2.upper,s2l:s2.lower,pos,sigs,eq,data}};
}}

function calcPerf(eq,pos){{
  const ev=eq.map(e=>e[1]),init=ev[0],totalRet=ev[ev.length-1]/init-1;
  const annRet=Math.pow(1+totalRet,252/ev.length)-1;
  let peak=ev[0],maxDD=0;for(const v of ev){{if(v>peak)peak=v;const dd=(v-peak)/peak;if(dd<maxDD)maxDD=dd}}
  const rets=[];for(let i=1;i<ev.length;i++)rets.push((ev[i]-ev[i-1])/ev[i-1]);
  const avgR=rets.reduce((s,r)=>s+r,0)/rets.length,stdR=Math.sqrt(rets.reduce((s,r)=>s+Math.pow(r-avgR,2),0)/rets.length);
  const sharpe=stdR>0?(avgR-0.02/252)/stdR*Math.sqrt(252):0;
  const wins=pos.filter(p=>p.pnl>0),losses=pos.filter(p=>p.pnl<=0);
  const tw=wins.reduce((s,p)=>s+p.pnl,0),tl=Math.abs(losses.reduce((s,p)=>s+p.pnl,0));
  const avgHD=pos.length>0?pos.reduce((s,p)=>s+(new Date(toDate(p.exd))-new Date(toDate(p.ed)))/864e5,0)/pos.length:0;
  return{{totalRet,annRet,maxDD,sharpe,winRate:pos.length?wins.length/pos.length:0,pf:tl>0?tw/tl:99,trades:pos.length,avgHD,avgWin:wins.length?tw/wins.length:0,avgLoss:losses.length?tl/losses.length:0}};
}}

let timer=null;
function debounceRun(){{clearTimeout(timer);timer=setTimeout(runBacktest,250)}}
function getParams(){{return{{s1e:+$$('s1e').value,s1x:+$$('s1x').value,s2e:+$$('s2e').value,s2x:+$$('s2x').value}}}}

function runBacktest(){{
  const code=sel.value,p=getParams();
  $$('loading').classList.add('show');$$('dashboard').style.display='none';$$('error').style.display='none';
  setTimeout(()=>{{
    try{{
      const stock=RAW[code];
      results=runTurtle(stock.data,stock.lot,p.s1e,p.s1x,p.s2e,p.s2x);
      results.name=stock.name;results.code=code;
      results.perf=calcPerf(results.eq,results.pos);
      $$('loading').classList.remove('show');$$('dashboard').style.display='block';
      render();
      setTimeout(resizeAll, 50);
    }}catch(e){{
      $$('loading').classList.remove('show');
      $$('error').style.display='block';$$('error').textContent='计算错误: '+e.message+'\\n'+e.stack;
    }}
  }},20);
}}

function switchStock(){{runBacktest()}}

function setPreset(t){{
  const presets={{classic:[20,10,55,20],fast:[10,5,40,15],slow:[30,15,70,30]}};
  const p=presets[t];
  ['s1e','s1x','s2e','s2x'].forEach((id,i)=>{{$$(id).value=p[i];$$(id+'V').textContent=p[i]}});
  document.querySelectorAll('.preset-btn').forEach(b=>b.classList.toggle('active',b.textContent.trim()==={{classic:'经典',fast:'灵敏',slow:'迟钝'}}[t]));
  runBacktest();
}}

function render(){{
  const r=results,p=r.perf,d=r.data;if(!r)return;
  const clr=(v,f)=>parseFloat(v)>0?'up':'dn';
  // KPI — 7 cards including cumulative return
  const kpis=[['累计收益',(p.totalRet*100).toFixed(1)+'%',clr(p.totalRet)],['年化收益',(p.annRet*100).toFixed(1)+'%',clr(p.annRet)],['最大回撤',(p.maxDD*100).toFixed(1)+'%','dn'],['夏普比率',p.sharpe.toFixed(2),clr(p.sharpe)],['胜率',(p.winRate*100).toFixed(0)+'%',clr(p.winRate-0.3)],['盈亏因子',p.pf.toFixed(2),clr(p.pf-1)],['交易笔数',p.trades,'']];
  $$('kpi').innerHTML=kpis.map(([l,v,c])=>`<div class="kpi"><div class="l">${{l}}</div><div class="v ${{c}}">${{v}}</div></div>`).join('');

  const dates=d.map(x=>toDate(x.d));

  // Chart 1: K-line — dynamic legend
  const c1=getChart('ch1');
  if(c1){{const mk=d.map((x,i)=>[dates[i],x.o,x.c,x.l,x.h]);
  const pp=getParams();
  const lS2e='系统2上轨('+pp.s2e+'日)',lS2x='系统2下轨('+pp.s2x+'日)';
  const lS1e='系统1上轨('+pp.s1e+'日)',lS1x='系统1下轨('+pp.s1x+'日)';
  c1.setOption({{tooltip:{{trigger:'axis',axisPointer:{{type:'cross'}}}},grid:{{left:65,right:25,top:35,bottom:20}},xAxis:{{type:'category',data:dates,axisLabel:{{fontSize:8,rotate:45,interval:Math.max(1,Math.floor(dates.length/10))}}}},yAxis:{{type:'value',scale:true,axisLabel:{{fontSize:9}}}},series:[
    {{name:'K线',type:'candlestick',data:mk,itemStyle:{{color:'#EF4444',color0:'#22C55E',borderColor:'#DC2626',borderColor0:'#16A34A'}}}},
    {{name:lS2e,type:'line',data:r.s2u.map((v,i)=>[dates[i],v]),lineStyle:{{color:'#3B82F6',width:1.8}},symbol:'none',connectNulls:false}},
    {{name:lS2x,type:'line',data:r.s2l.map((v,i)=>[dates[i],v]),lineStyle:{{color:'#3B82F6',width:1.8}},symbol:'none',connectNulls:false,areaStyle:{{color:'rgba(59,130,246,0.06)'}}}},
    {{name:lS1e,type:'line',data:r.s1u.map((v,i)=>[dates[i],v]),lineStyle:{{color:'#F97316',width:1.2,type:'dashed',dashOffset:3}},symbol:'none',connectNulls:false}},
    {{name:lS1x,type:'line',data:r.s1l.map((v,i)=>[dates[i],v]),lineStyle:{{color:'#EF4444',width:1.2,type:'dashed',dashOffset:3}},symbol:'none',connectNulls:false}},
    {{name:'买入',type:'scatter',data:r.sigs.filter(s=>s.tp==='BUY').map(s=>[toDate(s.d),s.px]),symbol:'arrow',symbolSize:16,itemStyle:{{color:'#DC2626'}}}},
    {{name:'加仓',type:'scatter',data:r.sigs.filter(s=>s.tp==='ADD').map(s=>[toDate(s.d),s.px]),symbol:'diamond',symbolSize:10,itemStyle:{{color:'#F97316',borderColor:'#fff',borderWidth:1}}}},
    {{name:'止损/出场',type:'scatter',data:r.sigs.filter(s=>s.tp==='SELL'||s.tp==='STOP').map(s=>[toDate(s.d),s.px]),symbol:'arrow',symbolSize:16,itemStyle:{{color:'#16A34A'}},symbolRotate:180}},
  ],legend:{{data:['K线',lS2e,lS2x,lS1e,lS1x,'买入','加仓','止损/出场'],top:0,left:'center',textStyle:{{fontSize:9}},itemGap:6}}}},true);}}

  // Chart 2: TR + ATR
  const c2=getChart('ch2');
  if(c2){{c2.setOption({{tooltip:{{trigger:'axis'}},grid:{{left:50,right:15,top:30,bottom:20}},xAxis:{{type:'category',data:dates,axisLabel:{{show:false}}}},yAxis:{{axisLabel:{{fontSize:9}}}},series:[{{name:'TR',type:'bar',data:r.tr.map((v,i)=>[dates[i],v]),itemStyle:{{color:'#BFDBFE'}},barWidth:2}},{{name:'ATR(20)=N',type:'line',data:r.atr.map((v,i)=>[dates[i],v]),lineStyle:{{color:'#8B5CF6',width:2}},symbol:'none'}}],legend:{{top:0,left:'center',textStyle:{{fontSize:9}}}}}},true);}}

  // Chart 3: Equity + Drawdown
  const c3=getChart('ch3');
  if(c3){{const ev=r.eq.map(e=>e[1]),init=ev[0];let peak=ev[0];const dd=ev.map(v=>{{if(v>peak)peak=v;return(v-peak)/peak*100}});const eqDates=r.eq.map(e=>toDate(e[0]));
  c3.setOption({{tooltip:{{trigger:'axis'}},grid:{{left:55,right:50,top:30,bottom:20}},xAxis:{{type:'category',data:eqDates,axisLabel:{{show:false}}}},yAxis:[{{type:'value',name:'净值',axisLabel:{{fontSize:9}}}},{{type:'value',name:'回撤%',axisLabel:{{fontSize:9}}}}],series:[{{name:'策略净值',type:'line',data:ev.map((v,i)=>[eqDates[i],v/init*100]),lineStyle:{{color:'#3B82F6',width:2.5}},symbol:'none',smooth:true}},{{name:'回撤',type:'line',yAxisIndex:1,data:dd.map((v,i)=>[eqDates[i],v]),lineStyle:{{color:'#EF4444',width:1}},symbol:'none',areaStyle:{{color:'rgba(239,68,68,0.18)'}}}}],legend:{{top:0,left:'center',textStyle:{{fontSize:9}}}}}},true);}}

  // Chart 4: Unit
  const c4=getChart('ch4');
  if(c4){{c4.setOption({{tooltip:{{trigger:'axis'}},grid:{{left:50,right:15,top:30,bottom:20}},xAxis:{{type:'category',data:r.eq.map(e=>toDate(e[0])),axisLabel:{{show:false}}}},yAxis:{{min:0,max:Math.max(5,Math.max(...r.eq.map(e=>e[2]))+1),axisLabel:{{fontSize:9}}}},series:[{{name:'持仓Unit',type:'line',data:r.eq.map(e=>[toDate(e[0]),e[2]]),lineStyle:{{color:'#3B82F6',width:2}},symbol:'none',areaStyle:{{color:'rgba(59,130,246,0.12)'}},step:'end'}},{{name:'上限4U',type:'line',data:[[toDate(r.eq[0][0]),4],[toDate(r.eq[r.eq.length-1][0]),4]],lineStyle:{{color:'#EF4444',width:1,type:'dashed'}},symbol:'none'}}],legend:{{top:0,left:'center',textStyle:{{fontSize:9}}}}}},true);}}

  // Chart 5: P&L bars
  const c5=getChart('ch5');
  if(c5&&r.pos.length>0){{c5.setOption({{tooltip:{{trigger:'axis'}},grid:{{left:60,right:15,top:10,bottom:25}},xAxis:{{type:'category',data:r.pos.map((_,i)=>'#'+(i+1)),axisLabel:{{fontSize:8}}}},yAxis:{{axisLabel:{{fontSize:9,formatter:v=>(v/1e3).toFixed(0)+'k'}}}},series:[{{name:'盈亏',type:'bar',data:r.pos.map(p=>p.pnl),itemStyle:{{color:p=>p.value>=0?'#22C55E':'#EF4444'}},barWidth:20,label:{{show:true,position:'top',fontSize:7,formatter:p=>(p.value/1e3).toFixed(0)+'k'}}}}]}},true);}}

  // Chart 6: S1 vs S2
  const c6=getChart('ch6');
  if(c6){{const s1p=r.pos.filter(p=>p.sys==='S1'),s2p=r.pos.filter(p=>p.sys==='S2'),s1pnl=s1p.reduce((s,p)=>s+p.pnl,0),s2pnl=s2p.reduce((s,p)=>s+p.pnl,0);
  c6.setOption({{tooltip:{{}},grid:{{left:80,right:20,top:20,bottom:25}},xAxis:{{type:'category',data:['系统1\n('+s1p.length+'笔)','系统2\n('+s2p.length+'笔)'],axisLabel:{{fontSize:11}}}},yAxis:{{axisLabel:{{fontSize:9,formatter:v=>(v/1e3).toFixed(0)+'k'}}}},series:[{{type:'bar',data:[s1pnl,s2pnl],itemStyle:{{color:p=>p.dataIndex===0?'#22C55E':'#3B82F6'}},barWidth:60,label:{{show:true,position:'top',fontSize:11,fontWeight:'bold',formatter:p=>'¥'+(p.value/1e3).toFixed(1)+'k'}}}}]}},true);}}

  // Chart 7: Exit reasons pie
  const c7=getChart('ch7');
  if(c7&&r.pos.length>0){{const xr={{}};r.pos.forEach(p=>{{xr[p.xr]=(xr[p.xr]||0)+1}});const cmap={{'STOP_LOSS':'#EF4444','TRAILING_S1':'#22C55E','TRAILING_S2':'#3B82F6','END_OF_DATA':'#94a3b8'}};
  c7.setOption({{tooltip:{{trigger:'item',formatter:'{{b}}: {{c}}笔 ({{d}}%)'}},series:[{{type:'pie',radius:['40%','70%'],data:Object.entries(xr).map(([k,v])=>({{name:k,value:v,itemStyle:{{color:cmap[k]||'#888'}}}})),label:{{fontSize:10}},emphasis:{{itemStyle:{{shadowBlur:8}}}}}}],legend:{{bottom:0,textStyle:{{fontSize:9}}}}}},true);}}

  // Metrics table
  $$('metric-table').innerHTML=`<table><thead><tr><th>指标</th><th>数值</th><th>指标</th><th>数值</th></tr></thead><tbody>${{[
    ['年化收益率',(p.annRet*100).toFixed(1)+'%','最大回撤',(p.maxDD*100).toFixed(1)+'%'],
    ['夏普比率',p.sharpe.toFixed(2),'胜率',(p.winRate*100).toFixed(0)+'%'],
    ['盈亏因子',p.pf.toFixed(2),'总交易笔数',p.trades],
    ['平均盈利',p.avgWin>0?'¥'+Math.round(p.avgWin).toLocaleString():'N/A','平均亏损',p.avgLoss>0?'¥'+Math.round(p.avgLoss).toLocaleString():'N/A'],
    ['平均持仓',Math.round(p.avgHD)+'天','TR均值',(r.tr.reduce((s,x)=>s+x,0)/r.tr.length).toFixed(2)],
    ['累计收益率 (Cumulative Return)',(p.totalRet*100).toFixed(1)+'%','买入持有收益 (Buy & Hold)',(d[d.length-1].c/d[0].c*100-100).toFixed(1)+'%']
  ].map(r=>'<tr>'+r.map(v=>`<td>${{v}}</td>`).join('')+'</tr>').join('')}}</tbody></table>`;

  // Trade log table
  if(r.pos.length>0){{
    $$('trade-log').innerHTML=`<table><thead><tr><th>#</th><th>系统</th><th>入场日期</th><th>入场价</th><th>出场日期</th><th>出场价</th><th>持仓天数</th><th>盈亏(¥)</th><th>盈亏%</th><th>出场原因</th></tr></thead><tbody>${{r.pos.map((p,i)=>`<tr>
      <td>${{i+1}}</td><td>${{p.sys}}</td><td>${{toDate(p.ed)}}</td><td>${{p.ep.toFixed(2)}}</td>
      <td>${{toDate(p.exd)}}</td><td>${{p.exp.toFixed(2)}}</td>
      <td>${{Math.round((new Date(toDate(p.exd))-new Date(toDate(p.ed)))/864e5)}}天</td>
      <td style="color:${{p.pnl>=0?'#EF4444':'#22C55E'}};font-weight:600">¥${{Math.round(p.pnl).toLocaleString()}}</td>
      <td style="color:${{p.pnl>=0?'#EF4444':'#22C55E'}}">${{(p.pnl/(p.ep*p.shares)*100).toFixed(2)}}%</td>
      <td>${{p.xr}}</td>
    </tr>`).join('')}}</tbody></table>`;
  }}else{{$$('trade-log').innerHTML='<p style="color:#7b8794;padding:8px">该股票在回测期内无交易信号</p>'}}
}}

// Start
setPreset('classic');
window.addEventListener('resize',resizeAll);
</script>
</body>
</html>'''

OUT.write_text(HTML, encoding="utf-8")
print(f"Dashboard generated: {OUT}")
print(f"Size: {OUT.stat().st_size:,} bytes")
