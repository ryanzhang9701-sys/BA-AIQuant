"""Strategy comparison dashboard — 5 stocks (A+H) + K-line + analysis"""
import os, json, pandas as pd, numpy as np

SCRIPT = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(SCRIPT)
DATA  = os.path.join(PROJ, 'data')
OUT   = os.path.join(PROJ, 'TASK3')
os.makedirs(OUT, exist_ok=True)

# Discover all stocks (A + HK)
LABELS = {
    '002594.SZ': 'BYD A', '002594.HK': 'BYD HK',
    '603986.SH': 'GigaDevice A',
    '688981.SH': 'SMIC A', '688981.HK': 'SMIC HK',
}
stocks = []
for dname in os.listdir(DATA):
    dpath = os.path.join(DATA, dname)
    if not os.path.isdir(dpath): continue
    parts = dname.split('_')
    code = parts[0] if parts else ''
    adj = os.path.join(dpath, 'daily_adjusted.csv')
    hk  = os.path.join(dpath, 'daily_hk.csv')
    if os.path.exists(adj):
        df = pd.read_csv(adj, encoding='utf-8-sig', parse_dates=['trade_date'])
        stocks.append({'name': LABELS.get(code, code), 'code': code,
            'market': 'A', 'data': df, 'price_col': 'close_qfq'})
    if os.path.exists(hk):
        df = pd.read_csv(hk, encoding='utf-8-sig', parse_dates=['trade_date'])
        hk_code = code.split('.')[0] + '.HK'
        stocks.append({'name': LABELS.get(hk_code, hk_code), 'code': hk_code,
            'market': 'HK', 'data': df, 'price_col': 'close'})

print(f'Found {len(stocks)} stocks')
for s in stocks: print(f'  {s["name"]}: {len(s["data"])} rows')

FAST, SLOW, MA200 = 10, 60, 200
COMM, STAMP, SLIP = 0.0003, 0.0005, 0.001

def run_strategy(df, price_col, use_filter=False):
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['price'] = df[price_col]
    df['ma_fast'] = df['price'].ewm(span=FAST, adjust=False).mean()
    df['ma_slow'] = df['price'].ewm(span=SLOW, adjust=False).mean()
    df['ma200'] = df['price'].rolling(MA200).mean()
    warmup = max(FAST, SLOW)
    dv = df.iloc[warmup:].copy().reset_index(drop=True)
    dv['signal_raw'] = (dv['ma_fast'] > dv['ma_slow']).astype(int)
    if use_filter:
        dv['trend_ok'] = ((dv['price'] > dv['ma200']) | dv['ma200'].isna()).astype(int)
        dv['signal'] = (dv['signal_raw'] & dv['trend_ok']).astype(int)
    else:
        dv['signal'] = dv['signal_raw']
    dv['cross'] = dv['signal'].diff()
    dv['position'] = dv['signal'].shift(1).fillna(0).astype(int)
    dv['price_ret'] = dv['price'].pct_change()
    dv['gross_ret'] = dv['position'] * dv['price_ret']
    dv['trade_act'] = dv['position'].diff().abs()
    dv['cost'] = dv['trade_act'] * (COMM + SLIP)
    dv.loc[dv['position'].diff() < 0, 'cost'] += STAMP
    dv['strat_ret'] = dv['gross_ret'] - dv['cost']
    dv['equity'] = (1 + dv['strat_ret'].fillna(0)).cumprod()
    dv['bench_eq'] = (1 + dv['price_ret'].fillna(0)).cumprod()
    pk = dv['equity'].cummax()
    dv['dd'] = (dv['equity'] - pk) / pk
    ret = dv['strat_ret'].dropna(); N = len(ret)
    total_ret = dv['equity'].iloc[-1] - 1
    bench_ret = dv['bench_eq'].iloc[-1] - 1
    ann_ret = (1+total_ret)**(252/N)-1 if N>0 else 0
    rc = ret[ret!=0] if len(ret[ret!=0])>0 else ret
    sharpe = np.sqrt(252)*rc.mean()/rc.std() if rc.std()>0 else 0
    max_dd = dv['dd'].min()
    calmar = ann_ret/abs(max_dd) if max_dd!=0 else 0
    g_entry = dv[dv['cross']==1]
    trades_ret = []
    for i in range(len(g_entry)):
        ei = dv[dv['trade_date']==g_entry['trade_date'].iloc[i]].index[0]
        xi = len(dv)-1 if i+1>=len(g_entry) else dv[dv['trade_date']==g_entry['trade_date'].iloc[i+1]].index[0]-1
        trades_ret.append(dv.loc[xi,'price']/dv.loc[ei,'price']-1)
    wins = [r for r in trades_ret if r>0]
    wr = len(wins)/len(trades_ret) if trades_ret else 0

    # Candlestick data [open, close, low, high] for the full range
    kline = []
    for _, row in dv.iterrows():
        o = row.get('open_qfq', row.get('open'))
        h = row.get('high_qfq', row.get('high'))
        l = row.get('low_qfq', row.get('low'))
        c = row.get(price_col)
        kline.append([round(o,2), round(c,2), round(l,2), round(h,2)])

    return dict(total_ret=total_ret, bench_ret=bench_ret, ann_ret=ann_ret,
                sharpe=sharpe, max_dd=max_dd, calmar=calmar, win_rate=wr,
                trades=len(trades_ret), equity=dv['equity'].round(4).tolist(),
                dates=dv['trade_date'].dt.strftime('%Y-%m-%d').tolist(),
                dd=dv['dd'].round(4).tolist(), kline=kline,
                ma_fast=dv['ma_fast'].round(2).tolist(),
                ma_slow=dv['ma_slow'].round(2).tolist())

results = {}
for s in stocks:
    print(f'Running: {s["name"]} (base)...')
    results[s['name']+'_base'] = run_strategy(s['data'], s['price_col'], False)
    print(f'Running: {s["name"]} (+MA200)...')
    results[s['name']+'_filter'] = run_strategy(s['data'], s['price_col'], True)
print('All done.')

# Build table
def mcells(key, is_pct):
    r = ''
    for s in stocks:
        b = results[s['name']+'_base']; f = results[s['name']+'_filter']
        bv, fv = b[key], f[key]
        bc = 'up' if bv>0 else ('down' if bv<0 else 'neutral')
        fc = 'up' if fv>0 else ('down' if fv<0 else 'neutral')
        d = fv-bv; dc = 'up' if d>0 else ('down' if d<0 else 'neutral')
        fmt = lambda v: f'{v:+.2%}' if is_pct else (f'{v:.2f}' if abs(v)<100 else f'{int(v)}')
        r += f'<td class="{bc}">{fmt(bv)}</td><td class="{fc}">{fmt(fv)}</td><td class="{dc}">{fmt(d)}</td>'
    return r

rd = [('Total Return','total_ret',True),('Benchmark','bench_ret',True),
      ('Annual Return','ann_ret',True),('Sharpe','sharpe',False),
      ('Max DD','max_dd',True),('Calmar','calmar',False),
      ('Win Rate','win_rate',True),('Trades','trades',False)]
trows = ''
for lb,k,ip in rd: trows += f'<tr><td>{lb}</td>{mcells(k,ip)}</tr>'

hdr_cols = ''.join(f'<th colspan="3" class="cg">{s["name"]}</th>' for s in stocks)
hdr_sub  = '<tr>'+''.join('<th class="cb">Base</th><th class="cf">+MA200</th><th class="cd">D</th>' for _ in stocks)+'</tr>'

# K-line + analysis data
KL_JSON = json.dumps({s['name']: results[s['name']+'_base']['kline'] for s in stocks}, ensure_ascii=False)
EQ_JSON = json.dumps({s['name']: {'dates':results[s['name']+'_base']['dates'],'equity':results[s['name']+'_base']['equity'],'dd':results[s['name']+'_base']['dd'],'equity_f':results[s['name']+'_filter']['equity'],'dd_f':results[s['name']+'_filter']['dd'],'ma_fast':results[s['name']+'_base']['ma_fast'],'ma_slow':results[s['name']+'_base']['ma_slow']} for s in stocks}, ensure_ascii=False)

# Analysis summary text
analysis = ''
for s in stocks:
    b = results[s['name']+'_base']; f = results[s['name']+'_filter']
    trend = 'strong uptrend' if b['bench_ret']>0.3 else ('moderate uptrend' if b['bench_ret']>0 else ('sideways/downtrend' if b['bench_ret']>-0.1 else 'strong downtrend'))
    base_ok = 'effective' if b['total_ret']>b['bench_ret'] else ('slightly worse' if abs(b['total_ret']-b['bench_ret'])<0.05 else 'underperformed')
    filt_eff = 'improved Sharpe by ' + f'{f["sharpe"]-b["sharpe"]:+.2f}' if f['sharpe']>b['sharpe'] else ('reduced trades from ' + str(b['trades']) + ' to ' + str(f['trades']) + ' but Sharpe unchanged' if f['trades']<b['trades'] else 'no significant improvement')
    analysis += f'<tr><td>{s["name"]}</td><td>{trend}</td><td>Base: {base_ok}</td><td>Filter: {filt_eff}</td></tr>\n'

html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Strategy Comparison — 5 Stocks + K-line</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root{{--bg:#fff;--bg2:#f8f9fa;--t:#1a1a2e;--t2:#555;--bd:#e0e0e0;--gn:#1D9E75;--rd:#D85A30;--pu:#534AB7;--bl:#378ADD;--ra:10px}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:var(--t);background:#f5f5f5;line-height:1.7}}
.hd{{background:#fff;border-bottom:1px solid var(--bd);padding:14px 28px}}
.hd h1{{font-size:20px;font-weight:600}}.hd .sub{{font-size:12px;color:#888}}
.container{{max-width:1400px;margin:0 auto;padding:16px 20px}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:var(--ra);padding:16px;margin-bottom:14px;overflow-x:auto}}
.card h3{{font-size:15px;font-weight:600;margin-bottom:12px;padding-left:8px;border-left:3px solid var(--pu)}}
.card h4{{font-size:14px;font-weight:600;margin:0 0 8px;color:var(--t)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:var(--bg2);text-align:left;padding:8px 10px;font-weight:600;border-bottom:2px solid var(--bd);white-space:nowrap}}
th.cg{{background:#EEEDFE;color:var(--pu);text-align:center;border-bottom:2px solid var(--pu)}}
th.cb{{background:#EEEDFE;color:var(--pu)}}
th.cf{{background:#E6F1FB;color:#378ADD}}
th.cd{{background:#f8f9fa;color:var(--t2)}}
td{{padding:8px 10px;border-bottom:1px solid var(--bd);text-align:right}}
tr:hover td{{background:#fafafa}}
td:first-child{{text-align:left;font-weight:500}}
.up{{color:var(--rd)}}.down{{color:var(--gn)}}.neutral{{color:var(--t)}}
.chart{{width:100%;height:380px}}
.analysis-box{{background:#FFFDE7;border:1px solid #FFE082;border-radius:var(--ra);padding:14px;margin-bottom:14px;font-size:13px}}
.analysis-box h4{{margin-bottom:8px;font-size:14px}}
.analysis-box p,.analysis-box li{{margin:4px 0;color:#555}}
.analysis-box ul{{padding-left:20px}}
.foot{{padding:8px 14px;font-size:11px;color:#888;text-align:center}}
</style></head><body>
<div class="hd"><h1>Strategy Comparison — 5 Stocks + K-line Analysis</h1><p class="sub">EMA(10)/EMA(60) Base vs +MA200 Trend Filter | Commission 0.03% + Stamp 0.05% + Slippage 0.1% | T+1 | 3 A-shares + 2 HK stocks</p></div>
<div class="container">

<div class="card"><h3>Performance Metrics Summary</h3>
<table><thead><tr><th rowspan="2">Metric</th>{hdr_cols}</tr>{hdr_sub}</thead><tbody>{trows}</tbody></table>
</div>

<div class="analysis-box"><h4>Strategy Effectiveness vs Price Pattern</h4>
<table><thead><tr><th>Stock</th><th>Price Pattern (1Y)</th><th>Base Strategy</th><th>+MA200 Filter Effect</th></tr></thead><tbody>
{analysis}
</tbody></table>
<p style="margin-top:12px"><b>Key Insight:</b> The MA crossover strategy works best in <b>strong trending markets</b> (either up or down).
In downtrends, the trend filter reduces whipsaw losses by avoiding dead-cat bounces.
In uptrends, the base strategy captures the full trend but may have larger drawdowns.
In sideways/choppy markets, both strategies struggle — this is the natural weakness of trend-following.</p>
</div>

<div class="card"><h3>K-line + MA overlay (Base strategy signals)</h3></div>
<div id="klineContainer" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px;padding:0 20px 16px">
{''.join(f'<div><h4>{s["name"]}</h4><div class="chart" id="kl{si}"></div></div>' for si, s in enumerate(stocks))}
</div>

</div>
<div class="foot">Based on ma_crossover_strategy_spec.md | Past performance != future results</div>
<script>
var KL={KL_JSON};var EQ={EQ_JSON};
var STKS={json.dumps([s['name'] for s in stocks])};
var allCharts=[];

// K-line charts
for(var i=0;i<STKS.length;i++){{
  var sn=STKS[i],kl=KL[sn],eq=EQ[sn];
  var dates=eq.dates;
  var c=echarts.init(document.getElementById('kl'+i));
  c.setOption({{
    tooltip:{{trigger:'axis',axisPointer:{{type:'cross'}}}},
    grid:{{left:70,right:30,top:40,bottom:60}},
    xAxis:{{type:'category',data:dates,axisLabel:{{formatter:function(v){{return v.slice(5)}}}}}},
    yAxis:{{type:'value',scale:true}},
    dataZoom:[{{type:'inside'}},{{type:'slider',bottom:6,height:22}}],
    series:[
      {{name:'K-line',type:'candlestick',data:kl,
        itemStyle:{{color:'#D85A30',color0:'#1D9E75',borderColor:'#D85A30',borderColor0:'#1D9E75'}},z:1}},
      {{name:'EMA10',type:'line',data:eq.ma_fast,lineStyle:{{color:'#D85A30',width:1.2}},symbol:'none',z:2}},
      {{name:'EMA60',type:'line',data:eq.ma_slow,lineStyle:{{color:'#534AB7',width:1.5}},symbol:'none',z:2}}
    ]
  }});
  allCharts.push(c);
}}

window.addEventListener('resize',function(){{allCharts.forEach(function(c){{c.resize()}})}});
</script>
</body></html>'''

out_path = os.path.join(OUT, 'strategy_comparison_dashboard.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Dashboard: {out_path}')
print(f'Size: {os.path.getsize(out_path)/1024:.0f} KB')
print('Done!')
