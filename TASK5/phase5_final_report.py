# -*- coding: utf-8 -*-
"""Phase 5 — 最终报告 HTML"""
import json, os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化选股 ML 建模 — 最终报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f6f8;color:#2c3e50;line-height:1.7;}
.container{max-width:1100px;margin:0 auto;padding:20px;}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:40px 30px;border-radius:10px;margin-bottom:24px;}
.hero h1{font-size:26px;margin-bottom:8px;}
.hero .meta{font-size:13px;color:#a0aec0;}
.card{background:#fff;border-radius:8px;padding:24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
.card h2{font-size:18px;color:#1a1a2e;border-left:4px solid #e74c3c;padding-left:12px;margin-bottom:16px;}
.card h3{font-size:15px;color:#333;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px solid #eee;}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;}
th{background:#f8f9fa;padding:8px 10px;text-align:center;font-weight:600;border:1px solid #e9ecef;font-size:12px;}
td{padding:7px 10px;text-align:center;border:1px solid #e9ecef;}
tr:hover{background:#f8f9fa;}
.chart-box{width:100%;height:400px;margin:10px 0;}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px;}
.tag-red{background:#fde8e8;color:#c0392b;}
.tag-green{background:#e8f8e8;color:#27ae60;}
.tag-blue{background:#e8f0fe;color:#2980b9;}
.tag-yellow{background:#fef9e7;color:#f39c12;}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;}
.badge-ok{background:#e8f8e8;color:#27ae60;}
.badge-warn{background:#fef9e7;color:#f39c12;}
.badge-fail{background:#fde8e8;color:#c0392b;}
ul{padding-left:20px;margin:6px 0;}
li{margin:4px 0;}
.highlight{background:#fff3cd;padding:2px 6px;border-radius:3px;font-weight:600;}
.delta-pos{color:#27ae60;font-weight:600;}
.delta-neg{color:#e74c3c;font-weight:600;}
.footer{text-align:center;font-size:12px;color:#999;padding:20px;margin-top:10px;}
</style>
</head>
<body>
<div class="container">

<!-- Hero -->
<div class="hero">
  <h1>量化选股机器学习建模 — 最终报告</h1>
  <div class="meta">基于 model_data_stock.csv | 4,281 只股票 × 5 期面板数据 | 2026-07-15</div>
</div>

<!-- 1. 数据概览 -->
<div class="card">
<h2>1. 数据集概览</h2>
<table>
<tr><th>指标</th><th>值</th><th>指标</th><th>值</th></tr>
<tr><td>总样本数</td><td>20,772</td><td>特征数</td><td>17 (8估值+8成长+1市值)</td></tr>
<tr><td>唯一股票数</td><td>4,281 (Code已脱敏)</td><td>目标变量</td><td>Y (bool, t+1期标签)</td></tr>
<tr><td>时间范围</td><td>2021-Q2 ~ 2022-Q2</td><td>缺失值</td><td><span class="badge badge-ok">0</span></td></tr>
<tr><td>全覆盖股票</td><td>3,967 (92.7%)</td><td>有效配对</td><td>16,465 (t→t+1)</td></tr>
</table>

<h3>Y 标签跨期分布</h3>
<table>
<tr><th>时期</th><th>2021-Q2</th><th>2021-Q3</th><th>2021-Q4</th><th>2022-Q1</th><th>2022-Q2</th></tr>
<tr><td>Y=True 占比</td><td><span class="tag tag-red">45.1%</span></td><td><span class="tag tag-red">70.0%</span></td><td><span class="tag tag-green">24.6%</span></td><td><span class="tag tag-red">44.5%</span></td><td><span class="tag tag-green">18.9%</span></td></tr>
</table>
<p style="font-size:13px;color:#888;margin-top:8px;">⚠️ 正例占比波动幅度达 <span class="highlight">51.1 个百分点</span> (18.9%~70.0%)，标签存在显著的时变效应，是本次建模面临的最大挑战。</p>
</div>

<!-- 2. EDA 关键发现 -->
<div class="card">
<h2>2. EDA 关键发现</h2>
<h3>极端值与分布</h3>
<ul>
<li>全部 17 个特征均为厚尾分布，<span class="highlight">14 个偏度 |skew| > 5</span></li>
<li>估值因子（PB +103、PS +88、PE -103）和成长因子（营收同比 +129）极端值严重</li>
<li>解决方案：P1/P99 缩尾 + 4 个特征 log1p 变换，偏度大幅改善</li>
</ul>

<h3>隐藏的共线性</h3>
<table>
<tr><th>特征对</th><th>Spearman (秩相关)</th><th>Pearson (线性)</th><th>现象</th></tr>
<tr><td>净利润同比 ↔ 利润总额同比</td><td><span class="highlight">0.985</span></td><td>0.106</td><td rowspan="4">极端值压低了线性相关<br>缩尾后 VIF 从 ~1 飙升至 10~15</td></tr>
<tr><td>利润总额同比 ↔ 营业利润同比</td><td><span class="highlight">0.981</span></td><td>0.244</td></tr>
<tr><td>净利润同比 ↔ 营业利润同比</td><td><span class="highlight">0.968</span></td><td>0.065</td></tr>
<tr><td>净利润同比 ↔ 基本EPS同比</td><td><span class="highlight">0.968</span></td><td>0.144</td></tr>
</table>
<p style="font-size:13px;color:#888;margin-top:8px;">解决方案：profit_cluster ×4 → PCA 第一主成分 (解释方差 89.1%)，缩尾后 VIF 最大值仅 2.44</p>
</div>

<!-- 3. 预处理流水线 -->
<div class="card">
<h2>3. 预处理流水线</h2>
<table>
<tr><th>步骤</th><th>方法</th><th>拟合范围</th><th>结果</th></tr>
<tr><td>1. 极端值</td><td>Winsorize P1/P99</td><td>仅 Train</td><td>偏度大幅改善</td></tr>
<tr><td>2. 对数变换</td><td>log1p (4个右偏特征)</td><td>—</td><td>PE/PB/PS/MV 偏度 → 可接受范围</td></tr>
<tr><td>3. 特征降维</td><td>PCA (profit_cluster ×4 → 1)</td><td>仅 Train</td><td>第一主成分解释 89.1% 方差</td></tr>
<tr><td>4. 环境特征</td><td>全市场PE中位数 → 二分类</td><td>全截面</td><td>阈值=28.50 (Train中位数)</td></tr>
<tr><td>5. 交互特征</td><td>env × 每个常规特征</td><td>—</td><td>13个常规 → 27个特征(含env+交互)</td></tr>
<tr><td>6. 标准化</td><td>StandardScaler (Z-score)</td><td>仅 Train</td><td>均值≈0, 标准差≈1</td></tr>
</table>
</div>

<!-- 4. 数据划分 -->
<div class="card">
<h2>4. 数据集划分</h2>
<table>
<tr><th>方案</th><th>Train</th><th>Val</th><th>Test</th></tr>
<tr><td>固定划分</td><td>2021-Q2+Q3 (8,088)</td><td>2021-Q4 (4,167)</td><td>2022-Q1 (4,210, 正例19.0%)</td></tr>
<tr><td>扩展窗口 CV (P0)</td>
  <td style="font-size:12px;">W1: Q2<br>W2: Q2+Q3<br>W3: Q2+Q3+Q4</td>
  <td style="font-size:12px;">W1: Q3<br>W2: Q4<br>W3: Q1</td>
  <td style="font-size:12px;">Q1 (固定Test)</td></tr>
</table>
</div>

<!-- 5. 模型结果 -->
<div class="card">
<h2>5. 模型性能对比</h2>

<h3>5.1 固定划分方案 (Phase 3~4)</h3>
<table>
<tr><th>模型</th><th>Val AUC</th><th>Val IC</th><th>Test AUC</th><th>Test IC</th><th>评价</th></tr>
<tr><td>LogisticRegression</td><td><span class="delta-neg">0.4114</span></td><td><span class="delta-neg">-0.1526</span></td><td><span class="delta-pos">0.5802</span></td><td><span class="delta-pos">+0.1089</span></td><td>Test优于Val</td></tr>
<tr><td>XGBoost</td><td><span class="delta-neg">0.4177</span></td><td><span class="delta-neg">-0.1418</span></td><td><span class="delta-pos">0.5745</span></td><td><span class="delta-pos">+0.1012</span></td><td>略优于LR</td></tr>
<tr><td>LightGBM</td><td><span class="delta-neg">0.4185</span></td><td><span class="delta-neg">-0.1404</span></td><td><span class="delta-pos">0.5774</span></td><td><span class="delta-pos">+0.1052</span></td><td>与XGB持平</td></tr>
<tr><td>RandomForest</td><td><span class="delta-neg">0.4687</span></td><td><span class="delta-neg">-0.0539</span></td><td><span class="delta-pos">0.5607</span></td><td><span class="delta-pos">+0.0824</span></td><td>Val最好,Test最差</td></tr>
</table>
<p style="font-size:13px;color:#888;margin-top:8px;">⚠️ 所有模型 Val AUC < 0.5 (Val 环境为高PE转折期，Train未见过)，但 Test AUC > 0.55。</p>

<h3>5.2 P0方案：扩展窗口 CV + 交互特征</h3>
<table>
<tr><th>窗口</th><th>Train</th><th>Train env</th><th>Val env</th><th>LR AUC</th><th>LR IC</th><th>XGB AUC</th><th>XGB IC</th></tr>
<tr><td>W1</td><td>4,000</td><td>全高PE</td><td>低PE</td><td><span class="delta-pos">0.5986</span></td><td><span class="delta-pos">+0.1477</span></td><td>0.5497</td><td>+0.0744</td></tr>
<tr><td>W2</td><td>8,088</td><td>混合</td><td>高PE</td><td><span class="delta-neg">0.4139</span></td><td><span class="delta-neg">-0.1483</span></td><td>0.4541</td><td>-0.0790</td></tr>
<tr><td>W3</td><td>12,255</td><td>混合</td><td>低PE</td><td><span class="delta-pos">0.5701</span></td><td><span class="delta-pos">+0.0952</span></td><td><span class="delta-pos">0.5743</span></td><td><span class="delta-pos">+0.1010</span></td></tr>
<tr style="font-weight:600;background:#f8f9fa;"><td>平均</td><td>—</td><td>—</td><td>—</td><td><span class="delta-pos">0.5275</span></td><td><span class="delta-pos">+0.0315</span></td><td><span class="delta-pos">0.5261</span></td><td><span class="delta-pos">+0.0321</span></td></tr>
</table>
<p style="font-size:13px;color:#888;margin-top:8px;">✅ 扩展窗口CV揭示：环境切换方向决定模型成败。Train有高PE样本→Val低PE时AUC>0.57；反向则AUC<0.45。</p>

<h3>5.3 Bootstrap 集成</h3>
<table>
<tr><th>模型</th><th>AUC</th><th>IC</th><th>Top100</th><th>vs 单模型 Δ</th></tr>
<tr><td>LR (单模型)</td><td>0.5562</td><td>+0.0764</td><td>43.0%</td><td>—</td></tr>
<tr><td>LR (Bootstrap)</td><td>0.5583</td><td>+0.0792</td><td>42.0%</td><td><span class="delta-pos">+0.0021</span></td></tr>
<tr><td>XGBoost (单模型)</td><td>0.5743</td><td>+0.1010</td><td>39.0%</td><td>—</td></tr>
<tr><td>XGBoost (Bootstrap)</td><td>0.5754</td><td>+0.1024</td><td>37.0%</td><td><span class="delta-pos">+0.0011</span></td></tr>
</table>
<p style="font-size:13px;color:#888;margin-top:8px;">Bootstrap 模型间预测相关度 0.91~0.97，多样性不足，提升微乎其微。</p>
</div>

<!-- 6. 模型对比图 -->
<div class="card">
<h2>6. 模型对比可视化</h2>
<div class="chart-box" id="chart-compare"></div>
</div>

<!-- 7. 环境特征分析 -->
<div class="card">
<h2>7. 环境特征分析</h2>
<table>
<tr><th>特征</th><th>定义</th><th>类型</th><th>阈值</th></tr>
<tr><td><code>env_market_pe</code></td><td>全市场PE(TTM,扣非)中位数二值化</td><td>二分类 (0/1)</td><td>Train中位数 = 28.50</td></tr>
</table>

<h3>各期取值</h3>
<table>
<tr><th>日期</th><th>PE中位数</th><th>env_market_pe</th><th>划分</th><th>Y正例占比</th></tr>
<tr><td>2021-06-30</td><td>29.04</td><td><span class="tag tag-red">1 (高PE)</span></td><td>Train</td><td>→ 69.99%</td></tr>
<tr><td>2021-09-30</td><td>27.96</td><td><span class="tag tag-green">0 (低PE)</span></td><td>Train</td><td>→ 24.56%</td></tr>
<tr><td>2021-12-31</td><td>31.26</td><td><span class="tag tag-red">1 (高PE)</span></td><td>Val</td><td>→ 44.47%</td></tr>
<tr><td>2022-03-31</td><td>27.66</td><td><span class="tag tag-green">0 (低PE)</span></td><td>Test</td><td>→ 18.93%</td></tr>
</table>

<h3>特征重要性 (XGBoost, W3)</h3>
<p style="font-size:13px;"><span class="highlight">env_market_pe 重要性: 50.2%</span> — 远超第二名 MV (7.9%)，证实环境特征是模型中占主导地位的信号。</p>
<p style="font-size:13px;">交互特征 <code>PS×env</code> (16.8%) 和 <code>PB×env</code> (5.3%) 也进入 Top-5，说明市销率和市净率在高低PE环境下对Y的影响方向不同。</p>
</div>

<!-- 8. 方法论演进 -->
<div class="card">
<h2>8. 方法论演进路径</h2>
<table>
<tr><th>版本</th><th>方案</th><th>CV AUC</th><th>Val AUC</th><th>Test AUC</th><th>关键变化</th></tr>
<tr><td>v1</td><td>基础LR (13特征, 固定划分)</td><td>0.5705</td><td><span class="delta-neg">0.4114</span></td><td><span class="delta-pos">0.5802</span></td><td>baseline</td></tr>
<tr><td>v2</td><td>+ env_market_pe (14特征)</td><td><span class="delta-pos">0.7895</span></td><td><span class="delta-neg">0.4018</span></td><td><span class="delta-pos">0.5725</span></td><td>环境特征，CV AUC 大幅提升</td></tr>
<tr><td>v3 (P0)</td><td>+ 交互特征 + 扩展窗口CV</td><td>—</td><td><span class="delta-pos">0.5275 (avg)</span></td><td><span class="delta-pos">0.5743</span></td><td>评估更稳健，Top100=39%</td></tr>
<tr><td>v4</td><td>+ Bootstrap集成</td><td>—</td><td><span class="delta-pos">0.5754</span></td><td>—</td><td>提升微乎其微 (+0.001)</td></tr>
</table>
</div>

<!-- 9. 最终推荐 -->
<div class="card">
<h2>9. 最终推荐模型</h2>

<h3>最佳配置</h3>
<table>
<tr><th>组件</th><th>选择</th></tr>
<tr><td>模型</td><td><strong>XGBoost</strong> (n=100, depth=3, lr=0.05)</td></tr>
<tr><td>特征</td><td>27个 (13常规 + 1环境 + 13交互)</td></tr>
<tr><td>验证</td><td>扩展窗口 CV (3窗口) + 固定 Test</td></tr>
<tr><td>预处理</td><td>Winsorize(P1/P99) → Log1p → PCA → env → 交互 → StandardScaler</td></tr>
</table>

<h3>Test 集表现</h3>
<table>
<tr><th>AUC</th><th>IC</th><th>Top100 Prec</th><th>Top200 Prec</th><th>Top500 Prec</th></tr>
<tr><td><strong>0.5743</strong></td><td><strong>+0.1010</strong></td><td><strong>39.0%</strong></td><td><strong>36.5%</strong></td><td><strong>30.4%</strong></td></tr>
</table>
<p style="font-size:13px;color:#888;margin-top:8px;">Test 正例基线 19.0%，Top100 命中率 39.0% = <span class="highlight">基线 × 2.05</span>，Top500 命中率 30.4% = 基线 × 1.60。</p>
</div>

<!-- 10. 局限性与建议 -->
<div class="card">
<h2>10. 局限性与未来建议</h2>

<h3>已知限制</h3>
<ul>
<li><span class="badge badge-warn">样本量</span> 仅 5 个季度截面，Train 最多 3 期，模型泛化能力受限于时间维度</li>
<li><span class="badge badge-warn">标签漂移</span> Y 正例占比 19%~70%，Train/Test 分布不一致，AUC 指标可能低估真实排序能力</li>
<li><span class="badge badge-warn">Code 脱敏</span> 无行业标签，无法做行业中性化</li>
<li><span class="badge badge-warn">环境特征</span> 仅用 PE 中位数，可能遗漏其他维度的市场状态信号</li>
</ul>

<h3>未来增强方向</h3>
<ul>
<li><strong>更多时间序列</strong>：扩展到 8~12 个季度，使时序模型（LSTM、滚动窗口）可行</li>
<li><strong>行业中性化</strong>：获取行业标签后消除行业偏差</li>
<li><strong>排序模型</strong>：将二分类改为 LambdaRank，直接优化排序质量，对标签漂移更鲁棒</li>
<li><strong>宏观因子</strong>：引入利率、PMI、社融等外部变量作为全局特征</li>
<li><strong>因子衰减分析</strong>：追踪因子 IC 的时变性，识别失效信号并动态淘汰</li>
</ul>
</div>

<div class="footer">
  TASK5 量化选股 ML 建模项目 · 最终报告 · 2026-07-15<br>
  脚本目录: TASK5/ | 产出目录: TASK5/outputs/
</div>

</div><!-- container -->

<script>
// 模型对比图
(function(){
  var c=echarts.init(document.getElementById('chart-compare'));
  c.setOption({
    tooltip:{trigger:'axis'},
    legend:{data:['Val AUC','Test AUC','Val IC','Test IC'],bottom:0},
    grid:{left:'8%',right:'8%',top:'8%',bottom:'12%'},
    xAxis:{type:'category',
      data:['LR\n(基础)','LR\n(+env)','LR\n(扩窗CV)','XGBoost\n(+env)','XGBoost\n(扩窗CV)','XGBoost\n(Bootstrap)']},
    yAxis:[
      {type:'value',name:'AUC',min:0.35,max:0.65,axisLabel:{formatter:'{value}'}},
      {type:'value',name:'IC',min:-0.2,max:0.2,axisLabel:{formatter:'{value}'}}
    ],
    series:[
      {name:'Val AUC',type:'bar',data:[0.4114,0.4018,0.5275,0.4177,0.5261,0.5754],
       itemStyle:{color:'rgba(52,152,219,0.7)'},barWidth:'40%'},
      {name:'Test AUC',type:'bar',data:[0.5802,0.5725,'-',0.5745,'-','-'],
       itemStyle:{color:'rgba(231,76,60,0.7)'},barWidth:'40%'},
      {name:'Val IC',type:'line',yAxisIndex:1,data:[-0.1526,-0.1691,0.0315,-0.1418,0.0321,0.1024],
       lineStyle:{color:'#f39c12',width:2.5},symbol:'circle',symbolSize:8},
      {name:'Test IC',type:'line',yAxisIndex:1,data:[0.1089,0.0985,'-',0.1012,'-','-'],
       lineStyle:{color:'#27ae60',width:2.5},symbol:'diamond',symbolSize:8}
    ]
  });
  window.addEventListener('resize',function(){c.resize();});
})();
</script>
</body>
</html>"""

with open(os.path.join(OUTPUT_DIR, 'final_report.html'), 'w', encoding='utf-8') as f:
    f.write(html)

print("  -> outputs/final_report.html")
print("Phase 5 完成")
