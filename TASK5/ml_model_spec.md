# 量化选股 ML 建模 Spec

> **版本**: 3.0.0 | **日期**: 2026-07-14 | **状态**: Phase 0 已完成 → 下一步 Phase 1

---

## 1. 问题定义

| 维度 | 说明 |
|------|------|
| **任务类型** | 二分类 |
| **目标变量** | `Y`（bool），**已确认为未来一期标签（t+1），可直接用作因变量** |
| **正例含义** | 满足选股条件（具体定义由数据提供方确定） |
| **正例占比** | 40.39%（True:False ≈ 4:6），轻度不平衡，无需重采样 |
| **预测场景** | 在每个季度末，基于当期财务因子，预测下一期 Y 的概率 |
| **应用目标** | 量化因子选股——对全市场股票打分排序，精选 Top-K 构建组合 |

### 关键假设

- **无前视偏差**：Y 是 t+1 期标签，建模时严格使用 t 期特征 → t+1 期标签
- **无幸存者偏差**：数据包含退市股（Code 1~4 出现次数不完整的 314 只股票）
- **Y 定义一致性**：所有股票的 Y 标签基于同一套规则生成

---

## 2. 数据概览

### 2.1 基础信息

| 指标 | 值 |
|------|-----|
| 文件 | `TASK5/model_data_stock.csv` |
| 样本数 | 20,772 |
| 列数 | 20（含 Date、Code、Y） |
| 特征数 | 17（8 估值 + 8 成长 + 1 市值） |
| 股票数 | 4,281（Code 已脱敏为 1~605,599 之间的整数） |
| 时间范围 | 2021-06-30 ~ 2022-06-30（5 个季度末截面） |
| 全覆盖股票 | 3,967 只（92.7%，全部 5 期均有数据） |
| 部分覆盖股票 | 314 只（次新股或退市股，仅出现 1~4 次） |
| 缺失值 | 0（全部完整） |
| 重复行 | 0 |

### 2.2 各截面股票数

| 日期 | 股票数 |
|------|--------|
| 2021-06-30 | 4,011 |
| 2021-09-30 | 4,098 |
| 2021-12-31 | 4,173 |
| 2022-03-31 | 4,228 |
| 2022-06-30 | 4,262 |

股票数逐季递增，反映新股上市。Y 作为 t+1 标签，有效的特征-标签对为前 4 期（约 16,500 条有效样本），第 5 期仅有特征、无标签。

### 2.3 Y 标签跨期分布

| 日期 | 股票数 | Y=True 占比 |
|------|--------|------------|
| 2021-06-30 | 4,011 | 45.10% |
| 2021-09-30 | 4,098 | 69.99% |
| 2021-12-31 | 4,173 | 24.56% |
| 2022-03-31 | 4,228 | 44.47% |
| 2022-06-30 | 4,262 | 18.93% |

> **关键发现**：Y 正例占比在 18.9%~70.0% 之间剧烈波动，波动幅度约 51 个百分点。这意味着标签本身存在显著的时间效应——某些季度"好股票"的标准可能完全不同。建模时需关注时间泛化能力。

---

## 3. 描述性统计与可视化分析 ✅

> **Phase 0 已完成**。EDA 阶段独立于建模流程，使用全部原始数据（不划分、不缩尾）。脚本：`TASK5/phase0_eda.py`。

### 3.0 可视化技术说明

所有图表使用 **ECharts 5.5** 渲染，Python 端生成 HTML 文件。经过 4 轮迭代，沉淀以下技术约定：

| 场景 | 方案 | 原因 |
|------|------|------|
| 单图页面（热力图、VIF、目标变量、点双列相关） | `raw_js_page()` 模板 — option 以原生 JS 写入，`formatter` 直接写 JS 函数 | `json.dumps` 无法序列化函数，ECharts 的 `label.formatter` 需要 JS 函数才能正确渲染数值 |
| 多图网格（直方图、箱线图、KDE、散点图） | `multi_chart_page()` 模板 — CSS Grid 布局，每张图独立 ECharts 实例（280~400px 固定高度） | ECharts grid 子图系统在多特征场景下坐标计算复杂且容易溢出，独立实例更稳定 |
| 热力图着色 | 数据必须为 3 元素 `[x, y, value]` | `visualMap` 默认映射第 3 维；4 元素数组（含 label 字符串）会导致着色失效 |
| 散点图性能 | 随机采样至 1500 点/对 | 全量 2 万点导致 HTML 体积 4.2MB，采样后仅 391KB |

### 3.1 描述性统计输出

对全部 17 个数值特征（不含 Date、Code、Y），输出以下统计量表：

| 统计量 | 说明 |
|--------|------|
| count | 有效样本数 |
| mean | 均值 |
| std | 标准差 |
| min | 最小值 |
| P1 / P5 / P25 / P50 / P75 / P95 / P99 | 分位数 |
| max | 最大值 |
| skew | 偏度（接近 0 = 对称，> 2 = 严重右偏，< -2 = 严重左偏） |
| kurtosis | 峰度（> 3 = 厚尾，远大于 3 = 极端值严重） |
| IQR | 四分位距（P75 - P1），衡量离散度 |
| CV | 变异系数（std / mean），跨特征可比离散度 |

**输出格式**：`TASK5/outputs/descriptive_stats.csv` + Markdown 表格嵌入最终报告

### 3.2 目标变量分析

| 分析项 | 实现 |
|--------|------|
| Y 整体分布 | 环形图，红色=True(40.39%)，绿色=False(59.61%) |
| Y 按日期分布 | **柱状图**（非折线图），Y 轴自适应范围（15% padding），橙色虚线标注均值 |

> Y 正例占比跨度极大（18.9%~70.0%），标签存在显著时间效应。

**图表**：`TASK5/outputs/eda_target_distribution.html`（700px 固定高度，单页适配）

### 3.3 特征分布可视化

#### 3.3.1 直方图 + KDE 密度曲线

每个特征一张 280px 独立图表，4 列 CSS Grid 布局：
- 蓝色半透明直方图（40 bins）+ 红色 KDE 密度曲线
- 蓝色虚线标注均值（μ），橙色虚线标注中位数（M，置于竖线内侧避免与标题重叠）
- 标题栏显示偏度/峰度值
- X 轴裁剪至 [P1, P99]

**图表**：`TASK5/outputs/eda_dist_histograms.html`

#### 3.3.2 箱线图 — 跨期分组对比

每个特征一张 280px 独立图表，4 列 CSS Grid 布局：
- 5 期分组箱线，Y 轴裁剪至 [P1, P99]
- 红色虚线连接各期中位数

**图表**：`TASK5/outputs/eda_boxplot_by_date.html`

#### 3.3.3 极端值热力图

单页 750px，raw_js_page 模板：
- 列：P0.1 / P1 / P25 / P50 / P75 / P99 / P99.9 / 超P1-P99%
- 颜色：绿→白→橙→红渐变
- 数值以 JS formatter 函数直接渲染

**图表**：`TASK5/outputs/eda_outlier_heatmap.html`

### 3.4 相关性分析

#### 3.4.1 Pearson & Spearman 双值热力图

单页 850px，raw_js_page 模板：
- **数据格式**：3 元素 `[x, y, Spearman值]` 用于 visualMap 着色，Pearson 值存入独立 `PEARSON_MAP` JS 对象
- 下三角矩阵，颜色：蓝(-1)→白(0)→红(+1)
- 每格两行数值：上行 Spearman，下行 Pearson（label formatter 通过 `PEARSON_MAP` 反查）
- tooltip 显示完整 4 位小数

**图表**：`TASK5/outputs/eda_correlation_heatmap.html`

#### 3.4.2 特征对散点图矩阵

每个高相关对一张 380px 独立图表，2~3 列 CSS Grid，原始数据随机采样至 1500 点/对：
- 红色 = Y=True，绿色 = Y=False
- 右上角标注 Spearman/Pearson 值（避免与 Y 轴名称重叠）
- X/Y 轴裁剪至 [P1, P99]

**重点分析对**：
1. 净利润同比 vs 利润总额同比（S=0.985, P=0.106）
2. 利润总额同比 vs 营业利润同比（S=0.981, P=0.244）
3. 净利润同比 vs 营业利润同比（S=0.968, P=0.065）
4. 净利润同比 vs 基本EPS同比（S=0.968, P=0.144）
5. 利润总额同比 vs 基本EPS同比（S=0.956, P=0.269）
6. 基本EPS同比 vs 营业利润同比（S=0.941, P=0.191）
7. PE(TTM) vs PE(TTM,扣非)（S=0.719, P=0.007）

**图表**：`TASK5/outputs/eda_scatter_pairs.html`

#### 3.4.3 VIF 缩尾前后对比

单页 700px，raw_js_page 模板：
- 水平条形图，灰色=缩尾前，蓝色=缩尾后
- 橙色虚线 VIF=5，红色虚线 VIF=10
- markLine 嵌套在蓝色 series 内部确保正确渲染

**关键发现**：缩尾后 VIF 从 ~1 跃升至 10~15（利润总额 VIF=14.8, 营业利润 VIF=10.3），证实共线性被极端值掩盖。

**图表**：`TASK5/outputs/eda_vif_barchart.html`

### 3.5 特征与目标变量的关系

#### 3.5.1 分组 KDE 密度曲线

每个特征一张 280px 独立图表，4 列 CSS Grid：
- 红色密度区域 = Y=True，绿色 = Y=False
- 标题栏显示 KS 统计量（两样本 Kolmogorov-Smirnov 检验，值越大区分力越强）

**图表**：`TASK5/outputs/eda_feature_vs_target.html`

#### 3.5.2 特征-目标相关系数排名

单页 650px，raw_js_page 模板：
- 水平柱状图，点双列相关系数（Point-Biserial Correlation）
- 红色=正相关，绿色=负相关，按 |r| 降序排列

**图表**：`TASK5/outputs/eda_feature_target_corr.html`

---

## 4. 特征目录

### 4.1 标识列（不参与建模）

| 列名 | 类型 | 用途 |
|------|------|------|
| `Date` | str | 时间索引，分组交叉验证的分组键之一 |
| `Code` | int64 | 股票标识（已脱敏），分组键，防止同股票跨集泄漏 |

### 4.2 估值因子（8 个）

| # | 列名 | 偏度 | 极端值 | 处理方案 |
|---|------|------|--------|----------|
| 1 | `企业倍数(EV除EBITDA)` | +82.1 | 严重 | winsorize → log1p |
| 2 | `市净率PB(MRQ)` | +103.1 | 严重 | winsorize → log1p |
| 3 | `市现率PCF(现金净流量TTM)` | -25.8 | 严重 | winsorize（含大量负值，不做 log） |
| 4 | `市现率PCF(经营现金流TTM)` | +1.7 | 中度 | winsorize |
| 5 | ~~`市盈率PE(TTM)`~~ | -102.8 | 严重 | **剔除**（与扣非PE冗余，且负PE含义模糊） |
| 6 | `市盈率PE(TTM,扣除非经常性损益)` | -26.2 | 严重 | winsorize（保留，排除一次性损益干扰） |
| 7 | `市销率PS(TTM)` | +87.8 | 严重 | winsorize → log1p |
| 8 | `股息率(近12个月)` | +4.7 | 中度 | winsorize |

### 4.3 成长因子（8 个）—— 含 profit_cluster

| # | 列名 | 偏度 | 极端值 | 所属簇 |
|---|------|------|--------|--------|
| 9 | `净利润同比增长率` | -61.4 | 严重 | **profit_cluster** |
| 10 | `利润总额(同比增长率)` | +41.6 | 严重 | **profit_cluster** |
| 11 | `营业利润(同比增长率)` | -67.5 | 严重 | **profit_cluster** |
| 12 | `基本每股收益(同比增长率)` | -2.2 | 中度 | **profit_cluster** |
| 13 | `净资产同比增长率` | -0.4 | 可接受 | — |
| 14 | `总资产同比增长率` | +20.7 | 严重 | — |
| 15 | `现金净流量同比增长率` | -42.6 | 严重 | — |
| 16 | `营业总收入(同比增长率)` | +129.2 | 严重 | — |

#### profit_cluster 共线性诊断

| 特征对 | Spearman | Pearson |
|--------|----------|---------|
| 净利润同比 ↔ 利润总额同比 | **0.985** | 0.106 |
| 利润总额同比 ↔ 营业利润同比 | **0.981** | 0.244 |
| 净利润同比 ↔ 营业利润同比 | **0.968** | 0.065 |
| 净利润同比 ↔ 基本EPS同比 | **0.968** | 0.144 |
| 利润总额同比 ↔ 基本EPS同比 | **0.956** | 0.269 |
| 基本EPS同比 ↔ 营业利润同比 | **0.941** | 0.191 |

> **关键发现**：Spearman 全部 > 0.94（近乎完美的单调关系），但 Pearson 仅 0.06~0.27。极端值把线性关系彻底冲毁了。缩尾后 Pearson 会急剧上升，VIF 可能从当前的 ~1 跃升至数十。

**处理方案：方案 B — PCA 降维。** 对 profit_cluster 的 4 个因子做 PCA，取第一主成分 `profit_pc1` 替代原始 4 个特征。第一主成分预期解释 90%+ 的方差。

### 4.4 规模因子（1 个）

| # | 列名 | 偏度 | 处理方案 |
|---|------|------|----------|
| 17 | `MV`（总市值） | +15.4 | winsorize → log1p |

### 4.5 特征构建总览

```
原始 17 个特征
  ├─ 剔除 PE(TTM)（与扣非PE冗余）                     →  -1
  ├─ profit_cluster ×4 → PCA 第一主成分 profit_pc1     →  -4, +1
  └─ 保留 12 个独立特征
──────────────────────────────────────────────────────
  最终特征数: 13
```

| 构建后特征 | 来源 | 变换 |
|------------|------|------|
| `企业倍数(EV除EBITDA)_log` | 估值 | winsorize + log1p |
| `市净率PB(MRQ)_log` | 估值 | winsorize + log1p |
| `市现率PCF(现金净流量TTM)` | 估值 | winsorize |
| `市现率PCF(经营现金流TTM)` | 估值 | winsorize |
| `市盈率PE(TTM,扣非)` | 估值 | winsorize |
| `市销率PS(TTM)_log` | 估值 | winsorize + log1p |
| `股息率(近12个月)` | 估值 | winsorize |
| `profit_pc1` | 成长（PCA） | winsorize 后 PCA |
| `净资产同比增长率` | 成长 | winsorize |
| `总资产同比增长率` | 成长 | winsorize |
| `现金净流量同比增长率` | 成长 | winsorize |
| `营业总收入(同比增长率)` | 成长 | winsorize |
| `MV_log` | 规模 | winsorize + log1p |

---

## 5. 数据集划分

### 5.1 主方案：时序分组划分

由于 Y 是 t+1 期标签，有效样本为前 4 期：

| 集合 | 特征日期（X） | 标签日期（Y） | 有效样本数（约） |
|------|-------------|-------------|-----------------|
| **Train** | 2021-Q2, 2021-Q3 | 2021-Q3, 2021-Q4 | ~8,100 |
| **Val** | 2021-Q4 | 2022-Q1 | ~4,100 |
| **Test** | 2022-Q1 | 2022-Q2 | ~4,200 |

**划分规则**：

- `group_key = Code`：同一只股票的所有记录强制在同一集合中
- 严格按时间顺序：Train 最早 → Val 中间 → Test 最晚
- 所有预处理统计量（缩尾分位数、标准化均值/标准差、PCA 载荷）**仅在 Train 上拟合**，再应用到 Val/Test

### 5.2 备选方案：扩展窗口 CV

| 窗口 | Train | Val |
|------|-------|-----|
| W1 | 2021-Q2 | 2021-Q3 |
| W2 | 2021-Q2~Q3 | 2021-Q4 |
| W3 | 2021-Q2~Q4 | 2022-Q1 |

> 主方案选 5.1；如过拟合严重或 Val 性能波动大，切换到 5.2 做稳健性检验。

---

## 6. 预处理流水线

### Step 0 — 数据集划分

按 §5.1 方案执行，输出 `X_train / X_val / X_test` 及对应标签 `y_train / y_val / y_test`。

### Step 1 — 极端值缩尾（Winsorize）

| 参数 | 值 |
|------|-----|
| 方法 | 百分位数缩尾 |
| 下界 | P1（1%） |
| 上界 | P99（99%） |
| 拟合范围 | **仅 Train 集**，再应用到 Val/Test |
| 独立性 | 每个特征独立计算分位数 |

输出：缩尾前后各特征 `[min, max]` 对比表 + 偏度变化。

### Step 2 — 对数变换

对右偏严重的特征做 `log1p(x) = log(x + 1)` 变换：

| 特征 | 变换前偏度 | 预期变换后偏度 |
|------|-----------|---------------|
| 企业倍数(EV除EBITDA) | +82.1 | < +5 |
| 市净率PB(MRQ) | +103.1 | < +5 |
| 市销率PS(TTM) | +87.8 | < +5 |
| MV | +15.4 | < +3 |

负值特征（PCF、PE）**不参与 log 变换**。

### Step 3 — 特征构建（PCA）

对 profit_cluster 的 4 个特征（缩尾后）执行 PCA：

1. **仅在 Train 集上**拟合 PCA，提取第一主成分 `profit_pc1`
2. 将拟合好的 PCA 应用到 Val 和 Test
3. 报告第一主成分的解释方差比 + 各因子载荷

| 原始特征 | 预期载荷 |
|----------|---------|
| 净利润同比增长率 | ~0.50 |
| 利润总额(同比增长率) | ~0.50 |
| 营业利润(同比增长率) | ~0.50 |
| 基本每股收益(同比增长率) | ~0.50 |

> 4 个因子几乎等权，`profit_pc1` 可理解为"综合盈利增长因子"。

构建后：原始 4 列替换为 1 列 `profit_pc1`。

### Step 4 — 缺失值处理

当前数据集无缺失值，此步骤保留为空壳，以防未来数据引入缺失。

策略预案：按 Date 分组，用同截面中位数填充。

### Step 5 — 标准化（StandardScaler）

| 参数 | 值 |
|------|-----|
| 方法 | Z-score（均值 0，标准差 1） |
| 拟合范围 | **仅 Train 集** |

标准化后验证：Train 集每列 mean ≈ 0，std ≈ 1。

### Step 6 — 市值中性化（可选，默认关闭）

对每个截面日期，将因子值对 `log(MV)` 回归，取残差。

由于 Code 已脱敏、无行业标签，仅能做出市值中性化。当前默认为关闭状态。

---

## 7. 模型候选

### 7.1 逻辑回归 — Baseline（P0 必做）

| 属性 | 说明 |
|------|------|
| 类型 | 线性分类器 |
| 可解释性 | 高 — 系数即因子权重，直接可读 |
| 优势 | 训练快、天然概率输出、量化研报标准 |
| 劣势 | 仅线性、对残余异常值敏感 |

**超参数搜索空间**：

| 超参数 | 候选值 |
|--------|--------|
| `C`（正则化强度倒数） | 0.01, 0.1, 1.0, 10.0 |
| `penalty` | L1, L2 |
| `class_weight` | balanced, null |

预期 AUC：0.55~0.65。

### 7.2 XGBoost（P0 必做）

| 属性 | 说明 |
|------|------|
| 类型 | 梯度提升树（GBDT） |
| 可解释性 | 中 — 特征重要性 + SHAP |
| 优势 | 非线性交互、对异常值稳健、竞赛常胜 |
| 劣势 | 超参数多、需仔细调参 |

**超参数搜索空间**：

| 超参数 | 候选值 |
|--------|--------|
| `n_estimators` | 100, 200, 500 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `subsample` | 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `reg_alpha` | 0, 0.1, 1.0 |
| `reg_lambda` | 1.0, 5.0, 10.0 |
| `scale_pos_weight` | 1.0, 1.5 |

预期 AUC：0.60~0.70。

### 7.3 LightGBM（P1）

| 属性 | 说明 |
|------|------|
| 类型 | 梯度提升树（leaf-wise） |
| 优势 | 比 XGBoost 更快、leaf-wise 更高效 |
| 劣势 | 小数据易过拟合 |

**超参数搜索空间**：

| 超参数 | 候选值 |
|--------|--------|
| `n_estimators` | 100, 200, 500 |
| `max_depth` | 3, 5, 7, -1 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `num_leaves` | 15, 31, 63 |
| `subsample` | 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `reg_alpha` | 0, 0.1, 1.0 |
| `reg_lambda` | 0, 1.0, 5.0 |

### 7.4 随机森林（P1 — 对照）

| 属性 | 说明 |
|------|------|
| 类型 | Bagging 集成 |
| 优势 | 不易过拟合、对超参数不敏感 |
| 劣势 | 金融数据上通常弱于 Boosting |

**超参数搜索空间**：

| 超参数 | 候选值 |
|--------|--------|
| `n_estimators` | 100, 200, 500 |
| `max_depth` | 5, 10, 15, null |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 5 |

### 7.5 MLP 神经网络（P2 — 可选探索）

- 隐藏层架构：`[32]` / `[64, 32]` / `[128, 64, 32]`
- 激活函数：ReLU, Tanh
- 正则化：alpha ∈ [0.0001, 0.001, 0.01]

> 数据量仅 ~1.6 万训练样本，神经网络优势有限。作为探索性对比。

---

## 8. 评估指标

### 8.1 核心指标

| 指标 | 公式 / 含义 | 好 | 优秀 |
|------|------------|-----|------|
| **AUC-ROC** | 模型整体排序能力 | > 0.65 | > 0.75 |
| **Precision@TopK** | Top 100/200/500 中正例比例 | > 45% | > 55% |
| **IC (Rank IC)** | 每期 Spearman(pred, Y_true) 的均值 | > 0.03 | > 0.05 |
| **ICIR** | IC 均值 / IC 标准差 | > 0.5 | > 1.0 |

### 8.2 辅助指标

| 指标 | 说明 |
|------|------|
| F1 Score | 精确率与召回率的调和平均 |
| KS 值 | 模型区分正负样本的最大间隔 |

### 8.3 稳健性检查

| 检查项 | 阈值 |
|--------|------|
| Train/Val AUC 差距 | \|gap\| < 0.05 可接受 |
| IC 时间衰减 | IC 不应随测试时间推移而显著下降 |
| 特征重要性稳定性 | 不同时期 Top-5 特征是否一致 |

---

## 9. 执行计划

### Phase 0 — 探索性数据分析（EDA）✅ 已完成

**脚本**：`TASK5/phase0_eda.py`（经 4 轮迭代：v1 初始 → v2 CSS Grid 重构 → v3 JS formatter 修复 → v4 热力图着色 + 目标变量图表修正）

**独立于建模流程，使用全部原始数据（不划分、不缩尾），以图表为主。**

- [x] **0a — 描述性统计**：生成全特征统计表（mean/std/skew/kurtosis/分位数/IQR/CV）
- [x] **0b — 目标变量可视化**：环形图（Y 整体分布）+ 柱状图（Y 正例占比按日期趋势，均值线标注）
- [x] **0c — 特征分布可视化**：
  - 直方图 + KDE（17 特征，CSS Grid 4 列独立实例）
  - 箱线图按日期分组（观察分布漂移）
  - 极端值热力图（P0.1/P1/P25/P50/P75/P99/P99.9/超P1-P99%）
- [x] **0d — 相关性分析**：
  - Spearman+Pearson 双值热力图（3 元素数据 + PEARSON_MAP 反查）
  - 高相关特征对散点图矩阵（7 对，随机采样 1500 点/对）
  - VIF 缩尾前/后对比条形图（markLine 嵌入 series 内）
- [x] **0e — 特征-目标关系**：
  - 分组 KDE 密度曲线（Y=True vs Y=False，KS 统计量标注）
  - 点双列相关系数排名柱状图

**产出**：

| 文件 | 内容 |
|------|------|
| `TASK5/outputs/descriptive_stats.csv` | 描述性统计表 |
| `TASK5/outputs/eda_target_distribution.html` | 目标变量分布图 |
| `TASK5/outputs/eda_dist_histograms.html` | 直方图 + KDE |
| `TASK5/outputs/eda_boxplot_by_date.html` | 跨期箱线图 |
| `TASK5/outputs/eda_outlier_heatmap.html` | 极端值热力图 |
| `TASK5/outputs/eda_correlation_heatmap.html` | Pearson + Spearman 热力图 |
| `TASK5/outputs/eda_scatter_pairs.html` | 高相关对散点图矩阵 |
| `TASK5/outputs/eda_vif_barchart.html` | VIF 条形图 |
| `TASK5/outputs/eda_feature_vs_target.html` | 分组密度曲线 |
| `TASK5/outputs/eda_feature_target_corr.html` | 特征-目标相关系数 |

### Phase 1 — 数据加载与划分

- [ ] 读取 `model_data_stock.csv`
- [ ] 确认数据无缺失、无重复
- [ ] 构建特征-标签对：X(t) → Y(t+1)
- [ ] 按 §5.1 方案划分 Train/Val/Test（group_key=Code）
- [ ] 输出各集合样本量和正例占比

**产出**：划分统计报告 `TASK5/outputs/data_split_report.txt`

### Phase 2 — 预处理

- [ ] Step 1a — Winsorize（缩尾前后对比表 + 偏度变化）
- [ ] Step 1b — Log 变换（偏度变化对比）
- [ ] Step 1c — PCA 构建 profit_pc1（方差解释比 + 载荷报告）
- [ ] Step 1d — 缩尾后重算 Spearman/Pearson/VIF（与 Phase 0 对比）
- [ ] Step 1e — StandardScaler
- [ ] 确认最终特征维度 = 13

**产出**：

| 文件 | 内容 |
|------|------|
| `TASK5/outputs/winsorize_comparison.csv` | 缩尾前后对比 |
| `TASK5/outputs/skew_comparison.csv` | 偏度变化对比 |
| `TASK5/outputs/pca_report.json` | PCA 方差解释 + 载荷 |
| `TASK5/outputs/corr_after_winsorize.csv` | 缩尾后相关性矩阵 |
| `TASK5/outputs/vif_after_winsorize.csv` | 缩尾后 VIF 表 |

### Phase 3 — 逻辑回归 Baseline

- [ ] GridSearchCV 调参（C, penalty, class_weight）
- [ ] 输出系数表（方向是否符合金融直觉）
- [ ] Val 集 AUC / IC / Precision@TopK
- [ ] ROC 曲线

**产出**：`lr_coefficients.csv` + `roc_curve.html` + Val 指标

### Phase 4 — 树模型

- [ ] XGBoost + GridSearch 调参
- [ ] LightGBM + GridSearch 调参
- [ ] 随机森林（对照）
- [ ] 特征重要性排名
- [ ] 对比全部模型的 Val 性能

**产出**：`model_comparison.csv` + `feature_importance.html` + 超参数搜索结果

### Phase 5 — 测试集评估

- [ ] 选定 Val 最佳模型
- [ ] Test 集完整评估（AUC / IC / Precision@TopK / ICIR）
- [ ] 按日期拆分的 IC 序列（观察衰减）
- [ ] 生成最终报告

**产出**：`test_evaluation.json` + `test_predictions.csv` + `final_report.html`

### Phase 6 — 模型解释（P2，可选）

- [ ] SHAP 特征重要性 + 特征效应方向
- [ ] 部分依赖图（PDP）
- [ ] 单样本预测解释（随机选 5 只股票）

---

## 10. 配置

```yaml
random_seed: 42
cv_folds: 5
n_jobs: -1
verbose: true

# 预处理
outlier_method: winsorize
winsorize_percentiles: [0.01, 0.99]
log_transform_features:
  - 企业倍数(EV除EBITDA)
  - 市净率PB(MRQ)
  - 市销率PS(TTM)
  - MV
pca_features: [净利润同比增长率, 利润总额(同比增长率), 营业利润(同比增长率), 基本每股收益(同比增长率)]
pca_n_components: 1
neutralization_enabled: false  # 市值中性化，默认关闭

# 特征
dropped_features:
  - 市盈率PE(TTM)  # 与扣非PE冗余
  - Date           # 不参与建模
  - Code           # 仅用于分组，不参与建模
final_feature_count: 13

# 验证
train_dates: [2021-06-30, 2021-09-30]
val_dates: [2021-12-31]
test_dates: [2022-03-31]
group_key: Code
```

---

## 11. 已知风险 & 缓解措施

| 风险 | 影响 | 缓解 |
|------|------|------|
| **Code 脱敏，无行业标签** | 无法做行业中性化、无法验证个股 | 后续获取行业标签后启用中性化 |
| **仅 5 个时间点** | 模型可能过度拟合 2021 年风格 | 扩展窗口 CV 做稳健性检验；输出 IC 时间序列观察衰减 |
| **Y 的具体定义未公开** | 无法判断模型学到的信号是否具有经济意义 | 通过逻辑回归系数方向做直觉检验 |
| **幸存者偏差** | 数据可能剔除了退市股 | 已确认 314 只不完整覆盖股票存在，默认保留 |
| **ECharts JSON 序列化限制** | `json.dumps` 无法序列化 JS 函数，导致 heatmap label formatter 失效 | 热力图类图表使用 `raw_js_page()` 模板，formatter 以原生 JS 写入 HTML；简单图表使用 ECharts 模板语法 `'{c}'` 替代函数 |
| **多图网格布局不稳定** | ECharts grid 子图坐标在多特征场景下容易溢出或重叠 | 改用 CSS Grid + 独立 ECharts 实例方案（`multi_chart_page()`），每图固定高度 |

---

## 12. 未来增强

- 添加行业分类 → 行业中性化 + 行业内 Ranking
- 扩展时间范围到更多季度
- 加入宏观因子（利率、PMI、CPI）
- 引入市场状态（牛/熊分位数）作为交互特征
- 分层建模：先行业分类、再行业内选股
- 回测框架：从 Top-K 推荐到模拟组合净值曲线

---

## 13. 产出物清单

### EDA 阶段（Phase 0）

| 文件 | 类型 | 说明 |
|------|------|------|
| `TASK5/outputs/descriptive_stats.csv` | 数据表 | 描述性统计 |
| `TASK5/outputs/eda_target_distribution.html` | 可视化 | Y 分布（环形图 + 柱状图） |
| `TASK5/outputs/eda_dist_histograms.html` | 可视化 | 直方图 + KDE（17特征分面） |
| `TASK5/outputs/eda_boxplot_by_date.html` | 可视化 | 跨期箱线图（17特征分面） |
| `TASK5/outputs/eda_outlier_heatmap.html` | 可视化 | 极端值分位数热力图 |
| `TASK5/outputs/eda_correlation_heatmap.html` | 可视化 | Pearson + Spearman 双热力图 |
| `TASK5/outputs/eda_scatter_pairs.html` | 可视化 | 高相关对散点图矩阵 |
| `TASK5/outputs/eda_vif_barchart.html` | 可视化 | VIF 条形图（缩尾前后对比） |
| `TASK5/outputs/eda_feature_vs_target.html` | 可视化 | 分组 KDE 密度曲线 |
| `TASK5/outputs/eda_feature_target_corr.html` | 可视化 | 特征-目标相关系数排名 |

### 建模阶段（Phase 1~6）

| 文件 | 类型 | Phase |
|------|------|-------|
| `TASK5/outputs/data_split_report.txt` | 文本报告 | 1 |
| `TASK5/outputs/winsorize_comparison.csv` | 数据表 | 2 |
| `TASK5/outputs/skew_comparison.csv` | 数据表 | 2 |
| `TASK5/outputs/pca_report.json` | JSON 报告 | 2 |
| `TASK5/outputs/corr_after_winsorize.csv` | 相关性矩阵 | 2 |
| `TASK5/outputs/vif_after_winsorize.csv` | VIF 表 | 2 |
| `TASK5/outputs/lr_coefficients.csv` | LR 系数 | 3 |
| `TASK5/outputs/roc_curve.html` | 可视化 | 3 |
| `TASK5/outputs/model_comparison.csv` | 模型对比 | 4 |
| `TASK5/outputs/feature_importance.html` | 可视化 | 4 |
| `TASK5/outputs/test_evaluation.json` | JSON 报告 | 5 |
| `TASK5/outputs/test_predictions.csv` | 预测结果 | 5 |
| `TASK5/outputs/final_report.html` | 可视化总报告 | 5 |

---

> **下一步**：审阅 Phase 0 产出（`TASK5/outputs/` 目录下 10 个文件），确认 EDA 结论无误后进入 Phase 1（数据加载与划分）。
