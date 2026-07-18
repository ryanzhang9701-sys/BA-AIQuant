# 量化选股 ML 建模 Spec

## 1. 项目概述

### 1.1 核心架构：模型层 + 策略层分离

```
┌──────────────────────────────┐
│  模型层（固定，不依赖偏好）    │
│  输入 X → 输出 P(涨) ∈ [0,1] │
├──────────────────────────────┤
│  策略层（用户可配 6 个参数）   │
│  阈值 / 仓位 / 凯利系数 / …  │
├──────────────────────────────┤
│  交易执行（买入 / 卖出 / 观望）│
└──────────────────────────────┘
```

### 1.2 设计原则

- **模型只做一件事**：根据基本面特征预测下季度股价上涨的概率 P(涨)
- **策略由用户控制**：不同风险偏好的用户通过调整阈值和仓位参数，在同模型上实现个性化策略
- **滚动窗口回测**：Walk-Forward 模拟真实时序决策，评估策略表现

## 2. 数据描述

### 2.1 数据源

| 属性 | 值 |
|------|-----|
| 文件 | `model_data.csv` |
| 行数 | 39,617 |
| 股票数 | 4,282 只 A 股 |
| 时间范围 | 2020Q1 ~ 2022Q2（10 个季度） |
| 频率 | 季度 |
| 原始列数 | 22 列（含 Date, Code, Next_Ret） |

### 2.2 原始字段

| 序号 | 字段名 | 类别 | 方向 |
|------|--------|------|------|
| 1 | 企业倍数(EV/EBITDA) | 估值 | 越低越好 |
| 2 | 市净率 PB(MRQ) | 估值 | 越低越好 |
| 3 | 市现率 PCF(现金净流量 TTM) | 估值 | 越低越好 |
| 4 | 市现率 PCF(经营现金流 TTM) | 估值 | 越低越好 |
| 5 | 市盈率 PE(TTM) | 估值 | 越低越好 |
| 6 | 市盈率 PE(TTM, 扣除非经常性损益) | 估值 | 越低越好 |
| 7 | 市销率 PS(TTM) | 估值 | 越低越好 |
| 8 | 股息率(近 12 个月) | 估值 | 越高越好 |
| 9 | MV（市值，亿元） | 规模 | — |
| 10 | 净利润同比增长率 | 成长 | 越高越好 |
| 11 | 净资产同比增长率 | 成长 | 越高越好 |
| 12 | 利润总额同比增长率 | 成长 | 越高越好 |
| 13 | 基本每股收益同比增长率 | 成长 | 越高越好 |
| 14 | 总资产同比增长率 | 成长 | 越高越好 |
| 15 | 现金净流量同比增长率 | 成长 | 越高越好 |
| 16 | 经营活动产生的现金流量净额同比增长率 | 成长 | 越高越好 |
| 17 | 营业利润同比增长率 | 成长 | 越高越好 |
| 18 | 营业总收入同比增长率 | 成长 | 越高越好 |
| 19 | 营业收入同比增长率 | 成长 | 越高越好 |
| 20 | Next_Ret | 目标 | 未来一期收益率 |

## 3. 特征工程（自变量 X）

### 3.1 数据清洗

- 删除 Next_Ret 为空的样本（无标签不可训练）
- 缺失值处理：数值列填充截面中位数；分类场景下不适用则跳过
- 极端值处理：对原始因子做 1%/99% Winsorize 缩尾
- PE/PB/PS/PCF/EV 取倒数处理：将"越低越好"统一为"越高越好"方向

### 3.2 第一级：原始因子（19 个）

全部清洗后的原始字段（Next_Ret 除外），按经济学含义分组：

#### 估值因子（8 个）

| 变量名 | 来源 | 处理 |
|--------|------|------|
| `EV_EBITDA` | 企业倍数 | Winsorize |
| `PB` | 市净率 | Winsorize |
| `PCF_NetCash` | 市现率(现金净流量) | Winsorize |
| `PCF_Operating` | 市现率(经营现金流) | Winsorize |
| `PE_TTM` | 市盈率 | Winsorize |
| `PE_TTM_Deducted` | 市盈率(扣非) | Winsorize |
| `PS_TTM` | 市销率 | Winsorize |
| `Dividend_Yield` | 股息率 | Winsorize |

#### 成长因子（10 个）

| 变量名 | 来源 |
|--------|------|
| `Profit_Growth_YoY` | 净利润同比增长率 |
| `NetAsset_Growth_YoY` | 净资产同比增长率 |
| `TotalProfit_Growth_YoY` | 利润总额同比增长率 |
| `EPS_Growth_YoY` | 基本每股收益同比增长率 |
| `TotalAsset_Growth_YoY` | 总资产同比增长率 |
| `NetCash_Growth_YoY` | 现金净流量同比增长率 |
| `OperatingCF_Growth_YoY` | 经营活动现金流同比增长率 |
| `OperatingProfit_Growth_YoY` | 营业利润同比增长率 |
| `Revenue1_Growth_YoY` | 营业总收入同比增长率 |
| `Revenue2_Growth_YoY` | 营业收入同比增长率 |

#### 规模因子（1 个）

| 变量名 | 来源 | 处理 |
|--------|------|------|
| `MV_Log` | MV | `ln(MV)` 对数化降偏度 |

### 3.3 第二级：截面 Rank 标准化（8 个）

每个时间截面（同一 Date）内独立计算百分位排名，映射到 [0, 1]。

**计算规则**：训练/验证/测试各集合内部独立计算，禁止跨集合信息泄露。

| 变量名 | 来源 | Rank 方向 |
|--------|------|-----------|
| `R_PE` | PE_TTM | 降序（PE 越低排名越高） |
| `R_PB` | PB | 降序 |
| `R_PS` | PS_TTM | 降序 |
| `R_EV` | EV_EBITDA | 降序 |
| `R_Profit_Growth` | 净利润同比增长率 | 升序 |
| `R_Revenue_Growth` | 营业收入同比增长率 | 升序 |
| `R_Dividend` | 股息率 | 升序 |
| `R_MV` | MV | 升序 |

### 3.4 第三级：复合因子（4 个）

| 变量名 | 公式 | 逻辑 |
|--------|------|------|
| `Value_Composite` | `mean(R_PE, R_PB, R_PS, R_EV)` | 估值综合得分 |
| `Growth_Composite` | `mean(R_Profit_Growth, R_Revenue_Growth)` | 成长综合得分 |
| `GARP_Signal` | `(Value_Composite + Growth_Composite) / 2` | 价值成长均衡信号 |
| `Quality_Score` | `mean(R_Profit_Growth, R_Revenue_Growth, R_Dividend)` | 质量综合得分 |

### 3.5 特征汇总

| 级别 | 数量 | 内容 |
|------|------|------|
| 原始因子 | 19 | 估值 + 成长 + 规模 |
| 截面 Rank | 8 | 百分位标准化 |
| 复合因子 | 4 | 多因子合成 |
| **合计** | **31** | |

## 4. 因变量设计（Y）

### 4.1 唯一目标：上涨概率

```
Y = 1,  if Next_Ret > 0   （下季度取得正收益 = 获胜）
Y = 0,  if Next_Ret <= 0  （下季度零/负收益 = 失败）
```

| 属性 | 值 |
|------|-----|
| 任务类型 | 二分类 |
| 损失函数 | Binary CrossEntropy (LogLoss) |
| 模型输出 | 原始预测分数 → Platt Scaling → 校准后的 P(涨) ∈ [0, 1] |
| 校准评估 | Reliability Diagram + Expected Calibration Error (ECE) |

### 4.2 为什么是 P(涨) 而非分类标签

| 对比 | 三分类（P70/P30 切分） | P(涨) 概率 |
|------|------------------------|-----------|
| 阈值 | 硬编码，不可调 | **用户在策略层自由设定 T_buy / T_sell** |
| 风险偏好 | 模型替用户决定了 | 激进者 T=0.55，保守者 T=0.70 |
| 与凯利对接 | 需要额外模型输出 p | **p 直接是模型输出，天然对接** |
| 可解释性 | "这只股在 Top 30%" | **"这只股下季度涨的概率是 68%"** |

### 4.3 赔率 b 的估计

```
b = mean(Next_Ret | Next_Ret > 0, 训练集) / abs(mean(Next_Ret | Next_Ret <= 0, 训练集))
```

每个 Walk-Forward 窗口在训练集上独立估计 b，不随单笔交易变化。

## 5. 数据划分

### 5.1 Walk-Forward 滚动窗口（回测用）

| 窗口 | 训练集 | 验证集 | 测试/预测期 |
|------|--------|--------|------------|
| W1 | 2020Q1 ~ 2020Q4 | 2021Q1 | 2021Q2 |
| W2 | 2020Q1 ~ 2021Q1 | 2021Q2 | 2021Q3 |
| W3 | 2020Q1 ~ 2021Q2 | 2021Q3 | 2021Q4 |
| W4 | 2020Q1 ~ 2021Q3 | 2021Q4 | 2022Q1 |
| W5 | 2020Q1 ~ 2021Q4 | 2022Q1 | 2022Q2 |

### 5.2 固定切分（模型开发阶段，可选）

| 数据集 | 时间范围 | 用途 |
|--------|----------|------|
| 训练集 | 2020Q1 ~ 2021Q2 | 模型参数学习 |
| 验证集 | 2021Q3 ~ 2021Q4 | 超参调优 + 早停 + Platt 校准 + b 估计 |
| 测试集 | 2022Q1 ~ 2022Q2 | 最终评估 |

### 5.3 防前视偏差规则

- 截面 Rank 标准化：各集合内部独立计算
- 百分位阈值：各集合内部独立计算
- 缺失值填充（截面中位数）：各集合内部独立计算
- Winsorize 阈值：从训练集学习，应用到验证/测试集

## 6. 模型架构

### 6.1 模型池（4 个）

| # | 模型 | 定位 | 算法 | 概率支持 |
|---|------|------|------|:--:|
| 1 | **Logistic Regression** | 线性基准 | sklearn `LogisticRegression` (L2 正则) | `predict_proba()` |
| 2 | **Random Forest** | 非线性基准 | sklearn `RandomForestClassifier` | `predict_proba()` |
| 3 | **XGBoost** | 主力模型 | `XGBClassifier` (binary:logistic) | `predict_proba()` |
| 4 | **LightGBM** | 主力备选 | `LGBMClassifier` (binary) | `predict_proba()` |

**为什么这四个**：

- Logistic 是最简单的基线——如果它都跑不出来，说明因子本身没有线性信息量
- Random Forest 代表了 Bagging 集成——不依赖梯度，对超参不敏感
- XGBoost/LightGBM 是梯度提升树——当前表格数据建模的事实标准
- 四者从线性→Bagging→Boosting 覆盖了完整的模型复杂度光谱

### 6.2 各模型超参搜索

在每个 Walk-Forward 窗口的验证集上独立调参。

#### Logistic Regression

| 超参 | 搜索范围 |
|------|----------|
| C（正则强度倒数） | [0.01, 0.1, 1.0, 10.0] |
| penalty | ['l1', 'l2'] |
| solver | ['liblinear', 'saga'] |
| max_iter | 1000 |

#### Random Forest

| 超参 | 搜索范围 |
|------|----------|
| n_estimators | [100, 200, 300] |
| max_depth | [5, 10, 15, None] |
| min_samples_split | [2, 5, 10] |
| min_samples_leaf | [1, 2, 4] |
| max_features | ['sqrt', 'log2', None] |

#### XGBoost

| 超参 | 搜索范围 |
|------|----------|
| max_depth | [3, 5, 7] |
| learning_rate | [0.01, 0.05, 0.1] |
| n_estimators | [100, 200, 300] + early_stopping |
| subsample | [0.7, 0.8, 1.0] |
| colsample_bytree | [0.7, 0.8, 1.0] |
| reg_alpha | [0, 0.1, 1.0] |
| reg_lambda | [0.1, 1.0, 5.0] |

#### LightGBM

| 超参 | 搜索范围 |
|------|----------|
| num_leaves | [15, 31, 63] |
| learning_rate | [0.01, 0.05, 0.1] |
| n_estimators | [100, 200, 300] + early_stopping |
| subsample | [0.7, 0.8, 1.0] |
| colsample_bytree | [0.7, 0.8, 1.0] |
| reg_alpha | [0, 0.1, 1.0] |
| reg_lambda | [0.1, 1.0, 5.0] |

早停条件（XGB/LGB）：验证集 LogLoss 连续 20 轮不降。

### 6.3 概率校准

所有四个模型均做 Platt Scaling（在验证集上训练 Sigmoid 校准器）：

- 输入：各模型原始预测分数
- 输出：校准后的 P(涨) ∈ [0, 1]
- 评估校准质量：Reliability Diagram + Expected Calibration Error (ECE)
- **RF 和 XGB/LGB 的原生概率往往极端（趋近 0 或 1），校准尤为关键**

### 6.4 模型对比体系

#### 分类性能（每窗口 + 跨窗口汇总）

| 指标 | 说明 |
|------|------|
| AUC-ROC | 区分正负样本的整体能力 |
| Accuracy / Precision / Recall / F1 | 不同阈值下的分类表现 |
| LogLoss | 概率预测的准确性（越低越好） |
| Brier Score | 概率校准的综合指标 |

#### 概率校准质量

| 指标 | 说明 |
|------|------|
| ECE | Expected Calibration Error，概率预测的偏差 |
| Reliability Diagram | 预测概率 vs 实际频率的可视化对比 |

#### 回测表现（用同一组策略参数跑）

| 指标 | 说明 |
|------|------|
| 年化收益率 / Sharpe / MaxDD | 收益-风险三维 |
| 季度胜率 | 正收益窗口比例 |
| Rank IC 均值 / IC_IR | 预测与实际收益的排序相关性 |

#### 因子解释性

| 模型 | 输出 |
|------|------|
| Logistic | 系数 β 及其 p-value — 直接读取每个因子的方向和显著性 |
| RF | Mean Decrease Impurity / Permutation Importance |
| XGBoost | Gain / Cover / Frequency |
| LightGBM | Split / Gain |

### 6.5 模型参数保存

每个 Walk-Forward 窗口训练完成后，保存模型参数到 `TASK6/models/`：

```
TASK6/models/
  W1/
    logistic_params.pkl       # 模型参数 + 超参
    rf_params.pkl
    xgboost_params.pkl
    lgb_params.pkl
    calibrators.pkl           # 四个模型的 Platt 校准器
    metadata.json             # 超参、训练日期、验证集指标
  W2/ ...
  W3/ ...
  W4/ ...
  W5/ ...
  best_model_meta.json        # 全局最佳模型及参数摘要
```

**`best_model_meta.json` 示例**：

```json
{
  "best_model": "xgboost",
  "best_window": "W4",
  "avg_auc_cross_window": 0.582,
  "avg_sharpe_cross_window": 0.85,
  "model_rankings": {
    "xgboost": { "avg_auc": 0.582, "avg_sharpe": 0.85, "rank": 1 },
    "lightgbm": { "avg_auc": 0.579, "avg_sharpe": 0.81, "rank": 2 },
    "random_forest": { "avg_auc": 0.561, "avg_sharpe": 0.62, "rank": 3 },
    "logistic": { "avg_auc": 0.545, "avg_sharpe": 0.45, "rank": 4 }
  }
}
```

## 7. 回测设计

### 7.1 用户可配策略参数

参数存储在 `strategy_config.json`，用户可直接编辑，无需修改代码。

| 参数 | 含义 | 默认中性 | 激进 | 保守 |
|------|------|:--:|:--:|:--:|
| `T_buy` | 买入概率阈值：P(涨) > T_buy 才买入 | 0.60 | 0.55 | 0.65 |
| `T_sell` | 卖出概率阈值：P(涨) < T_sell 则卖出 | 0.50 | 0.45 | 0.55 |
| `α` | 凯利折减系数：`f_position = f_kelly × α` | 0.5 | 1.0 | 0.25 |
| `cap_stock` | 单只股票仓位上限（占总资金比例） | 5% | 8% | 3% |
| `N_max` | 最大持仓股票数 | 20 | 30 | 10 |
| `β` | 总仓位系数：全部持仓合计上限（占总资金比例） | 0.6 | 1.0 | 0.4 |

**选股逻辑**：先筛 P(涨) > T_buy，再按 P(涨) 降序取前 N_max。宁可少选，只挑最确定的。

### 7.2 每期决策流程

```
Step 1: 模型对所有股票预测 P(涨)，经 Platt 校准

Step 2: 筛选候选池
        candidates = { 股票 i | P_i(涨) > T_buy }
        按 P(涨) 降序，取前 N_max 只

Step 3: 计算单只原始仓位
        f_raw_i = max(0, (b × P_i − (1 − P_i)) / b)    // 凯利公式
        f_raw_i = f_raw_i × α                            // 凯利折减
        f_raw_i = min(f_raw_i, cap_stock)                // 单只上限截断

Step 4: 总仓位约束缩放
        F_raw = Σ f_raw_i
        if F_raw > β:
            scale = β / F_raw
            f_final_i = f_raw_i × scale
        else:
            f_final_i = f_raw_i    （信号不足时不强行加仓）

Step 5: 现金处理
        cash = 1 − Σ f_final_i     （零收益）
```

### 7.3 双阈值决策逻辑

```
P(涨) > T_buy            → 买入（进入候选池，按上述流程配仓）
T_sell <= P(涨) <= T_buy  → 观望/持有不动
P(涨) < T_sell           → 卖出/不参与
```

中间区域是"缓冲区"——模型不够确信时不做操作，避免在边缘概率上频繁换手。

### 7.4 调仓细则

| 规则 | 设定 |
|------|------|
| 调仓频率 | 每季度末（数据发布后） |
| 交易成本 | 单边 0.1%（含印花税 0.05% + 佣金 0.025% + 滑点 0.025%） |
| 换仓 | 已在持仓但本期不在候选池 → 全部卖出（触发卖出成本） |
| 再平衡 | 已在持仓且仍在候选池 → 按新 f_final 调整仓位（多退少补，触发部分成本） |

### 7.5 净值计算

```
期初净值 = 1.0

每期：
  调仓成本 = (新增买入额 + 卖出额) × 0.1%
  持仓收益 = Σ(股票 i 权重 × 股票 i 的 Next_Ret)
  期末净值 = 期初净值 × (1 + 持仓收益) − 调仓成本
```

### 7.6 评估指标

#### 收益类

| 指标 | 公式/说明 |
|------|-----------|
| 累计收益率 | `期末净值 − 1` |
| 年化收益率 | `期末净值 ^ (1 / 预测期年数) − 1` |
| 相对基准超额 | 策略年化 − 基准年化 |

#### 风险类

| 指标 | 说明 |
|------|------|
| 年化波动率 | 季度收益率标准差 × 2（√4） |
| 最大回撤 (MDD) | `max((Peak_t − Valley_t) / Peak_t)` |
| 最大回撤持续期 | Peak → Recovery 的季度数 |

#### 性价比类

| 指标 | 公式 |
|------|------|
| Sharpe Ratio | `(年化收益 − 无风险利率) / 年化波动` |
| Calmar Ratio | `年化收益 / 最大回撤` |
| 季度胜率 | 正收益窗口数 / 总窗口数 |

#### 信号质量类

| 指标 | 说明 |
|------|------|
| Rank IC | 每期 p_win 与实际 Next_Ret 的 Spearman 相关系数 |
| IC_IR | `mean(IC) / std(IC)`，> 0.3 为良好 |
| 平均换手率 | 每期调仓金额 / 总资产 |

#### 基准

| 基准 | 数据源 |
|------|--------|
| 沪深 300 总回报 | 沪深 300 指数季度收益率 |
| 全 A 等权组合 | 当期全部股票 Next_Ret 的等权均值 |

## 8. 输出物

### 8.1 数据文件

| 文件 | 内容 |
|------|------|
| `features_YYYYMMDD.csv` | 特征工程后的完整数据集（含 X 和 Y） |
| `backtest_results.csv` | 每期调仓明细（模型、股票、权重、收益） |
| `model_comparison.csv` | 四模型分类指标 + 回测指标对比表 |

### 8.2 模型参数文件

| 路径 | 内容 |
|------|------|
| `models/W{n}/*_params.pkl` | 各窗口各模型的训练参数 |
| `models/W{n}/calibrators.pkl` | Platt 校准器 |
| `models/W{n}/metadata.json` | 超参、验证集指标 |
| `models/best_model_meta.json` | 全局最佳模型摘要 |

### 8.3 图表

| 图表 | 说明 |
|------|------|
| 净值曲线 | 四模型策略 vs 沪深 300 vs 全 A 等权，多线对比 |
| 回撤曲线 | 各模型净值回撤序列 |
| 特征重要性 | 四模型 Top 15 特征横排对比 |
| IC 序列 | 每期各模型 Rank IC 柱状图 |
| 校准曲线 | 四模型 Reliability Diagram 对比 |
| 模型对比雷达图 | AUC / Sharpe / MaxDD / 胜率 / IC_IR 五维雷达 |

### 8.3 综合评估表

一张汇总表包含上述全部指标，格式如：

```
═══════════════════════════════════
          策略回测评估报告
═══════════════════════════════════
累计收益率:      X%
年化收益率:      Y%
年化波动率:      Z%
最大回撤:        D%  (持续 N 季)
Sharpe Ratio:    S
Calmar Ratio:    C
季度胜率:        W% (N/M)
Rank IC 均值:    R
IC_IR:           I
平均换手率:      T%
═══════════════════════════════════
```

## 9. 实现步骤

| 步骤 | 内容 | 输出 |
|------|------|------|
| Step 1 | 数据清洗 + 特征工程（31 个 X） | `features.csv` |
| Step 2 | 构造 Y 标签（Binary: Next_Ret > 0） | 更新 `features.csv` |
| Step 3 | Walk-Forward 循环（5 个窗口）：| |
| | 3a. 切分训练/验证/测试 | |
| | 3b. 截面 Rank 标准化（各集合独立） | |
| | 3c. 训练 4 个模型（Logistic / RF / XGB / LGB） | |
| | 3d. 网格搜索 + 早停，记录最佳超参 | |
| | 3e. Platt Scaling 概率校准（4 个校准器） | |
| | 3f. 估计 b（赔率），从训练集 | |
| | 3g. 在测试集预测 → 四个模型分别按策略参数生成持仓 | |
| | 3h. 计算各模型本期收益 | |
| | 3i. 保存模型参数到 `models/W{n}/` | |
| Step 4 | 汇总回测结果 + 计算评估指标 | `backtest_results.csv` |
| Step 5 | 模型对比：分类指标 + 回测指标 + 因子解释性 | `model_comparison.csv` |
| Step 6 | 生成可视化报告（净值曲线 + 特征重要性 + IC + 校准曲线 + 模型对比） | HTML Dashboard |

## 10. 风险与注意事项

- **前视偏差**：截面操作必须在各集合内独立进行，严禁跨集合使用统计量
- **生存偏差**：当前数据假设不存在退市股票的数据缺失问题；如有，需标记
- **过拟合**：XGBoost 在少量特征+短时间序列上容易过拟合，需配合较强的正则化
- **市场结构变化**：2020~2022 涵盖疫情冲击+复苏，模型学到的模式可能在未来失效
- **概率校准**：Platt Scaling 假设对数几率与预测分数线性相关，若验证集数据少则校准不稳定
