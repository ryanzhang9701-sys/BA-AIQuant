# -*- coding: utf-8 -*-
"""
Phase 1 — 数据加载与划分
构建 t→t+1 特征-标签对，按时间顺序划分 Train/Val/Test
"""
import pandas as pd
import numpy as np
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_FILE = "model_data_stock.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("Phase 1 — 数据加载与划分")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print(f"\n[1] 原始数据: {df.shape[0]:,} 行 × {df.shape[1]} 列")
print(f"    股票数: {df['Code'].nunique():,}")
print(f"    日期范围: {df['Date'].min()} ~ {df['Date'].max()}")
print(f"    缺失值: {df.isnull().sum().sum()} (期望 0)")
print(f"    重复行: {df.duplicated().sum()} (期望 0)")

# ============================================================
# 2. 构建 t → t+1 特征-标签对
# ============================================================
print(f"\n[2] 构建 t → t+1 特征-标签对...")

dates = sorted(df['Date'].unique())
print(f"    全部日期: {dates}")

# 对每个 (date_t, date_t1) 配对，将 date_t 的特征与 date_t1 的 Y 匹配
pairs = []
for i in range(len(dates) - 1):
    date_t = dates[i]
    date_t1 = dates[i + 1]

    # 期 t 的特征 (所有行)
    df_t = df[df['Date'] == date_t].copy()

    # 期 t+1 的 Code → Y 映射
    y_next = df[df['Date'] == date_t1][['Code', 'Y']].copy()
    y_next.rename(columns={'Y': 'Y_next'}, inplace=True)

    # 合并: 每只股票的 t 期特征 + t+1 期标签
    merged = df_t.merge(y_next, on='Code', how='inner')
    merged['feature_date'] = date_t
    merged['label_date'] = date_t1
    pairs.append(merged)

    n_before = len(df_t)
    n_after = len(merged)
    print(f"    {date_t} → {date_t1}: {n_before:,} 只股票 → {n_after:,} 匹配 (丢失 {n_before - n_after})")

df_model = pd.concat(pairs, ignore_index=True)
print(f"\n    有效样本总数: {len(df_model):,} (丢失 {len(df) - len(df_model):,} = 第5期无t+1标签)")

# 目标变量
y_col = 'Y_next'

# ============================================================
# 3. 按时间顺序划分 Train / Val / Test
# ============================================================
print(f"\n[3] 划分 Train / Val / Test...")

# Spec §5.1:
#   Train: feature_date ∈ {2021-06-30, 2021-09-30}
#   Val:   feature_date ∈ {2021-12-31}
#   Test:  feature_date ∈ {2022-03-31}

train_dates = ['2021-06-30', '2021-09-30']
val_dates   = ['2021-12-31']
test_dates  = ['2022-03-31']

train_mask = df_model['feature_date'].isin(train_dates)
val_mask   = df_model['feature_date'].isin(val_dates)
test_mask  = df_model['feature_date'].isin(test_dates)

df_train = df_model[train_mask].copy()
df_val   = df_model[val_mask].copy()
df_test  = df_model[test_mask].copy()

# 验证无泄漏: 检查 Code 跨集重叠
train_codes = set(df_train['Code'])
val_codes   = set(df_val['Code'])
test_codes  = set(df_test['Code'])

# 注意：同一 Code 自然会出现在不同时期，这是面板数据的特性
# 关键检查：同一 Code 在同一 feature_date 不应出现在两个集合中
overlap_train_val = train_codes & val_codes
overlap_train_test = train_codes & test_codes
overlap_val_test = val_codes & test_codes

print(f"    Code 跨集重叠 (面板数据正常现象):")
print(f"      Train ∩ Val:  {len(overlap_train_val):,} 只")
print(f"      Train ∩ Test: {len(overlap_train_test):,} 只")
print(f"      Val ∩ Test:   {len(overlap_val_test):,} 只")
print(f"    注意：Code 跨集重叠是面板数据的正常特征——同一只股票在不同时期出现。")
print(f"    关键：时间维度无重叠，Train 最新 feature_date={df_train['feature_date'].max()} < Val={df_val['feature_date'].min()} < Test={df_test['feature_date'].min()}")

# ============================================================
# 4. 输出划分统计报告
# ============================================================
print(f"\n[4] 划分统计报告")

def report(name, df_sub):
    n = len(df_sub)
    n_pos = df_sub[y_col].sum()
    n_stocks = df_sub['Code'].nunique()
    dates_in = sorted(df_sub['feature_date'].unique())
    return {
        '集合': name,
        '样本数': n,
        '正例数': int(n_pos),
        '正例占比': f"{n_pos/n*100:.2f}%",
        '股票数': n_stocks,
        '特征日期': ', '.join(str(d) for d in dates_in),
        '标签日期': ', '.join(str(d) for d in sorted(df_sub['label_date'].unique()))
    }

results = [
    report('Train', df_train),
    report('Val',   df_val),
    report('Test',  df_test),
    report('总计',  df_model),
]

# 打印
print()
header = f"{'集合':<10} {'样本数':>8} {'正例占比':>10} {'股票数':>8} {'特征日期':<30} {'标签日期':<30}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['集合']:<10} {r['样本数']:>8,} {r['正例占比']:>10} {r['股票数']:>8,} "
          f"{r['特征日期']:<30} {r['标签日期']:<30}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR, 'data_split_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Phase 1 — 数据加载与划分报告\n")
    f.write(f"生成时间: {pd.Timestamp.now()}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"原始数据: {len(df):,} 行, {df['Code'].nunique():,} 只股票\n")
    f.write(f"有效样本 (t→t+1): {len(df_model):,} 行\n\n")

    f.write("划分方案: 时间顺序 (Spec §5.1)\n")
    f.write("-" * 40 + "\n")
    for r in results:
        f.write(f"\n{r['集合']}:\n")
        f.write(f"  样本数:    {r['样本数']:,}\n")
        f.write(f"  正例数:    {r['正例数']:,}\n")
        f.write(f"  正例占比:  {r['正例占比']}\n")
        f.write(f"  股票数:    {r['股票数']:,}\n")
        f.write(f"  特征日期:  {r['特征日期']}\n")
        f.write(f"  标签日期:  {r['标签日期']}\n")

    f.write(f"\nCode 跨集情况 (面板数据正常):\n")
    f.write(f"  Train ∩ Val:  {len(overlap_train_val):,}\n")
    f.write(f"  Train ∩ Test: {len(overlap_train_test):,}\n")
    f.write(f"  Val ∩ Test:   {len(overlap_val_test):,}\n")

print(f"\n报告已保存: {report_path}")

# ============================================================
# 5. 导出处理后的 CSV（供 Phase 2 使用）
# ============================================================
# 仅保留建模需要的列: 剔除 Date (用 feature_date 替代), 保留 Code, Y_next, 所有特征
feat_cols = [c for c in df.columns if c not in ['Date', 'Y'] and c in df_model.columns]
export_cols = ['feature_date', 'label_date', 'Code', 'Y_next'] + [c for c in feat_cols if c != 'Y']

df_train[export_cols].to_csv(os.path.join(OUTPUT_DIR, 'train_raw.csv'), index=False, encoding='utf-8-sig')
df_val[export_cols].to_csv(os.path.join(OUTPUT_DIR, 'val_raw.csv'), index=False, encoding='utf-8-sig')
df_test[export_cols].to_csv(os.path.join(OUTPUT_DIR, 'test_raw.csv'), index=False, encoding='utf-8-sig')

print(f"\n数据导出:")
print(f"  训练集: {os.path.join(OUTPUT_DIR, 'train_raw.csv')} ({len(df_train):,} 行)")
print(f"  验证集: {os.path.join(OUTPUT_DIR, 'val_raw.csv')} ({len(df_val):,} 行)")
print(f"  测试集: {os.path.join(OUTPUT_DIR, 'test_raw.csv')} ({len(df_test):,} 行)")

print(f"\n{'='*60}")
print("Phase 1 完成")
print(f"{'='*60}")
