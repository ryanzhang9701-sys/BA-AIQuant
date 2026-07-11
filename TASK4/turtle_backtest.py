"""
海龟策略回测引擎
Spec v1.0.0 — 独立模式 + 组合模式
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # BA工作坊/
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 策略参数（来自 spec 第二部分）
# ============================================================
PARAMS = {
    # 系统1
    "s1_entry": 20, "s1_exit": 10,
    # 系统2
    "s2_entry": 55, "s2_exit": 20,
    # ATR
    "atr_period": 20,
    # 仓位
    "risk_pct": 0.01, "lot_size": 100, "account_init": 1_000_000,
    # 加仓
    "max_units": 4, "add_interval_n": 0.5,
    # 止损
    "stop_n": 2.0,
    # 滑点
    "slippage_pct": 0.001,
    # 风控
    "single_max": 4, "group_max": 6, "direction_max": 12,
}

STOCKS = [
    # A 股（tushare，前复权价格）
    {"ts_code": "688981.SH", "name": "中芯国际(A)", "market": "A",
     "price_file": "data/688981.SH_中芯国际/daily_adjusted.csv",
     "group": "semiconductor",
     "price_cols": {"date": "trade_date", "open": "open_qfq", "high": "high_qfq",
                     "low": "low_qfq", "close": "close_qfq", "volume": "vol", "amount": "amount"},
     "lot_size": 100},
    {"ts_code": "002594.SZ", "name": "比亚迪(A)", "market": "A",
     "price_file": "data/002594.SZ_比亚迪/daily_adjusted.csv",
     "group": "nev",
     "price_cols": {"date": "trade_date", "open": "open_qfq", "high": "high_qfq",
                     "low": "low_qfq", "close": "close_qfq", "volume": "vol", "amount": "amount"},
     "lot_size": 100},
    {"ts_code": "603986.SH", "name": "兆易创新", "market": "A",
     "price_file": "data/603986.SH_兆易创新/daily_adjusted.csv",
     "group": "semiconductor",
     "price_cols": {"date": "trade_date", "open": "open_qfq", "high": "high_qfq",
                     "low": "low_qfq", "close": "close_qfq", "volume": "vol", "amount": "amount"},
     "lot_size": 100},
    # H 股（akshare，原始价格，无需复权）
    {"ts_code": "00981.HK", "name": "中芯国际(H)", "market": "HK",
     "price_file": "data/688981.SH_中芯国际/daily_hk.csv",
     "group": "semiconductor",
     "price_cols": {"date": "trade_date", "open": "open", "high": "high",
                     "low": "low", "close": "close", "volume": "vol", "amount": "amount"},
     "lot_size": 500, "currency": "HKD"},
    {"ts_code": "01211.HK", "name": "比亚迪(H)", "market": "HK",
     "price_file": "data/002594.SZ_比亚迪/daily_hk.csv",
     "group": "nev",
     "price_cols": {"date": "trade_date", "open": "open", "high": "high",
                     "low": "low", "close": "close", "volume": "vol", "amount": "amount"},
     "lot_size": 500, "currency": "HKD"},
]

PRICE_COLS_DEFAULT = {"date": "trade_date", "open": "open_qfq", "high": "high_qfq",
              "low": "low_qfq", "close": "close_qfq", "volume": "vol", "amount": "amount"}


# ============================================================
# 数据加载 & 指标计算
# ============================================================

def load_stock(stock):
    """加载单只股票数据，计算 TR、ATR、唐奇安通道"""
    path = BASE_DIR / stock["price_file"]
    pcols = stock.get("price_cols", PRICE_COLS_DEFAULT)
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 只取所需价格列，避免与未复权列冲突
    cols_in = [pcols[k] for k in ["date","open","high","low","close","volume","amount"]]
    df = df[cols_in].copy()
    df.columns = ["date", "open", "high", "low", "close", "vol", "amount"]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    df["ts_code"] = stock["ts_code"]
    df["name"] = stock["name"]

    # TR — pure numpy
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(h)
    tr = np.empty(n)
    tr[0] = float(h[0] - l[0])
    tr[1:] = np.maximum(h[1:] - l[1:],
               np.maximum(np.abs(h[1:] - c[:-1]),
                          np.abs(l[1:] - c[:-1])))
    df["tr"] = tr
    df.loc[0, "tr"] = df.loc[0, "high"] - df.loc[0, "low"]

    # ATR (Wilder exponential smoothing)
    p = PARAMS["atr_period"]
    atr = np.full(len(df), np.nan)
    atr[p - 1] = df["tr"].iloc[:p].mean()
    for i in range(p, len(df)):
        atr[i] = (atr[i - 1] * (p - 1) + df["tr"].iloc[i]) / p
    df["n"] = atr

    # Donchian Channels — System 1
    df["s1_upper"] = df["high"].shift(1).rolling(PARAMS["s1_entry"]).max()
    df["s1_lower"] = df["low"].shift(1).rolling(PARAMS["s1_exit"]).min()

    # Donchian Channels — System 2
    df["s2_upper"] = df["high"].shift(1).rolling(PARAMS["s2_entry"]).max()
    df["s2_lower"] = df["low"].shift(1).rolling(PARAMS["s2_exit"]).min()

    return df


# ============================================================
# 独立模式回测
# ============================================================

class TurtleBacktest:
    """单股票独立海龟策略回测"""

    def __init__(self, df, name, lot_size=100):
        self.df = df.reset_index(drop=True)
        self.name = name
        self.lot_size = lot_size
        self.signals = []
        self.positions = []
        self.equity_curve = []
        self.cash = PARAMS["account_init"]
        self.holdings = []  # list of dicts: {system, entry_price, entry_n, units, entry_idx}

        self._last_s1_pnl = 0  # for reentry filter

    def unit_shares(self, n_val):
        """计算 1 Unit 的股数"""
        risk_amount = self.cash * PARAMS["risk_pct"]
        raw = int(risk_amount / (n_val * self.lot_size))
        return max(raw * self.lot_size, self.lot_size)  # 至少 1 手

    def run(self):
        df = self.df
        n_rows = len(df)
        # 等 ATR 和通道有值后再开始（至少 max(55, 20)+1 行）
        start_idx = PARAMS["s2_entry"] + 1

        for i in range(start_idx, n_rows):
            # 当日价格
            today = df.iloc[i]
            n_val = today["n"]
            if pd.isna(n_val) or n_val <= 0:
                self._record_equity(i)
                continue

            # 先检查止损（所有持仓）
            self._check_stops(i)

            # 系统2 出场
            if today["close"] < today["s2_lower"] and not pd.isna(today["s2_lower"]):
                self._exit_system("S2", i, "TRAILING_EXIT_S2")

            # 系统1 出场
            if today["close"] < today["s1_lower"] and not pd.isna(today["s1_lower"]):
                self._exit_system("S1", i, "TRAILING_EXIT_S1")

            # 系统2 入场
            if today["close"] > today["s2_upper"] and not pd.isna(today["s2_upper"]):
                if not self._has_system("S2"):
                    self._enter("S2", i, n_val)

            # 系统1 入场 (with reentry filter)
            if today["close"] > today["s1_upper"] and not pd.isna(today["s1_upper"]):
                if not self._has_system("S1") and self._last_s1_pnl >= 0:
                    self._enter("S1", i, n_val)

            # 加仓检查
            self._check_pyramid(i, n_val)

            # 记录日终净值
            self._record_equity(i)

        # 最后一天强制平仓
        self._force_exit_all(n_rows - 1)
        return self._results()

    def _enter(self, system, idx, n_val):
        """入场：以次日开盘价成交"""
        if idx + 1 >= len(self.df):
            return
        exec_price = self.df.iloc[idx + 1]["open"] * (1 + PARAMS["slippage_pct"])
        us = self.unit_shares(n_val)
        if us <= 0:
            return
        cost = exec_price * us
        if cost > self.cash:
            return
        self.cash -= cost
        pos = {
            "system": system, "entry_price": exec_price,
            "entry_n": n_val, "units": 1, "entry_idx": idx + 1,
            "shares": us, "add_prices": [exec_price],
            "stop_price": exec_price - PARAMS["stop_n"] * n_val,
        }
        self.holdings.append(pos)
        self.signals.append({
            "date": self.df.iloc[idx]["date"], "system": system,
            "type": "BUY", "price": exec_price, "units": 1,
            "n": n_val, "equity": self.cash,
        })

    def _exit_system(self, system, idx, reason):
        to_remove = [h for h in self.holdings if h["system"] == system]
        for h in to_remove:
            exec_price = self.df.iloc[idx]["open"] * (1 - PARAMS["slippage_pct"])
            self.cash += exec_price * h["shares"]
            self.positions.append(self._make_position(h, idx, exec_price, reason))
            self.signals.append({
                "date": self.df.iloc[idx]["date"], "system": system,
                "type": "SELL", "price": exec_price, "units": h["units"],
                "n": h["entry_n"], "equity": self.cash,
            })
            if system == "S1":
                self._last_s1_pnl = (exec_price - h["entry_price"]) * h["shares"]
            self.holdings.remove(h)

    def _check_stops(self, idx):
        today = self.df.iloc[idx]
        to_remove = []
        for h in self.holdings:
            if today["low"] < h["stop_price"]:
                exec_price = h["stop_price"]
                self.cash += exec_price * h["shares"]
                self.positions.append(self._make_position(h, idx, exec_price, "STOP_LOSS"))
                self.signals.append({
                    "date": today["date"], "system": h["system"],
                    "type": "STOP_OUT", "price": exec_price, "units": h["units"],
                    "n": h["entry_n"], "equity": self.cash,
                })
                if h["system"] == "S1":
                    self._last_s1_pnl = (exec_price - h["entry_price"]) * h["shares"]
                to_remove.append(h)
        for h in to_remove:
            self.holdings.remove(h)

    def _check_pyramid(self, idx, n_val):
        if idx + 1 >= len(self.df):
            return
        for h in self.holdings:
            if h["units"] >= PARAMS["max_units"]:
                continue
            needed = h["add_prices"][-1] + PARAMS["add_interval_n"] * h["entry_n"]
            if self.df.iloc[idx]["close"] >= needed:
                exec_price = self.df.iloc[idx + 1]["open"] * (1 + PARAMS["slippage_pct"])
                us = self.unit_shares(n_val)
                if us <= 0 or exec_price * us > self.cash:
                    continue
                self.cash -= exec_price * us
                h["units"] += 1
                h["shares"] += us
                h["add_prices"].append(exec_price)
                # 更新止损到加仓价的 -2N
                new_stop = exec_price - PARAMS["stop_n"] * n_val
                h["stop_price"] = max(h["stop_price"], new_stop)
                self.signals.append({
                    "date": self.df.iloc[idx]["date"], "system": h["system"],
                    "type": "ADD", "price": exec_price, "units": h["units"],
                    "n": n_val, "equity": self.cash,
                })

    def _has_system(self, system):
        return any(h["system"] == system for h in self.holdings)

    def _record_equity(self, idx):
        today = self.df.iloc[idx]
        mkt_val = sum(
            h["shares"] * today["close"] for h in self.holdings
        )
        total = self.cash + mkt_val
        self.equity_curve.append({
            "date": today["date"], "equity": total, "cash": self.cash,
            "mkt_val": mkt_val, "units": sum(h["units"] for h in self.holdings),
        })

    def _force_exit_all(self, idx):
        for h in list(self.holdings):
            exec_price = self.df.iloc[idx]["close"]
            self.cash += exec_price * h["shares"]
            self.positions.append(self._make_position(h, idx, exec_price, "END_OF_DATA"))
            self.holdings.remove(h)
        self._record_equity(idx)

    def _make_position(self, h, exit_idx, exit_price, reason):
        pnl = (exit_price - h["entry_price"]) * h["shares"]
        n_at_entry = h["entry_n"]
        return {
            "name": self.name,
            "system": h["system"],
            "entry_date": self.df.iloc[h["entry_idx"]]["date"],
            "entry_price": h["entry_price"],
            "n_at_entry": n_at_entry,
            "exit_date": self.df.iloc[exit_idx]["date"] if exit_idx < len(self.df) else None,
            "exit_price": exit_price,
            "exit_reason": reason,
            "units": h["units"],
            "shares": h["shares"],
            "pnl": pnl,
            "pnl_pct": pnl / (h["entry_price"] * h["shares"]),
            "pnl_r": pnl / (n_at_entry * h["shares"]) if n_at_entry * h["shares"] > 0 else 0,
        }

    def _results(self):
        eq_df = pd.DataFrame(self.equity_curve)
        pos_df = pd.DataFrame(self.positions) if self.positions else pd.DataFrame()
        sig_df = pd.DataFrame(self.signals) if self.signals else pd.DataFrame()

        perf = self._calc_performance(eq_df, pos_df)
        return {"name": self.name, "equity": eq_df, "positions": pos_df,
                "signals": sig_df, "performance": perf}

    def _calc_performance(self, eq_df, pos_df):
        if eq_df.empty:
            return {}
        eq_df["ret"] = eq_df["equity"].pct_change()
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["drawdown"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]

        n_days = len(eq_df)
        total_ret = eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0
        max_dd = eq_df["drawdown"].min()
        daily_rf = 0.02 / 252
        excess = eq_df["ret"].dropna() - daily_rf
        sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

        if pos_df.empty:
            return {"total_return": total_ret, "annual_return": ann_ret,
                    "max_drawdown": max_dd, "sharpe": sharpe,
                    "win_rate": 0, "profit_factor": 0, "avg_r": 0, "total_trades": 0}

        winners = pos_df[pos_df["pnl"] > 0]
        losers = pos_df[pos_df["pnl"] <= 0]
        total_win = winners["pnl"].sum()
        total_loss = abs(losers["pnl"].sum())

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "win_rate": len(winners) / len(pos_df) if len(pos_df) > 0 else 0,
            "profit_factor": total_win / total_loss if total_loss > 0 else float("inf"),
            "avg_win": winners["pnl"].mean() if len(winners) > 0 else 0,
            "avg_loss": losers["pnl"].mean() if len(losers) > 0 else 0,
            "avg_r": pos_df["pnl_r"].mean() if len(pos_df) > 0 else 0,
            "total_trades": len(pos_df),
            "avg_hold_days": (pos_df["exit_date"] - pos_df["entry_date"]).dt.days.mean() if len(pos_df) > 0 else 0,
        }


# ============================================================
# 主流程：所有股票独立运行 + 生成 dashboard 数据
# ============================================================

def run_all():
    print("=" * 60)
    print("海龟策略回测 — 独立模式")
    print("=" * 60)

    all_results = []
    for stock in STOCKS:
        print(f"\n[{stock['name']} ({stock['ts_code']})]")
        df = load_stock(stock)
        print(f"  数据: {len(df)} 行, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        bt = TurtleBacktest(df, stock["name"], lot_size=stock.get("lot_size", 100))
        res = bt.run()
        perf = res["performance"]
        print(f"  年化收益: {perf.get('annual_return', 0):.2%}")
        print(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")
        print(f"  夏普比率: {perf.get('sharpe', 0):.2f}")
        print(f"  胜率: {perf.get('win_rate', 0):.1%}")
        print(f"  盈亏因子: {perf.get('profit_factor', 0):.2f}")
        print(f"  交易笔数: {perf.get('total_trades', 0)}")
        all_results.append({**res, "ts_code": stock["ts_code"], "group": stock["group"]})

    # 保存汇总
    summary = []
    for r in all_results:
        summary.append({"name": r["name"], "ts_code": r["ts_code"],
                        "group": r["group"], **r["performance"]})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "turtle_performance.csv", index=False, encoding="utf-8-sig")

    # 保存详细结果
    all_positions = pd.concat([r["positions"].assign(stock=r["name"]) for r in all_results], ignore_index=True)
    all_positions.to_csv(OUTPUT_DIR / "turtle_positions.csv", index=False, encoding="utf-8-sig")

    all_signals = pd.concat([r["signals"].assign(stock=r["name"]) for r in all_results], ignore_index=True)
    all_signals.to_csv(OUTPUT_DIR / "turtle_signals.csv", index=False, encoding="utf-8-sig")

    # 合并净值曲线
    eq_dfs = []
    for r in all_results:
        eq = r["equity"].copy()
        eq["stock"] = r["name"]
        eq_dfs.append(eq)
    eq_all = pd.concat(eq_dfs, ignore_index=True)
    eq_all.to_csv(OUTPUT_DIR / "turtle_equity.csv", index=False, encoding="utf-8-sig")

    print(f"\n输出文件已保存至 {OUTPUT_DIR}/")
    print(f"  turtle_performance.csv  — 绩效汇总")
    print(f"  turtle_positions.csv    — 持仓账本")
    print(f"  turtle_signals.csv      — 信号日志")
    print(f"  turtle_equity.csv       — 净值曲线")

    return all_results


# ============================================================
# Dashboard 数据生成
# ============================================================

def make_dashboard_data(all_results):
    """生成 HTML 看板所需的 JSON 数据"""

    # 绩效汇总
    perf_list = []
    for r in all_results:
        p = r["performance"]
        perf_list.append({
            "name": r["name"], "ts_code": r["ts_code"], "group": r["group"],
            "annual_return": round(p.get("annual_return", 0), 4),
            "max_drawdown": round(p.get("max_drawdown", 0), 4),
            "sharpe": round(p.get("sharpe", 0), 2),
            "win_rate": round(p.get("win_rate", 0), 3),
            "profit_factor": round(min(p.get("profit_factor", 0), 10), 2),
            "avg_hold_days": round(p.get("avg_hold_days", 0), 1),
            "total_trades": p.get("total_trades", 0),
        })

    # 净值曲线合并（按日期对齐）
    eq_list = []
    for r in all_results:
        eq = r["equity"]
        eq_norm = []
        init_eq = eq["equity"].iloc[0] if len(eq) > 0 else 1
        for _, row in eq.iterrows():
            eq_norm.append({
                "date": str(row["date"].date()),
                "equity": round(row["equity"] / init_eq * 100, 2),
            })
        eq_list.append({"name": r["name"], "data": eq_norm})

    # 月度热力图
    heatmap = []
    for r in all_results:
        eq = r["equity"].copy()
        eq["month"] = pd.to_datetime(eq["date"]).dt.to_period("M")
        monthly = eq.groupby("month").apply(
            lambda g: (g["equity"].iloc[-1] / g["equity"].iloc[0] - 1) if len(g) > 1 else 0,
            include_groups=False
        )
        for m, ret in monthly.items():
            heatmap.append({
                "stock": r["name"],
                "year": m.year,
                "month": m.month,
                "return": round(ret, 4),
            })

    # 特征对比表
    features = []
    for r in all_results:
        df = r["equity"]
        init = float(df["equity"].iloc[0])
        features.append({
            "name": r["name"],
            "total_return": round(float(r["performance"].get("total_return", 0)), 4),
            "max_drawdown": round(float(r["performance"].get("max_drawdown", 0)), 4),
            "group": r["group"],
            "avg_hold_days": round(float(r["performance"].get("avg_hold_days", 0)), 1),
            "total_trades": int(r["performance"].get("total_trades", 0)),
            "annual_return": round(float(r["performance"].get("annual_return", 0)), 4),
            "sharpe": round(float(r["performance"].get("sharpe", 0)), 2),
        })

    data = {
        "performance": perf_list,
        "equity_curves": eq_list,
        "monthly_heatmap": heatmap,
        "features": features,
    }
    return data


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    results = run_all()
    dash_data = make_dashboard_data(results)

    with open(OUTPUT_DIR / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(dash_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"Dashboard 数据已保存至 {OUTPUT_DIR / 'dashboard_data.json'}")
