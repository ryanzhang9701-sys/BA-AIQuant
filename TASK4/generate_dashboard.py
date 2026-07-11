"""Generate self-contained Turtle Strategy dashboard HTML with embedded price data"""
import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "turtle_comparison.html"

STOCKS = [
    {"code": "688981.SH", "name": "中芯国际(A)", "file": "data/688981.SH_中芯国际/daily_adjusted.csv",
     "cols": {"date":"trade_date","open":"open_qfq","high":"high_qfq","low":"low_qfq","close":"close_qfq"}, "lot": 100},
    {"code": "00981.HK",  "name": "中芯国际(H)", "file": "data/688981.SH_中芯国际/daily_hk.csv",
     "cols": {"date":"trade_date","open":"open","high":"high","low":"low","close":"close"}, "lot": 500},
    {"code": "002594.SZ", "name": "比亚迪(A)",  "file": "data/002594.SZ_比亚迪/daily_adjusted.csv",
     "cols": {"date":"trade_date","open":"open_qfq","high":"high_qfq","low":"low_qfq","close":"close_qfq"}, "lot": 100},
    {"code": "01211.HK",  "name": "比亚迪(H)",  "file": "data/002594.SZ_比亚迪/daily_hk.csv",
     "cols": {"date":"trade_date","open":"open","high":"high","low":"low","close":"close"}, "lot": 500},
    {"code": "603986.SH", "name": "兆易创新",   "file": "data/603986.SH_兆易创新/daily_adjusted.csv",
     "cols": {"date":"trade_date","open":"open_qfq","high":"high_qfq","low":"low_qfq","close":"close_qfq"}, "lot": 100},
]

raw_data = {}
for s in STOCKS:
    df = pd.read_csv(BASE / s["file"], encoding="utf-8-sig")
    cc = s["cols"]
    df = df[[cc[k] for k in ["date","open","high","low","close"]]].copy()
    df.columns = ["date","open","high","low","close"]
    df["date"] = df["date"].astype(str)
    records = []
    for _, row in df.iterrows():
        records.append({
            "d": row["date"],
            "o": round(float(row["open"]), 2),
            "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2),
            "c": round(float(row["close"]), 2),
        })
    group = "semiconductor" if s["code"] in ("688981.SH","00981.HK","603986.SH") else "nev"
    raw_data[s["code"]] = {
        "name": s["name"], "group": group, "lot": s["lot"], "data": records
    }

data_json = json.dumps(raw_data, ensure_ascii=False)

# Read the template HTML
html = Path(__file__).parent / "turtle_comparison.html"
content = html.read_text(encoding="utf-8")

# Replace the RAW_DATA placeholder with actual data
# Find: "const RAW_DATA = {"
# Replace up to the closing "};" of the object
import re

# Simpler approach: find and replace
old_start = 'const RAW_DATA = {'
new_start = f'const RAW_DATA = {data_json};'
idx = content.find(old_start)
if idx >= 0:
    # Find the matching }; for this const
    depth = 0
    end_idx = idx
    for i in range(idx, len(content)):
        if content[i] == '{': depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 2  # include };
                break
    content = content[:idx] + new_start + content[end_idx:]

html.write_text(content, encoding="utf-8")
print(f"Dashboard generated: {OUT}")
print(f"Data entries: {sum(len(v['data']) for v in raw_data.values())}")
