# fix_submission_ids.py
import os, shutil, pandas as pd, re

SUB_PATH = "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/feature_by_store/submission.csv"

# 1) 备份
bak = SUB_PATH + ".bak"
shutil.copy2(SUB_PATH, bak)
print("已备份到:", bak)

# 2) 读取
sub = pd.read_csv(SUB_PATH)
assert "id" in sub.columns, "CSV 里没有 id 列？"

n = len(sub)
print("行数 =", n)
# M5 正常应为 60980 行（30490 validation + 30490 evaluation）
half = n // 2

def force_suffix(series: pd.Series, suffix: str) -> pd.Series:
    s = series.astype(str).str.strip()
    # 去掉可能的奇怪空白，并强制把结尾后缀改成指定的
    s = s.str.replace(r"_(validation|evaluation)\s*$", f"_{suffix}", regex=True)
    # 若有极少数没有后缀的，直接补上
    needs_suffix = ~s.str.contains(r"_(validation|evaluation)\s*$", regex=True)
    s.loc[needs_suffix] = s.loc[needs_suffix] + f"_{suffix}"
    return s

# 3) 强制前半部分为 validation，后半部分为 evaluation
sub.loc[:half-1, "id"] = force_suffix(sub.loc[:half-1, "id"], "validation")
sub.loc[half:,   "id"] = force_suffix(sub.loc[half:,   "id"], "evaluation")

# 4) 自检
n_val = sub["id"].astype(str).str.endswith("_validation").sum()
n_eval = sub["id"].astype(str).str.endswith("_evaluation").sum()
print(f"validation 行: {n_val}, evaluation 行: {n_eval}")

# 检查是否有重复 id
dups = sub["id"].duplicated().sum()
if dups:
    # 打印前几条重复，方便排查
    print("⚠️ 发现重复 id 数量:", dups)
    print(sub[sub["id"].duplicated(keep=False)].sort_values("id").head(10))

# 5) 写回
sub.to_csv(SUB_PATH, index=False)
print("✅ 已修复并保存：", SUB_PATH)

# 6) 再次快速确认前后各3条
print("\n前3条：")
print(sub.head(3)[["id"]])
print("\n中间3条（分界附近）：")
print(sub.iloc[half-2:half+1][["id"]])
print("\n最后3条：")
print(sub.tail(3)[["id"]])
