# -*- coding: utf-8 -*-
"""
Gilman 滑坡 InSAR 时序提取与预处理
从 USGS shapefile 中定位核心蠕变区高相干活跃点，提取累计LOS位移时序，
转为 PINN 需要的 t, strain 输入。
"""
import shapefile, numpy as np, pandas as pd, os

# 原始数据根：用环境变量 GILMAN_BASE 指向含 unzipped/ 的 Gilman 原始数据目录
BASE = os.environ.get("GILMAN_BASE", os.path.dirname(os.path.abspath(__file__)))
shp = os.path.join(BASE, "unzipped", "timeseries_ERA5_demErr_Tintmasked.shp")
out_dir = os.path.join(BASE, "preprocessed_data")
os.makedirs(out_dir, exist_ok=True)

sf = shapefile.Reader(shp)
fields = [f[0] for f in sf.fields[1:]]
date_cols = [f for f in fields if f.startswith("D20")]
idx = {f: i for i, f in enumerate(fields)}
recs = sf.records()

# 核心蠕变区范围（最活跃点集中区）
LAT_LO, LAT_HI = 58.82, 58.85
LON_LO, LON_HI = -137.10, -137.06

core = []
for r in recs:
    lon = r[idx["XCOORD"]]; lat = r[idx["YCOORD"]]
    v = r[idx["VEL"]]; c = r[idx["COHERENCE"]]
    if None in (lon, lat, v, c):
        continue
    if LON_LO <= lon <= LON_HI and LAT_LO <= lat <= LAT_HI:
        core.append((r, lon, lat, v, c))

print(f"核心区点数: {len(core)}")
if not core:
    raise SystemExit("核心区无点")

# 按 负速率绝对值大 + 相干性高 排序（综合评分）
def score(x):
    return -x[3] * x[4]   # (-VEL)*COH, 越大越好
core.sort(key=score, reverse=True)

# 取综合最优的点
best = core[0]
r, lon, lat, v, c = best
print(f"选中点: lon={lon:.5f} lat={lat:.5f} VEL={v:.3f} mm/yr COH={c:.3f}")

# 提取该点 68 期位移（mm）
dvals = []
for dcol in date_cols:
    val = r[idx[dcol]]
    dvals.append(val if val is not None else np.nan)
dates = [d[1:] for d in date_cols]   # 去掉D前缀
dates_dt = pd.to_datetime(dates, format="%Y%m%d")

df = pd.DataFrame({"date": dates_dt, "los_mm": dvals})
df = df.dropna().sort_values("date").reset_index(drop=True)

# 相对时间 t（天），以首个观测为0
t0 = df["date"].iloc[0]
df["t_days"] = (df["date"] - t0).dt.days

# LOS -> 沿滑动方向位移（负值表示远离卫星/向下滑动，取正为滑动量级）
# 参考剖面长度 L（米），位移转应变
L_slide = 1200.0
disp_m = -df["los_mm"] / 1000.0   # 取正：滑动位移大小
df["strain"] = disp_m / L_slide

print(f"有效观测期数: {len(df)}  时间跨度: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
print(f"累计滑动位移: {df['los_mm'].iloc[0]:.2f} -> {df['los_mm'].iloc[-1]:.2f} mm")

# ---- 干净化：剔除早期稀疏噪声段，从稳定增长起点重新归零 ----
# 累计位移从首正单调增长段开始（滑动方向取正）
disp = df["strain"].values
t = df["t_days"].values
# 找第一个 strain 稳定为正且之后持续增长的点（至少后10点均值>0）
n = len(disp)
start = 0
for i in range(n - 10):
    if disp[i] > 0 and np.mean(disp[i:i+10]) > 0 and disp[i+9] >= disp[i]:
        start = i
        break
# 重新归零：以该点位移为基准
t_new = t[start:] - t[start]
strain_new = disp[start:] - disp[start]
df_clean = pd.DataFrame({"t": t_new, "strain": strain_new})
df_clean = df_clean[df_clean["t"] >= 0]

# 原始（未归零）完整序列
df_out = df[["t_days", "strain"]].rename(columns={"t_days": "t"})
csv_pinn = os.path.join(out_dir, "timeseries.csv")
df_out.to_csv(csv_pinn, index=False)

# 干净时序（PINN 用）
csv_clean = os.path.join(out_dir, "timeseries_clean.csv")
df_clean.to_csv(csv_clean, index=False)

meta = {
    "source": "USGS Gilman_timeseries_2015_2021",
    "point_lon": lon, "point_lat": lat, "VEL_mm_yr": v, "COHERENCE": c,
    "n_obs": len(df), "t_start": str(df['date'].iloc[0].date()),
    "t_end": str(df['date'].iloc[-1].date()),
    "L_slide_m": L_slide, "los_to_strain": "strain = -los_mm/1000 / L",
    "clean_start_idx": start, "clean_start_date": str(df['date'].iloc[start].date()),
}
pd.Series(meta).to_csv(os.path.join(out_dir, "meta.txt"), header=False)

print("\n已保存:")
print(" ", csv_pinn)
print(" ", csv_clean)
print(" ", os.path.join(out_dir, "meta.txt"))
print(f"\n干净时序起点: t={t_new[0]} (日期 {df['date'].iloc[start].date()}), 有效点数={len(df_clean)}")
print(f"最终累计应变: {df_clean['strain'].iloc[-1]:.6f}  (位移约 {df_clean['strain'].iloc[-1]*L_slide*1000:.1f} mm)")
print("\n干净 t, strain 前8行:")
print(df_clean.head(8).to_string(index=False))
print("...")
print("干净 t, strain 末8行:")
print(df_clean.tail(8).to_string(index=False))
