# -*- coding: utf-8 -*-
"""
Achoma landslide (Peru) displacement time-series extraction from GeoTIFF stack.
================================================================================
Reads all NS/EW offset-tracking GeoTIFFs in 原始\, extracts a small window
around the max-displacement core point, and builds:
  - timeseries_raw.csv   : date, NS_m, EW_m, mag_m
  - timeseries_clean.csv : t (days since 2017-11-27), strain (= mag_m / L)
  - displacement_time.png / velocity_time.png
"""
import os, re, glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 原始/处理输出目录：用环境变量 ACHOMA_RAW / ACHOMA_PROC 覆盖
RAW  = os.environ.get("ACHOMA_RAW", os.path.dirname(os.path.abspath(__file__)))
PROC = os.environ.get("ACHOMA_PROC", os.path.join(RAW, "processed"))

# core point: row=456, col=481 on the 2020-06-17 max-displacement pixel
# (UTM x=212068.5, y=826677.5, EPSG:32719)
ROW, COL = 456, 481
HALF = 3   # 7x7 window = 21 m x 21 m

# reference length L (see meta.txt for justification)
L_m = 100.0   # m

# ---- enumerate epochs from NS file names ----
ns_files = sorted(glob.glob(os.path.join(RAW, "NS*.tif")))
epochs = []
for fp in ns_files:
    m = re.search(r"NS(\d{8})\.tif$", os.path.basename(fp))
    if m:
        epochs.append(m.group(1))
epochs.sort()
print(f"Found {len(epochs)} epochs: {epochs[0]} .. {epochs[-1]}", flush=True)

rows = []
for dt in epochs:
    ns_fp = os.path.join(RAW, f"NS{dt}.tif")
    ew_fp = os.path.join(RAW, f"EW{dt}.tif")
    if not (os.path.exists(ns_fp) and os.path.exists(ew_fp)):
        print(f"  SKIP {dt}: missing NS or EW", flush=True)
        continue
    with rasterio.open(ns_fp) as dns, rasterio.open(ew_fp) as dew:
        ns = dns.read(1)
        ew = dew.read(1)
    # 7x7 window mean
    r0, r1 = ROW-HALF, ROW+HALF+1
    c0, c1 = COL-HALF, COL+HALF+1
    ns_m = float(ns[r0:r1, c0:c1].mean())
    ew_m = float(ew[r0:r1, c0:c1].mean())
    mag = float(np.sqrt(ns_m**2 + ew_m**2))
    rows.append(dict(date=pd.Timestamp(dt), NS_m=ns_m, EW_m=ew_m, mag_m=mag))

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
print(f"Extracted {len(df)} epochs", flush=True)
print(df.head(), flush=True)
print(df.tail(), flush=True)

# relative time t (days since first epoch = reference, all-zero)
t0 = df["date"].iloc[0]
df["t"] = (df["date"] - t0).dt.days.astype(float)
df["strain"] = df["mag_m"] / L_m

# velocity from cumulative displacement (m/day)
df["v_m_d"] = np.gradient(df["mag_m"].values, df["t"].values)

# ---- write raw and clean CSV ----
raw_csv = os.path.join(PROC, "timeseries_raw.csv")
df[["date","NS_m","EW_m","mag_m"]].to_csv(raw_csv, index=False, encoding="utf-8", float_format="%.6f")

clean = df[["t","strain"]].copy()
clean_csv = os.path.join(PROC, "timeseries_clean.csv")
clean.to_csv(clean_csv, index=False, encoding="utf-8", float_format="%.6f")
print(f"Wrote {raw_csv}")
print(f"Wrote {clean_csv}")

# ---- summary ----
print(f"\nt range: {df['t'].iloc[0]:.0f} .. {df['t'].iloc[-1]:.0f} days "
      f"({(df['t'].iloc[-1]-df['t'].iloc[0])/30.44:.1f} months)")
print(f"mag range: {df['mag_m'].iloc[0]:.3f} .. {df['mag_m'].iloc[-1]:.3f} m")
print(f"strain range: {df['strain'].iloc[0]:.5f} .. {df['strain'].iloc[-1]:.5f}")

# ---- Plot 1: displacement-time ----
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(df["date"], df["mag_m"], "ko-", ms=4, lw=1.2)
ax.set_xlabel("Date"); ax.set_ylabel("Cumulative horizontal displacement  |u|  [m]")
ax.set_title("Achoma landslide (Peru) — core-point displacement from PlanetScope offset-tracking")
ax.grid(alpha=0.3); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig.autofmt_xdate()
fig.tight_layout()
p1 = os.path.join(PROC, "displacement_time.png")
fig.savefig(p1, dpi=130); plt.close(fig)
print(f"Wrote {p1}")

# ---- Plot 2: velocity-time ----
fig, ax = plt.subplots(figsize=(11, 5.5))
# raw gradient in m/day -> m/month
ax.plot(df["date"], df["v_m_d"]*30.44, "o-", ms=4, color="#c0392b", lw=1.2,
        label="velocity [m/month]")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Date"); ax.set_ylabel("Displacement velocity  [m/month]")
ax.set_title("Achoma landslide — velocity evolution (check monotonic acceleration before failure)")
ax.grid(alpha=0.3); ax.legend(); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
fig.autofmt_xdate()
fig.tight_layout()
p2 = os.path.join(PROC, "velocity_time.png")
fig.savefig(p2, dpi=130); plt.close(fig)
print(f"Wrote {p2}")

# ---- Print late-epoch detail for acceleration check ----
print("\n--- Last 25 epochs (acceleration window) ---")
late = df.tail(25)[["date","t","mag_m","v_m_d"]].copy()
late["v_m_month"] = late["v_m_d"]*30.44
print(late.to_string(index=False))
