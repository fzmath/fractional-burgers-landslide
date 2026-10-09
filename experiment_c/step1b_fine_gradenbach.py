# -*- coding: utf-8 -*-
"""
实验①补充：Gradenbach alpha 加密剖面（2.90~3.30, 步长0.02），
精确刻画全局盆附近的 95%CI 低/上界（上界受 eta_a=1e13 壁截断）。
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import common as C

OUT = os.path.dirname(os.path.abspath(__file__))
c = C.load_case("Gradenbach")
grid = np.round(np.arange(2.90, 3.31, 0.02), 3)
t0 = time.time()
pr = C.profile_alpha(c, grid, max_nfev=5000, xtol=1e-8, ftol=1e-8)
rss_min = pr["ref_rss"]
two_dll = (pr["rss"] - rss_min) / c["noise"] ** 2
in_ci = two_dll <= C.CHI2_1_95
print("alpha, rss, 2dll, in_ci")
for g, r, d, ic in zip(pr["grid"], pr["rss"], two_dll, in_ci):
    print(f"  {g:.2f}  {r:.6e}  {d:.2f}  {ic}", flush=True)
ci = pr["grid"][in_ci]
if len(ci):
    print(f"\n95%CI (fine) = [{ci.min():.3f}, {ci.max():.3f}]")
else:
    print("\n95%CI: 无网格点落入（壁截断，上侧开/不可达）")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(pr["grid"], two_dll, "o-", color="tab:blue", ms=5, lw=1.5)
ax.axhline(C.CHI2_1_95, color="r", ls="--", label="95% 阈值 3.84")
ax.set_xlabel("alpha"); ax.set_ylabel("2Delta ll")
ax.set_title("Gradenbach: alpha 加密剖面（全局盆附近）")
ax.set_ylim(-1, 200); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "profile_alpha_Gradenbach_fine.png"), dpi=140)
print(f"耗时 {time.time()-t0:.0f}s")
