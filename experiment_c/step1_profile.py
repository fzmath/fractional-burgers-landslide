# -*- coding: utf-8 -*-
"""
实验① 剖面似然：对四案例固定 alpha 网格、重优化其余 4 参数，
绘制 2*Delta(ll) 剖面曲线，用 chi2_{1,0.95}=3.84 给 alpha 95%CI，
并检验 alpha=1.0 是否落在 CI 内（是否显著区别于 1）。
同时输出：最优解参数相关矩阵、贴界参数。
输出目录: 脚本所在目录
"""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import common as C

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# 各案例 alpha 剖面网格（覆盖最优解与 1.0，密度在最优附近加密）
GRIDS = {
    "Gilman":     np.round(np.linspace(0.4, 1.9, 31), 3),
    "Achoma":     np.round(np.linspace(1.0, 5.5, 46), 3),
    "Gradenbach": np.round(np.linspace(0.8, 4.6, 39), 3),
    "Baijiabao":  np.round(np.linspace(0.5, 3.0, 31), 3),
}

ci_rows = []
prof_rows = []
corr_rows = []
wall_rows = []

for name in C.select_cases():
    c = C.load_case(name)
    print(f"\n===== {name} (n={c['n']}, noise={c['noise']:.3e}) =====", flush=True)
    t0 = time.time()
    # 全局 M3
    sol = C.fit_m3(c, max_nfev=5000, xtol=1e-8, ftol=1e-8)
    p = sol.x
    rss_min = C.rss_from_sol(sol, c["noise"])
    r2 = C.r2_from_rss(rss_min, c["eps"])
    aic, bic = C.aic_bic(c["n"], 5, rss_min, c["noise"])
    print(f"  全局最优: alpha={p[4]:.4f} R2={r2:.6f} AIC={aic:.1f} "
          f"E_H={p[0]:.3g} E_1={p[1]:.3g} eta_1={p[2]:.3g} eta_a={p[3]:.3g}", flush=True)

    # 剖面似然
    grid = GRIDS[name]
    pr = C.profile_alpha(c, grid, max_nfev=5000, xtol=1e-8, ftol=1e-8)
    rss_min2 = pr["ref_rss"]
    two_dll = (pr["rss"] - rss_min2) / c["noise"] ** 2
    in_ci = two_dll <= C.CHI2_1_95

    # CI：单调区间（取连续满足 in_ci 的段）
    ci_lo = np.nan; ci_hi = np.nan
    if in_ci.any():
        ci_lo = pr["grid"][in_ci].min()
        ci_hi = pr["grid"][in_ci].max()
    a_at_1 = None
    # alpha=1 处的 2dll（插值或最近网格）
    idx1 = np.argmin(np.abs(pr["grid"] - 1.0))
    dll_at_1 = two_dll[idx1]
    a_hat = p[4]
    sig_diff_1 = dll_at_1 > C.CHI2_1_95   # alpha=1 在 CI 外 => 显著区别于 1

    print(f"  alpha 95%CI=[{ci_lo},{ci_hi}], alpha_hat={a_hat:.3f}, "
          f"2dll(a=1)={dll_at_1:.2f} (阈值3.84), 显著区别于1={sig_diff_1}", flush=True)
    print(f"  剖面耗时={time.time()-t0:.0f}s", flush=True)

    # 存剖面数据
    for g, r, d, ic in zip(pr["grid"], pr["rss"], two_dll, in_ci):
        prof_rows.append(dict(case=name, alpha=g, rss=r, two_delta_ll=d,
                              in_95ci=bool(ic)))
    ci_rows.append(dict(case=name, n=c["n"], noise=c["noise"], alpha_hat=a_hat,
                       r2=r2, aic=aic, bic=bic,
                       ci95_lo=ci_lo, ci95_hi=ci_hi,
                       dll_at_alpha1=dll_at_1,
                       sig_diff_from_1=bool(sig_diff_1),
                       E_H=p[0], E_1=p[1], eta_1=p[2], eta_a=p[3]))

    # 相关矩阵
    corr = C.corr_matrix(sol)
    for i in range(5):
        for j in range(5):
            corr_rows.append(dict(case=name, row=C.PNAMES[i], col=C.PNAMES[j],
                                  corr=corr[i, j]))
    # 相关脊：|offdiag| 最大
    off = []
    for i in range(5):
        for j in range(i+1, 5):
            off.append((C.PNAMES[i], C.PNAMES[j], corr[i, j]))
    off.sort(key=lambda z: -abs(z[2]))
    print(f"  相关脊 top3: " +
          ", ".join([f"{a}-{b}={v:+.3f}" for a, b, v in off[:3]]), flush=True)

    # 贴界
    hits = C.boundary_hits(p, c["lo"], c["hi"])
    for h in hits:
        wall_rows.append(dict(case=name, param=h["param"], value=h["value"],
                             lo=h["lo"], hi=h["hi"],
                             hit_lo=h["hit_lo"], hit_hi=h["hit_hi"]))
        if h["hit_lo"] or h["hit_hi"]:
            print(f"  贴界: {h['param']}={h['value']:.4g} "
                  f"[lo={h['lo']:.3g},hi={h['hi']:.3g}] "
                  f"{'LO' if h['hit_lo'] else ''}{'HI' if h['hit_hi'] else ''}", flush=True)

    # 剖面图画
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr["grid"], two_dll, "o-", color="tab:blue", ms=4, lw=1.5, label="剖面 2Δll")
    ax.axhline(C.CHI2_1_95, color="r", ls="--", lw=1.2, label=f"95% 阈值 χ²={C.CHI2_1_95:.2f}")
    ax.axvline(a_hat, color="g", ls=":", lw=1.5, label=f"alpha*={a_hat:.3f}")
    ax.axvline(1.0, color="orange", ls="-.", lw=1.3, label="α=1 (整数阶)")
    if in_ci.any():
        ax.axvspan(ci_lo, ci_hi, color="green", alpha=0.12, label="95% CI")
    ax.set_xlabel("α"); ax.set_ylabel("2Δll (相对最小)")
    ax.set_title(f"{name}: α 剖面似然 (n={c['n']})")
    ax.set_ylim(-0.5, min(max(two_dll.max()*1.1, 10), 200))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"profile_alpha_{name}.png"), dpi=140)
    plt.close(fig)

# 相关矩阵热图（4 子图）
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, name in zip(axes.ravel(), ["Gilman", "Achoma", "Gradenbach", "Baijiabao"]):
    sub = [r for r in corr_rows if r["case"] == name]
    M = np.zeros((5, 5))
    for r in sub:
        M[C.PNAMES.index(r["row"]), C.PNAMES.index(r["col"])] = r["corr"]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(C.PNAMES, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(C.PNAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_title(f"{name} 参数相关矩阵")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
fig.savefig(os.path.join(OUT, "corr_matrix_all.png"), dpi=140)
plt.close(fig)

# 存表
pd.DataFrame(prof_rows).to_csv(os.path.join(OUT, "profile_alpha_data.csv"),
                               index=False, encoding="utf-8-sig")
pd.DataFrame(ci_rows).to_csv(os.path.join(OUT, "profile_ci_summary.csv"),
                             index=False, encoding="utf-8-sig")
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT, "corr_matrix.csv"),
                               index=False, encoding="utf-8-sig")
pd.DataFrame(wall_rows).to_csv(os.path.join(OUT, "boundary_hits.csv"),
                               index=False, encoding="utf-8-sig")
print("\n全部完成，已存 profile/corr/boundary 表与图。", flush=True)
