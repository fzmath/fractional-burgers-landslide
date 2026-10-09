# -*- coding: utf-8 -*-
"""
实验C 公共模块：分数阶 Burgers M3 反演、多起点 least_squares、AIC/BIC、
剖面似然、相关矩阵、边界命中检测。

模型（恒应力 sigma0=1）：
  M3: eps = s/E_H + s/E_1*(1-exp(-E_1*t/eta_1)) + s*t^alpha/(eta_a*Gamma(alpha+1))
      参数顺序 p = [E_H, E_1, eta_1, eta_a, alpha]  （5 参数）
噪声标定：noise = std(diff(eps))/sqrt(2)
似然：高斯、固定 noise；ll = -0.5*n*(log(2*pi*noise^2)+1) - rss/(2*noise^2)
      aic = 2k - 2ll ; bic = ln(n)*k - 2ll
剖面似然：固定 alpha=a，重优化其余 4 参数得 rss(a)；
      2*Delta_ll = (rss(a)-rss_min)/noise^2 <= chi2_{1,0.95}=3.84 为 95%CI。

本模块不修改任何原始数据/已有脚本，只读取 CSV。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as Gamma_fn

SIGMA0 = 1.0
PNAMES = ["E_H", "E_1", "eta_1", "eta_a", "alpha"]
CHI2_1_95 = 3.841458820694124   # chi2_{1 dof, 0.95}

# 数据根目录：默认 <仓库>/data；可用环境变量 CREEP_DATA_ROOT 覆盖
_DATA_ROOT = os.environ.get(
    "CREEP_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)

# 全部案例；可用 --skip <Case> 排除缺数据的案例
ALL_CASES = ["Gilman", "Achoma", "Gradenbach", "Baijiabao"]

def select_cases(argv=None):
    """返回要处理的案例列表。
    例: python step0_reproduce_bench.py --skip Baijiabao
    """
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--skip", nargs="*", default=[])
    args, _ = p.parse_known_args(argv)
    return [c for c in ALL_CASES if c not in args.skip]

# 各案例配置：CSV 路径、参数上下界、alpha 多起点网格
# 说明：
#  - Gilman 原论文 alpha 上界=0.999（近线性被压到界）；实验C为检验 alpha 是否显著区别于 1，
#    将上界放宽到 2.0（覆盖 1.0 及略超 1），E/eta 界沿用 Gilman 原脚本。
#  - Achoma/Gradenbach/Baijiabao 的 E/eta 界沿用各自原脚本（E [1e-3~1e-2,1e8], eta [1,1e13]）。
#  - Gradenbach 论文主解 alpha=3.125 用 alpha 上界=5.0（见对比报告 4.3）。
CASES = {
    "Gilman": dict(
        csv=os.path.join(_DATA_ROOT, "Gilman", "timeseries_clean.csv"),
        lo=[1e-2, 1e-2, 1e2, 1e2, 0.02],
        hi=[1e7, 1e7, 1e9, 1e9, 2.0],
        alpha_inits=[0.3, 0.6, 0.9, 1.2, 1.6],
        init_pack=[1e4, 1e4, 1e5, 1e5],
    ),
    "Achoma": dict(
        csv=os.path.join(_DATA_ROOT, "Achoma", "timeseries_clean.csv"),
        lo=[1e-3, 1e-3, 1e0, 1e0, 0.02],
        hi=[1e8, 1e8, 1e13, 1e13, 5.0],
        alpha_inits=[0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        init_pack=[1e3, 1e3, 1e4, 1e4],
    ),
    "Gradenbach": dict(
        csv=os.path.join(_DATA_ROOT, "Gradenbach", "timeseries_clean.csv"),
        lo=[1e-2, 1e-2, 1e0, 1e0, 0.02],
        hi=[1e8, 1e8, 1e13, 1e13, 5.0],
        alpha_inits=[0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        init_pack=[1e4, 1e4, 1e5, 1e5],
        # 论文报告全局解（对比报告 4.3）：E_H=280.99,E_1=10.262,eta_1=9.92e4,eta_a=1e13,alpha=3.125
        anchor=[281.0, 10.26, 9.92e4, 1.0e13, 3.125],
    ),
    "Baijiabao": dict(
        csv=os.path.join(_DATA_ROOT, "Baijiabao", "timeseries_clean.csv"),
        lo=[1e-2, 1e-2, 1e0, 1e0, 0.02],
        hi=[1e8, 1e8, 1e13, 1e13, 5.0],
        alpha_inits=[0.3, 0.7, 1.0, 1.5, 2.2, 2.8, 3.5, 4.2],
        init_pack=[1e4, 1e4, 1e5, 1e5],
    ),
}


def load_case(name):
    """返回 dict(t, eps, n, noise, lo, hi, ...)"""
    cfg = CASES[name]
    df = pd.read_csv(cfg["csv"])
    t = df["t"].values.astype(float)
    eps = df["strain"].values.astype(float)
    n = len(t)
    noise = max(float(np.std(np.diff(eps)) / np.sqrt(2.0)), 1e-12)
    d = dict(name=name, t=t, eps=eps, n=n, noise=noise,
             lo=np.array(cfg["lo"], float), hi=np.array(cfg["hi"], float),
             alpha_inits=list(cfg["alpha_inits"]),
             init_pack=list(cfg["init_pack"]))
    if "anchor" in cfg:
        d["anchor"] = list(cfg["anchor"])
    return d


def m3(p, t):
    """分数阶 Burgers 应变。p=[EH,E1,eta1,eta_a,alpha]。"""
    EH, E1, eta1, etaa, alpha = p
    return (SIGMA0 / EH
            + SIGMA0 / E1 * (1.0 - np.exp(-E1 * t / eta1))
            + SIGMA0 * t ** alpha / (etaa * Gamma_fn(alpha + 1.0)))


def aic_bic(n, k, rss, noise):
    ll = -0.5 * n * (np.log(2 * np.pi * noise ** 2) + 1.0) - rss / (2.0 * noise ** 2)
    return 2 * k - 2 * ll, np.log(n) * k - 2 * ll


def fit_m3(c, alpha_hi=None, alpha_inits=None, max_nfev=5000,
           xtol=1e-8, ftol=1e-8, verbose=False):
    """多起点 least_squares 反演 M3。返回 best sol（含 .x, .cost, .jac）。
    为跨越 (alpha, eta_a) trade-off 脊的两个局部盆，每个 alpha 初值搭配
    两种 eta_a 尺度（常规 + 大），共 len(a_inits)*2 个起点。"""
    t, eps, noise = c["t"], c["eps"], c["noise"]
    lo = c["lo"].copy()
    hi = c["hi"].copy()
    if alpha_hi is not None:
        hi[4] = float(alpha_hi)
    a_inits = c["alpha_inits"] if alpha_inits is None else list(alpha_inits)
    pack = c["init_pack"]
    # 两种 eta_a 尺度：常规 pack 与放大到接近上界的尺度（用于找到 alpha 大、eta_a 顶界的盆）
    packs = [pack, [pack[0], pack[1], pack[2], hi[3] * 0.1]]
    anchor = c.get("anchor")

    def resid(p):
        return (m3(p, t) - eps) / noise

    best = None
    for pk in packs:
        for a0 in a_inits:
            init = [pk[0], pk[1], pk[2], pk[3], a0]
            # 裁剪到界内（恢复模拟的 alpha 起点网格可能超出某案例上界）
            init = [float(np.clip(init[i], lo[i], hi[i])) for i in range(5)]
            sol = least_squares(resid, init, bounds=(lo, hi),
                                max_nfev=max_nfev, xtol=xtol, ftol=ftol)
            if verbose:
                print(f"    init a={a0:.2f} eta_a0={pk[3]:.1e}: "
                      f"cost={sol.cost:.4e} a*={sol.x[4]:.4f}", flush=True)
            if best is None or sol.cost < best.cost:
                best = sol
    if anchor is not None:
        anc = [float(np.clip(anchor[i], lo[i], hi[i])) for i in range(5)]
        sol = least_squares(resid, anc, bounds=(lo, hi),
                            max_nfev=max_nfev, xtol=xtol, ftol=ftol)
        if verbose:
            print(f"    anchor: cost={sol.cost:.4e} a*={sol.x[4]:.4f}", flush=True)
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def rss_from_sol(sol, noise):
    return 2.0 * sol.cost * noise ** 2


def r2_from_rss(rss, eps):
    return 1.0 - rss / np.sum((eps - eps.mean()) ** 2)


def profile_alpha(c, alpha_grid, alpha_hi=None, max_nfev=5000,
                 xtol=1e-8, ftol=1e-8):
    """固定 alpha=a，重优化其余 4 参数。返回 (grid, rss_grid, best_other_grid)。
    对每个 a 用多起点（从全局最优附近 + 均匀扫描）。"""
    t, eps, noise = c["t"], c["eps"], c["noise"]
    lo4 = c["lo"][:4].copy()
    hi4 = c["hi"][:4].copy()
    if alpha_hi is not None:
        pass  # 其余 4 参数界不变

    # 先跑一次全局 M3 拿“其它参数”的参考点
    full = fit_m3(c, alpha_hi=alpha_hi, max_nfev=max_nfev,
                  xtol=xtol, ftol=ftol)
    ref4 = full.x[:4].copy()
    ref_cost = full.cost
    ref_rss = 2.0 * ref_cost * noise ** 2

    out_rss = []
    out_other = []
    for a in alpha_grid:
        def resid4(p4):
            p = np.array([p4[0], p4[1], p4[2], p4[3], a])
            return (m3(p, t) - eps) / noise
        best = None
        # 起点：参考点 + 若干尺度扰动
        starts = [ref4.copy()]
        for sc in [0.1, 10.0]:
            s = ref4.copy() * sc
            s = np.clip(s, lo4, hi4)
            starts.append(s)
        # 均匀扫描起点
        pack = c["init_pack"]
        starts.append(np.array(pack, float))
        for st in starts:
            sol = least_squares(resid4, st, bounds=(lo4, hi4),
                                max_nfev=max_nfev, xtol=xtol, ftol=ftol)
            if best is None or sol.cost < best.cost:
                best = sol
        out_rss.append(2.0 * best.cost * noise ** 2)
        out_other.append(best.x.copy())
    return dict(grid=np.array(alpha_grid, float),
                rss=np.array(out_rss, float),
                other=np.array(out_other, float),
                ref_x=full.x.copy(), ref_rss=ref_rss, ref_cost=ref_cost)


def corr_matrix(sol):
    """由最小二乘雅可比 J 估计参数相关矩阵。corr = D^-1/2 (J'J)^-1 D^-1/2。"""
    J = np.asarray(sol.jac)          # (n,5)
    JtJ = J.T @ J
    try:
        cov = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JtJ)
    d = np.sqrt(np.diag(cov))
    dd = np.outer(d, d)
    corr = cov / dd
    return corr


def boundary_hits(p, lo, hi, log_tol=0.005, lin_tol=1e-3):
    """返回每个参数是否贴界。
    对跨度>2个数量级的正参数用对数尺度容差 log_tol（log10 差）；
    对近线性参数（alpha）用线性容差 lin_tol（相对界宽）。
    log_tol=0.005 即 |log10(p)-log10(bound)|<=0.005 ≈ 乘法 1.0116 倍，
    等价于最小二乘把参数压到界上的数值行为。"""
    hits = []
    for i in range(len(p)):
        wide = (hi[i] / max(lo[i], 1e-300)) > 100.0
        if wide:
            hit_lo = np.log10(p[i]) - np.log10(lo[i]) <= log_tol
            hit_hi = np.log10(hi[i]) - np.log10(p[i]) <= log_tol
        else:
            span = hi[i] - lo[i]
            hit_lo = (p[i] - lo[i]) <= lin_tol * span
            hit_hi = (hi[i] - p[i]) <= lin_tol * span
        hits.append(dict(param=PNAMES[i], value=float(p[i]),
                         lo=float(lo[i]), hi=float(hi[i]),
                         hit_lo=bool(hit_lo), hit_hi=bool(hit_hi)))
    return hits
