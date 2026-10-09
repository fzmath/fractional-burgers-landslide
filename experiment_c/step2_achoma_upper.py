# -*- coding: utf-8 -*-
"""
实验② Achoma alpha 上界放宽：5.0 -> 8.0 -> 10.0，重跑 M3 多起点反演，
检查 alpha 估计是否稳定（是否贴新上界）、AIC/BIC 模型选择（M1/M2/M3）是否改变。
输出: achoma_upper_relax.csv
"""
import os, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as Gamma_fn
import common as C

OUT = os.path.dirname(os.path.abspath(__file__))
c = C.load_case("Achoma")
t, eps, noise, n = c["t"], c["eps"], c["noise"], c["n"]

def m2(p):
    EH, E1, eta1, etaa = p
    return C.SIGMA0/EH + C.SIGMA0/E1*(1-np.exp(-E1*t/eta1)) + C.SIGMA0*t/etaa

# M1 线性（与上界无关）
A = np.vstack([t, np.ones_like(t)]).T
coef, *_ = np.linalg.lstsq(A, eps, rcond=None)
pred1 = A@coef; rss1 = np.sum((eps-pred1)**2)
aic1, bic1 = C.aic_bic(n, 2, rss1, noise)

# M2 整数阶（alpha=1，与上界无关）
def resid2(p): return (m2(p)-eps)/noise
b2 = (c["lo"][:4], c["hi"][:4])
best2 = None
for init in [(1e3,1e3,1e4,1e4),(1e2,1e2,1e3,1e3),(1e4,1e4,1e5,1e5)]:
    s = least_squares(resid2, init, bounds=b2, max_nfev=5000, xtol=1e-8, ftol=1e-8)
    if best2 is None or s.cost < best2.cost: best2 = s
rss2 = C.rss_from_sol(best2, noise)
aic2, bic2 = C.aic_bic(n, 4, rss2, noise)

rows = [dict(alpha_hi=np.nan, model="M1 linear", alpha=np.nan,
             r2=C.r2_from_rss(rss1,eps), aic=aic1, bic=bic1, wall_hit=""),
        dict(alpha_hi=np.nan, model="M2 integer Burgers", alpha=1.0,
             r2=C.r2_from_rss(rss2,eps), aic=aic2, bic=bic2, wall_hit="")]

for hi in [5.0, 8.0, 10.0]:
    t0 = time.time()
    cc = dict(c)
    # 临时把 alpha 上界改为 hi
    lo = c["lo"].copy(); hi_arr = c["hi"].copy(); hi_arr[4] = hi
    # 构造一个临时 case dict 传给 fit_m3
    tmp = dict(c)
    tmp["lo"] = lo; tmp["hi"] = hi_arr
    sol = C.fit_m3(tmp, alpha_hi=hi, max_nfev=5000, xtol=1e-8, ftol=1e-8)
    p = sol.x
    rss = C.rss_from_sol(sol, noise)
    r2 = C.r2_from_rss(rss, eps)
    aic, bic = C.aic_bic(n, 5, rss, noise)
    wall = "YES_alpha_at_hi" if (hi - p[4]) <= 0.005*hi else ""
    print(f"alpha_hi={hi}: alpha*={p[4]:.4f} eta_a={p[3]:.3e} "
          f"R2={r2:.6f} AIC={aic:.2f} BIC={bic:.2f} wall={wall} "
          f"({time.time()-t0:.0f}s)", flush=True)
    rows.append(dict(alpha_hi=hi, model="M3 fractional Burgers", alpha=p[4],
                     r2=r2, aic=aic, bic=bic, wall_hit=wall,
                     E_H=p[0], E_1=p[1], eta_1=p[2], eta_a=p[3]))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "achoma_upper_relax.csv"), index=False, encoding="utf-8-sig")
print("\nsaved achoma_upper_relax.csv")
print(df.to_string(index=False))
