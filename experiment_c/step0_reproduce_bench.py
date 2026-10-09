# -*- coding: utf-8 -*-
"""
step0: 复现四案例 M3 最优解 + 基准计时（验证 common.py 口径与已有结果一致）。
输出: reproduce_bench.csv
"""
import os, time, json
import numpy as np, pandas as pd
import common as C

rows = []
for name in C.select_cases():
    c = C.load_case(name)
    t0 = time.time()
    sol = C.fit_m3(c, max_nfev=5000, xtol=1e-8, ftol=1e-8)
    dt = time.time() - t0
    p = sol.x
    rss = C.rss_from_sol(sol, c["noise"])
    r2 = C.r2_from_rss(rss, c["eps"])
    aic, bic = C.aic_bic(c["n"], 5, rss, c["noise"])
    hits = C.boundary_hits(p, c["lo"], c["hi"])
    hit_str = ";".join([f"{h['param']}{'L' if h['hit_lo'] else ''}{'U' if h['hit_hi'] else ''}"
                        for h in hits if h["hit_lo"] or h["hit_hi"]])
    print(f"[{name}] n={c['n']} t=[{c['t'][0]:.0f},{c['t'][-1]:.0f}] "
          f"noise={c['noise']:.3e} 耗时={dt:.1f}s", flush=True)
    print(f"   E_H={p[0]:.4g} E_1={p[1]:.4g} eta_1={p[2]:.4g} "
          f"eta_a={p[3]:.4g} alpha={p[4]:.4f}", flush=True)
    print(f"   R2={r2:.6f} RMSE={np.sqrt(rss/c['n']):.3e} AIC={aic:.2f} BIC={bic:.2f}", flush=True)
    print(f"   贴界参数: {hit_str if hit_str else '无'}", flush=True)
    row = dict(case=name, n=c["n"], t_end=c["t"][-1], noise=c["noise"],
               E_H=p[0], E_1=p[1], eta_1=p[2], eta_a=p[3], alpha=p[4],
               R2=r2, RMSE=np.sqrt(rss/c["n"]), AIC=aic, BIC=bic,
               wall_hits=hit_str, time_s=dt)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce_bench.csv"),
          index=False, encoding="utf-8-sig")
print("\nsaved reproduce_bench.csv")
