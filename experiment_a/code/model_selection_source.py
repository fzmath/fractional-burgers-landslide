# -*- coding: utf-8 -*-
"""
加速蠕变数据的模型选择：线性 vs 整数阶 Burgers vs 分数阶 Burgers
================================================================
由 G:\\2026分数阶\\Gilman\\code\\model_selection.py 适配而来。

模型（同 Gilman，恒应力 sigma0=1）：
  M1 线性:         eps = a*t + b                                          (2参数)
  M2 整数Burgers:  eps = s/E_H + s/E1(1-e^{-E1 t/eta1}) + s*t/eta_a      (4参数, alpha=1)
  M3 分数Burgers:  eps = s/E_H + s/E1(1-e^{-E1 t/eta1}) + s*t^alpha/(eta_a Gamma(a+1))  (5参数)

相对 Gilman 版的适配（详见 ../反演结果与模型选择记录.md）：
  1. BASE/CSV/OUT 指向 AcceleratingCreep（6746 点，t∈[0,9210]d）。
  2. M3 的 alpha 上界 0.999 -> 3.0：加速蠕变段为超线性增长，
     Abel 黏壶 t^alpha/Gamma(alpha+1) 在 alpha>1 时表达加速；
     同时下界保留 0.02，alpha<1 方向（次扩散/记忆效应）一并检验。
  3. eta1/etaa 上界 1e9 -> 1e13：t 长达 9210d，alpha=3 时 t^3/Gamma(4)~1.3e11，
     若 eta_a 仍卡在 1e9 会把大 alpha 解人为截断。
  4. M2/M3 改为多初值多起点最小二乘（least_squares 局部收敛），
     M3 初值 alpha 取 [0.3,0.7,1.0,1.5,2.2,2.8] 覆盖次线性与超线性两方向，
     取 RSS 最小者，避免单点初值错过全局解。
"""
import numpy as np, pandas as pd, os, time
from scipy.optimize import least_squares
from scipy.special import gamma as Gamma_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"G:\2026分数阶\AcceleratingCreep"
CSV  = BASE + r"\processed\timeseries_clean.csv"
OUT  = BASE + r"\out"
os.makedirs(OUT, exist_ok=True)
SIGMA0 = 1.0

df = pd.read_csv(CSV)
t   = df["t"].values.astype(float)
eps = df["strain"].values.astype(float)
n   = len(t)
d = np.diff(eps)
noise = max(np.std(d)/np.sqrt(2.0), 1e-12)
print(f"数据: n={n}, t in [{t[0]:.0f},{t[-1]:.0f}] d, "
      f"strain in [{eps.min():.4f},{eps.max():.4f}], 标定噪声={noise:.3e}", flush=True)

def aic_bic(n, k, rss, noise):
    ll  = -0.5*n*(np.log(2*np.pi*noise**2) + 1) - rss/(2*noise**2)
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return aic, bic

results = []

# ---------- M1 线性 ----------
A = np.vstack([t, np.ones_like(t)]).T
coef, *_ = np.linalg.lstsq(A, eps, rcond=None)
pred = A @ coef
rss  = np.sum((eps-pred)**2)
r2   = 1 - rss/np.sum((eps-eps.mean())**2)
aic, bic = aic_bic(n, 2, rss, noise)
results.append(dict(model="M1 线性",
                    params=f"a={coef[0]:.4e}, b={coef[1]:.4e}",
                    r2=r2, rmse=np.sqrt(rss/n), aic=aic, bic=bic))
print(f"M1 done: R2={r2:.5f} RMSE={np.sqrt(rss/n):.3e} AIC={aic:.2f}", flush=True)

# ---------- M2 整数阶 Burgers (alpha=1, 4参数) ----------
def m2(p):
    EH, E1, eta1, etaa = p
    return SIGMA0/EH + SIGMA0/E1*(1-np.exp(-E1*t/eta1)) + SIGMA0*t/etaa
def resid2(p):
    return (m2(p)-eps)/noise

b2 = ([1e-2, 1e-2, 1e0, 1e0], [1e8, 1e8, 1e13, 1e13])   # 适配3: eta 上界放宽
best2 = None
t0 = time.time()
for init in [(1e4,1e4,1e5,1e5), (1e3,1e3,1e4,1e4), (5e4,5e4,1e6,1e6)]:
    sol = least_squares(resid2, list(init), bounds=b2, max_nfev=20000,
                        xtol=1e-12, ftol=1e-12)
    if best2 is None or sol.cost < best2.cost:
        best2 = sol
sol2 = best2
pred = m2(sol2.x); rss = np.sum((eps-pred)**2)
r2 = 1 - rss/np.sum((eps-eps.mean())**2)
aic, bic = aic_bic(n, 4, rss, noise)
results.append(dict(model="M2 整数Burgers",
                    params=(f"EH={sol2.x[0]:.4g},E1={sol2.x[1]:.4g},"
                            f"eta1={sol2.x[2]:.4g},etaa={sol2.x[3]:.4g}"),
                    r2=r2, rmse=np.sqrt(rss/n), aic=aic, bic=bic))
print(f"M2 done in {time.time()-t0:.1f}s: R2={r2:.5f} RMSE={np.sqrt(rss/n):.3e} "
      f"AIC={aic:.2f} cost={sol2.cost:.3e}", flush=True)

# ---------- M3 分数阶 Burgers (5参数, alpha∈[0.02,3.0]) ----------
def m3(p):
    EH, E1, eta1, etaa, alpha = p
    return (SIGMA0/EH + SIGMA0/E1*(1-np.exp(-E1*t/eta1))
            + SIGMA0*t**alpha/(etaa*Gamma_fn(alpha+1)))
def resid3(p):
    return (m3(p)-eps)/noise

# 适配2: alpha 上界 0.999 -> 3.0；适配3: eta 上界 -> 1e13
b3 = ([1e-2, 1e-2, 1e0, 1e0, 0.02], [1e8, 1e8, 1e13, 1e13, 3.0])
best3 = None
t0 = time.time()
for a0 in [0.3, 0.7, 1.0, 1.5, 2.2, 2.8]:   # 适配4: 多起点覆盖两方向
    init = [1e4, 1e4, 1e5, 1e5, a0]
    sol = least_squares(resid3, init, bounds=b3, max_nfev=20000,
                        xtol=1e-12, ftol=1e-12)
    print(f"  M3 init_alpha={a0}: cost={sol.cost:.4e}, "
          f"alpha*={sol.x[4]:.4f}, etaa*={sol.x[3]:.3e}", flush=True)
    if best3 is None or sol.cost < best3.cost:
        best3 = sol
sol3 = best3
pred = m3(sol3.x); rss = np.sum((eps-pred)**2)
r2 = 1 - rss/np.sum((eps-eps.mean())**2)
aic, bic = aic_bic(n, 5, rss, noise)
results.append(dict(model="M3 分数Burgers",
                    params=(f"EH={sol3.x[0]:.4g},E1={sol3.x[1]:.4g},"
                            f"eta1={sol3.x[2]:.4g},etaa={sol3.x[3]:.4g},"
                            f"alpha={sol3.x[4]:.4f}"),
                    r2=r2, rmse=np.sqrt(rss/n), aic=aic, bic=bic))
print(f"M3 done in {time.time()-t0:.1f}s: R2={r2:.5f} RMSE={np.sqrt(rss/n):.3e} "
      f"AIC={aic:.2f} alpha*={sol3.x[4]:.4f}", flush=True)

# ---------- 汇总 ----------
res = pd.DataFrame(results)
print("\n" + "="*90)
print("模型选择（Gradenbach/Eggerberg 加速蠕变, n=%d）" % n)
print("="*90)
print(res.to_string(index=False))
res.to_csv(OUT + r"\model_selection.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, eps, "k.", ms=3, label="观测")
ax.plot(t, A @ np.linalg.lstsq(A, eps, rcond=None)[0], "-",  lw=1.2, label="M1 线性")
ax.plot(t, m2(sol2.x), "--", lw=1.3, label="M2 整数Burgers")
ax.plot(t, m3(sol3.x), "-.", lw=1.6,
        label=f"M3 分数Burgers (α={sol3.x[4]:.3f})")
ax.set_xlabel("t / day")
ax.set_ylabel("strain")
ax.set_title("Gradenbach/Eggerberg 加速蠕变应变: 三模型拟合对比")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT + r"\model_compare.png", dpi=150)
plt.close(fig)
print("\n图已保存:", OUT + r"\model_compare.png", flush=True)
