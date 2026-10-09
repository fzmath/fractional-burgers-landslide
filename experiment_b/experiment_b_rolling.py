# -*- coding: utf-8 -*-
"""
实验B：Achoma 滑坡实时滚动预测（回应审稿人"事后选窗"质疑）
正式版。
主方法：非线性三参数 Fukuzono 速度模型 v(t)=A*(tf-t)^(-n)（A>0,n>0）。
  - 位移→速度：在截断全集[0,t_cut]上做局部二次滑动窗（半宽 W=7d）求导（因果，无未来信息）；
    再按窗口[t_cut-W,t_cut]取点拟合。
  - 加权：相对残差 (vpred-vv)/sqrt(vv)（幂律误差为乘性）。
  - 多起点 least_squares，xtol/ftol=1e-8, max_nfev=5000。
对照方法：逆速度线性法（斋藤/Fukuzono alpha=1）1/v = a + b t -> tf = -a/b。
预测区间：delta 法（残差雅可比协方差，t 分布 95%）。
窗口规则预先固定：截断点 60/65/70/75/80/85/88/89；末 30/20/14 d 三档。
"""
import numpy as np, pandas as pd, os, time
from scipy.optimize import least_squares
from scipy.stats import t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 数据根：默认 <仓库>/data；可用环境变量 CREEP_DATA_ROOT 覆盖
_DATA_ROOT = os.environ.get(
    "CREEP_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)
DATA = os.path.join(_DATA_ROOT, "Achoma", "timeseries_clean.csv")
OUT  = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
TF_TRUE = 92.0
L_DISP  = 100.0
V_WIN   = 7.0
CUTS    = [60, 65, 70, 75, 80, 85, 88, 89]
WINDOWS = [30, 20, 14]

df = pd.read_csv(DATA)
t_all = df["t"].values.astype(float)
u_all = df["strain"].values.astype(float) * L_DISP
print(f"[数据] n={len(t_all)}, t[{t_all.min():.0f},{t_all.max():.0f}] d, "
      f"u[{u_all.min():.3f},{u_all.max():.3f}] m, tf_true={TF_TRUE}", flush=True)

def local_velocity(tt, uu, W=V_WIN, minpts=3):
    vv = np.full_like(tt, np.nan)
    for i in range(len(tt)):
        m = (tt >= tt[i]-W) & (tt <= tt[i]+W)
        if m.sum() < minpts:
            continue
        x = tt[m]-tt[i]; y = uu[m]
        vv[i] = np.polyfit(x, y, 2)[1]
    return vv

def fit_main(tt, vv):
    m = np.isfinite(vv) & (vv > 1e-4)
    tt, vv = tt[m], vv[m]
    n = len(tt)
    if n < 5:
        return None, n
    tlast = tt.max()
    def resid(p):
        A, nn, tf = p
        if tf <= tlast + 1e-3:
            return np.full(n, 1e6)
        return (A*(tf-tt)**(-nn) - vv) / np.sqrt(vv)
    best = None
    for A0 in [0.05, 0.5, 3.0]:
        for n0 in [0.5, 1.5, 2.5, 3.5]:
            for tf0 in [tlast+3, tlast+10, tlast+25]:
                try:
                    s = least_squares(resid, [A0,n0,tf0],
                        bounds=([1e-4,0.05,tlast+0.5],[100,5.0,tlast+35]),
                        xtol=1e-8, ftol=1e-8, max_nfev=5000)
                    if best is None or s.cost < best.cost:
                        best = s
                except Exception:
                    pass
    if best is None:
        return None, n
    A, nn, tf = best.x
    hit = bool(tf <= tlast+0.6 or tf >= tlast+34.4 or nn <= 0.06 or nn >= 4.9)
    ci_lo=ci_hi=np.nan
    try:
        J = best.jac; dof=max(n-3,1)
        rss = 2*best.cost; sigma2 = rss/dof
        cov = sigma2*np.linalg.inv(J.T@J)
        se = np.sqrt(max(cov[2,2],0.0)); tcrit=tdist.ppf(0.975,dof)
        ci_lo=tf-tcrit*se; ci_hi=tf+tcrit*se
    except Exception:
        pass
    return dict(A=A,n_exp=nn,tf=tf,ci_lo=ci_lo,ci_hi=ci_hi,hit=hit,
                n_fit=n,tlast=tlast,tt=tt,vv=vv), n

def fit_ctrl(tt, vv):
    m = np.isfinite(vv) & (vv>1e-4)
    tt, vv = tt[m], vv[m]; n=len(tt)
    if n<5: return None, n
    invv=1.0/vv
    b,a = np.polyfit(tt,invv,1)
    tf=-a/b
    ci_lo=ci_hi=np.nan
    try:
        yhat=b*tt+a; r=invv-yhat; dof=max(n-2,1)
        s2=np.sum(r**2)/dof
        X=np.vstack([tt,np.ones_like(tt)]).T
        covab=s2*np.linalg.inv(X.T@X)
        g=np.array([a/b**2, -1.0/b])
        var=g@covab@g; tcrit=tdist.ppf(0.975,dof); se=np.sqrt(max(var,0.0))
        ci_lo=tf-tcrit*se; ci_hi=tf+tcrit*se
    except Exception:
        pass
    return dict(tf=tf,slope=b,intercept=a,ci_lo=ci_lo,ci_hi=ci_hi,n_fit=n,tt=tt,vv=vv), n

rows=[]; t0=time.time()
for cut in CUTS:
    # 截断全集（因果）
    mc = t_all <= cut+1e-9
    tc = t_all[mc]; uc = u_all[mc]
    vc = local_velocity(tc, uc, W=V_WIN)
    for W in WINDOWS:
        lo = cut-W
        mw = (tc>=lo-1e-9)&(tc<=cut+1e-9)
        tw, vw = tc[mw], vc[mw]
        lead = TF_TRUE-cut
        rb = dict(cut=cut,window=W,lo=lo,lead=lead,n_raw=int(mw.sum()))
        rm,_ = fit_main(tw,vw)
        rc,_ = fit_ctrl(tw,vw)
        if rm:
            rows.append(dict(**rb,method="主_Fukuzono_NLS",tf=rm["tf"],
                bias=rm["tf"]-TF_TRUE,n_exp=rm["n_exp"],A=rm["A"],
                ci_lo=rm["ci_lo"],ci_hi=rm["ci_hi"],bound=rm["hit"],
                cov92=(rm["ci_lo"]<=TF_TRUE<=rm["ci_hi"]),n_fit=rm["n_fit"]))
        else:
            rows.append(dict(**rb,method="主_Fukuzono_NLS",tf=np.nan,bias=np.nan,
                n_exp=np.nan,A=np.nan,ci_lo=np.nan,ci_hi=np.nan,bound=True,
                cov92=False,n_fit=0))
        if rc:
            rows.append(dict(**rb,method="对照_逆速度线性",tf=rc["tf"],
                bias=rc["tf"]-TF_TRUE,n_exp=np.nan,A=np.nan,
                ci_lo=rc["ci_lo"],ci_hi=rc["ci_hi"],bound=False,
                cov92=(rc["ci_lo"]<=TF_TRUE<=rc["ci_hi"]),n_fit=rc["n_fit"]))
        else:
            rows.append(dict(**rb,method="对照_逆速度线性",tf=np.nan,bias=np.nan,
                n_exp=np.nan,A=np.nan,ci_lo=np.nan,ci_hi=np.nan,bound=True,
                cov92=False,n_fit=0))
    print(f"  cut={cut:>3} done", flush=True)

res=pd.DataFrame(rows).sort_values(["cut","window","method"]).reset_index(drop=True)
res.to_csv(OUT+r"\results_table.csv",index=False,encoding="utf-8-sig")
pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
print(f"\n[耗时 {time.time()-t0:.0f}s]\n")
print(res[["cut","window","n_raw","lead","method","tf","bias","n_exp","bound","ci_lo","ci_hi","cov92"]].to_string(index=False))

main=res[res.method.str.startswith("主")]; ctrl=res[res.method.str.startswith("对照")]
print(f"\n主方法覆盖率: {main.cov92.mean()*100:.0f}% ({int(main.cov92.sum())}/{len(main)})")
print(f"对照覆盖率: {ctrl.cov92.mean()*100:.0f}% ({int(ctrl.cov92.sum())}/{len(ctrl)})")
print("\n主方法 |bias|<=5 的格子:")
print(main[main.bias.abs()<=5][["cut","window","tf","bias","n_exp","bound"]].to_string(index=False))
