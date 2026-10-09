# -*- coding: utf-8 -*-
"""出图：误差-提前量图 + 代表性截断点拟合外推图 + cut=90 回顾参照。"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t as tdist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
    try:
        font_manager.fontManager.addfont(fp); break
    except Exception: pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 数据根：默认 <仓库>/data；可用环境变量 CREEP_DATA_ROOT 覆盖
_DATA_ROOT = os.environ.get(
    "CREEP_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)
DATA = os.path.join(_DATA_ROOT, "Achoma", "timeseries_clean.csv")
OUT  = os.path.dirname(os.path.abspath(__file__))
TF_TRUE=92.0; L=100.0; V_WIN=7.0
df=pd.read_csv(DATA); t_all=df["t"].values.astype(float); u_all=df["strain"].values.astype(float)*L

def local_velocity(tt,uu,W=V_WIN,minpts=3):
    v=np.full_like(tt,np.nan)
    for i in range(len(tt)):
        m=(tt>=tt[i]-W)&(tt<=tt[i]+W)
        if m.sum()<minpts: continue
        v[i]=np.polyfit(tt[m]-tt[i],uu[m],2)[1]
    return v

def fit_main(tt,vv):
    m=np.isfinite(vv)&(vv>1e-4); tt,vv=tt[m],vv[m]; n=len(tt)
    if n<5: return None
    tlast=tt.max()
    def resid(p):
        A,nn,tf=p
        if tf<=tlast+1e-3: return np.full(n,1e6)
        return (A*(tf-tt)**(-nn)-vv)/np.sqrt(vv)
    best=None
    for A0 in [0.05,0.5,3.0]:
        for n0 in [0.5,1.5,2.5,3.5]:
            for tf0 in [tlast+3,tlast+10,tlast+25]:
                try:
                    s=least_squares(resid,[A0,n0,tf0],
                        bounds=([1e-4,0.05,tlast+0.5],[100,5.0,tlast+35]),
                        xtol=1e-8,ftol=1e-8,max_nfev=5000)
                    if best is None or s.cost<best.cost: best=s
                except: pass
    return best

def windowed(cut,W):
    mc=t_all<=cut+1e-9; tc=t_all[mc]; uc=u_all[mc]; vc=local_velocity(tc,uc)
    lo=cut-W; mw=(tc>=lo-1e-9)&(tc<=cut+1e-9)
    return tc[mw],vc[mw],tc,vc

# ---------- 图1：误差-提前量 ----------
res=pd.read_csv(OUT+r"\results_table.csv")
fig,ax=plt.subplots(figsize=(8,5.5))
for meth,col,mk in [("主_Fukuzono_NLS","#c0392b","o"),("对照_逆速度线性","#2c3e50","s")]:
    d=res[res.method==meth]
    ax.errorbar(d.lead, np.abs(d.bias), yerr=None, fmt=mk+"-", color=col,
                ms=6, lw=1.2, capsize=3, label=("主方法 非线性Fukuzono" if "主" in meth else "对照 逆速度线性"))
ax.axhline(2.0,color="green",ls=":",lw=1.2,label="±2 d 阈值")
ax.axhline(5.0,color="orange",ls="--",lw=1.2,label="±5 d 阈值")
ax.set_xlabel("提前量 = 92.0 − 截断时刻 t_cut  (d)")
ax.set_ylabel("|t_f 估计 − 92.0|  (d)")
ax.set_title("实验B：滚动预测误差 vs 提前量（三档窗口汇总）")
ax.invert_xaxis(); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT+r"\fig1_误差-提前量.png",dpi=150); plt.close(fig)
print("图1已存")

# ---------- 图2：代表性截断点拟合外推 ----------
reps=[(60,20,"提前32 d"),(75,20,"提前17 d"),(85,20,"提前7 d"),(89,20,"提前3 d")]
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=False)
for ax,(cut,W,lab) in zip(axes.ravel(),reps):
    tw,vw,_,_=windowed(cut,W)
    sol=fit_main(tw,vw)
    ax.plot(tw,vw,"k.",ms=7,label="平滑速度 v")
    if sol:
        A,nn,tf=sol.x
        xx=np.linspace(tw.min()+0.5, tf-0.3, 200)
        ax.plot(xx, A*(tf-xx)**(-nn),"-",color="#c0392b",lw=1.6,
                label=f"拟合 v=A(tf-t)^-n\ntf={tf:.1f}d, n={nn:.2f}")
        ax.axvline(tf,color="#c0392b",ls="--",lw=1.2)
    ax.axvline(TF_TRUE,color="green",ls=":",lw=1.8,label="真实 tf=92.0")
    ax.axvline(cut,color="gray",ls="-",lw=1,alpha=.6)
    ax.set_title(f"截断 t={cut} d（{lab}，末{W}d窗）")
    ax.set_xlabel("t / d"); ax.set_ylabel("v / (m/d)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("代表性截断点：平滑速度与 Fukuzono 外推（主方法）")
fig.tight_layout(); fig.savefig(OUT+r"\fig2_代表性拟合外推.png",dpi=150); plt.close(fig)
print("图2已存")

# ---------- 图3：cut=90 回顾参照（全数据） ----------
fig,ax=plt.subplots(figsize=(9,5.5))
# 全数据速度
v_all=local_velocity(t_all,u_all)
ax.plot(t_all,u_all,"k.-",label="位移 u(t)",alpha=.5)
ax.set_ylabel("u / m",color="k"); ax.set_xlabel("t / d")
ax2=ax.twinx()
ax2.plot(t_all,v_all,"b.-",label="平滑速度 v")
# 用 [60,90] 拟合
tw=t_all[t_all>=60]; vw=v_all[t_all>=60]
sol=fit_main(tw,vw)
if sol:
    A,nn,tf=sol.x
    xx=np.linspace(60.5,tf-0.3,200)
    ax2.plot(xx,A*(tf-xx)**(-nn),"r-",lw=1.8,label=f"全数据拟合 tf={tf:.1f}d, n={nn:.2f}")
    print(f"[cut=90回顾] 用[60,90]拟合: A={A:.3g} n={nn:.3f} tf={tf:.2f} (偏差{tf-92:+.2f})")
ax2.axvline(TF_TRUE,color="green",ls="--",label="真实 tf=92.0")
ax2.set_ylabel("v / (m/d)",color="b")
ax.set_title("回顾性参照：用全部失稳前数据（≤t=90）拟合能否复现 ~92 d")
ax.grid(alpha=.3)
lines1,lab1=ax.get_legend_handles_labels(); lines2,lab2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,lab1+lab2,fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig(OUT+r"\fig3_回顾性全数据拟合.png",dpi=150); plt.close(fig)
print("图3已存")

# ---------- 汇总：每截断点主方法三窗 tf 均值/范围 ----------
main=res[res.method.str.startswith("主")].copy()
summ=main.groupby("cut").agg(
    lead=("lead","first"),
    tf_mean=("tf","mean"), tf_min=("tf","min"), tf_max=("tf","max"),
    bias_mean=("bias","mean"),
    n_good5=("bias",lambda s:(s.abs()<=5).sum())).reset_index()
print("\n=== 主方法按截断点汇总（三窗）===")
print(summ.to_string(index=False))
summ.to_csv(OUT+r"\summary_by_cut.csv",index=False,encoding="utf-8-sig")
