# -*- coding: utf-8 -*-
"""
后处理：读取 raw_results_stage1.csv（或 raw_results.csv），
聚合 results_all_cells.csv + 4 张图 + env/runtime 补充。
与 run_experiment.py 中聚合/绘图逻辑一致。
"""
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文字体修复（仅绘图渲染，不改任何统计；run_experiment.py 出图时缺 CJK 字形，此处重出图时补上）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "out")
CODE = os.path.join(ROOT, "code")
T = 9000.0

params = json.load(open(os.path.join(CODE,"params.json"), encoding="utf-8"))

# 优先用含阶段(ii)的全量，否则退回 stage1
for cand in ["raw_results.csv", "raw_results_stage1.csv"]:
    p = os.path.join(OUT, cand)
    if os.path.exists(p):
        raw = pd.read_csv(p)
        print("读取", cand, "行数", len(raw))
        break

# 聚合
def agg(df):
    n=len(df); ok=df["fit_ok"].astype(bool)
    fail=1-ok.mean(); d=df[ok]
    if len(d)==0:
        return dict(reps=n,fit_fail_rate=fail,AIC_acc=np.nan,BIC_acc=np.nan,
            P_M1_AIC=np.nan,P_M2_AIC=np.nan,P_M3_AIC=np.nan,
            P_M1_BIC=np.nan,P_M2_BIC=np.nan,P_M3_BIC=np.nan,
            alpha_est_mean=np.nan,alpha_est_std=np.nan,
            alpha_bias_mean=np.nan,alpha_bias_std=np.nan,boundary_hit_rate=np.nan)
    p=d["AIC_choice"].value_counts(normalize=True)
    q=d["BIC_choice"].value_counts(normalize=True)
    ae=d["alpha_est"].dropna()
    bh=((ae-0.02).abs()<=1e-3)|((ae-3.0).abs()<=1e-3)
    return dict(reps=n,fit_fail_rate=fail,
        P_M1_AIC=p.get(0,0),P_M2_AIC=p.get(1,0),P_M3_AIC=p.get(2,0),
        P_M1_BIC=q.get(0,0),P_M2_BIC=q.get(1,0),P_M3_BIC=q.get(2,0),
        alpha_est_mean=ae.mean(),alpha_est_std=ae.std(ddof=1) if len(ae)>1 else 0.0,
        alpha_bias_mean=np.nan,alpha_bias_std=np.nan,
        boundary_hit_rate=bh.mean() if len(ae)>0 else np.nan)

rows=[]
for (fam,noise,dn,ln), sub in raw.groupby(["family","noise","density","length"]):
    a=agg(sub)
    tm=params["families"][fam]["true_model"]
    if tm=="M1":
        a["AIC_acc"]=a["P_M1_AIC"]; a["BIC_acc"]=a["P_M1_BIC"]
    elif tm=="M3":
        a["AIC_acc"]=a["P_M3_AIC"]; a["BIC_acc"]=a["P_M3_BIC"]
        at=params["families"][fam]["alpha_true"]
        ae=sub[sub.fit_ok.astype(bool)]["alpha_est"].dropna()
        bias=ae-at
        a["alpha_bias_mean"]=bias.mean() if len(bias)>0 else np.nan
        a["alpha_bias_std"]=bias.std(ddof=1) if len(bias)>1 else 0.0
    else:
        a["AIC_acc"]=np.nan; a["BIC_acc"]=np.nan
    row=dict(family=fam,noise=noise,density=dn,length=ln); row.update(a); rows.append(row)

cells=pd.DataFrame(rows)
cols=["family","noise","density","length","reps","AIC_acc","BIC_acc",
 "P_M1_AIC","P_M2_AIC","P_M3_AIC","P_M1_BIC","P_M2_BIC","P_M3_BIC",
 "alpha_est_mean","alpha_est_std","alpha_bias_mean","alpha_bias_std",
 "boundary_hit_rate","fit_fail_rate"]
cells=cells[cols].sort_values(["family","noise","density","length"])
cells.to_csv(os.path.join(OUT,"results_all_cells.csv"),index=False,encoding="utf-8-sig")
print("results_all_cells.csv 行数",len(cells))

# ---- 图 ----
tfine=np.linspace(0,T,20001)
def mk(fam,t):
    if fam=="near_linear": return 0.11/T*t+0.005
    from scipy.special import gamma as G
    if fam in ("weak_conv","strong_conv","strong2p5"):
        at={"weak_conv":1.3,"strong_conv":3.0,"strong2p5":2.5}[fam]
        EH,E1,eta1=2e6,1e2,5e5
        etaa=params["families"][fam]["eta_a"]
        return 1/EH+1/E1*(1-np.exp(-E1*t/eta1))+t**at/(etaa*G(at+1))
    if fam=="accel_relax":
        return 0.07/T*t+0.005+0.04*(t/1800.0)*np.exp(1-t/1800.0)
Afam=params["A_family"]

# 图1
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
tits={"near_linear":"近线性 (真M1)","weak_conv":"弱凸 α=1.3 (真M3)",
      "strong_conv":"强凸 α=3.0 (真M3)","accel_relax":"加速-松弛 (无真模型)"}
for ax,fam in zip(axes.ravel(),["near_linear","weak_conv","strong_conv","accel_relax"]):
    c=mk(fam,tfine); A=Afam[fam]
    ax.plot(tfine,c,"k-",lw=1.5,label="clean")
    for pp,col in zip([0.01,0.03,0.10],["tab:blue","tab:orange","tab:red"]):
        ax.fill_between(tfine,c-pp*A,c+pp*A,color=col,alpha=0.25,label=f"σ={int(pp*100)}%A")
    ax.set_title(tits[fam]); ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_ylabel("strain")
axes[1,0].set_xlabel("t / day"); axes[1,1].set_xlabel("t / day")
fig.suptitle("图1 四族合成曲线（clean + 1%/3%/10% 噪声带）")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig1_families.png"),dpi=150); plt.close(fig)

# 图2
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,met in zip(axes,["AIC_acc","BIC_acc"]):
    for fam in ["near_linear","weak_conv","strong_conv"]:
        sub=cells[cells.family==fam]
        for dn in ["sparse","medium","dense"]:
            for ln in ["short","mid","long"]:
                s2=sub[(sub.density==dn)&(sub.length==ln)].sort_values("noise")
                if len(s2)==0: continue
                st={"sparse":":","medium":"--","dense":"-"}[dn]
                lw={"short":1.0,"mid":1.6,"long":2.4}[ln]
                ax.plot(s2.noise*100,s2[met],st,lw=lw,marker="o",ms=3,
                        label=f"{fam[:6]}/{dn[:3]}/{ln[:4]}")
    ax.set_xlabel("噪声水平 (% A)"); ax.set_title(("AIC" if met=="AIC_acc" else "BIC")+" 正确率")
    ax.grid(alpha=0.3); ax.set_xticks([1,3,10])
axes[0].set_ylabel("模型选择正确率")
fig.suptitle("图2 真模型选择正确率（线：密度-记录长度）")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig2_accuracy.png"),dpi=150); plt.close(fig)

# 图3
fig,axes=plt.subplots(1,2,figsize=(12,5))
for fam,at in [("weak_conv",1.3),("strong_conv",3.0),("strong2p5",2.5)]:
    sub=cells[cells.family==fam]
    for dn in ["sparse","medium","dense"]:
        for ln in ["short","mid","long"]:
            s2=sub[(sub.density==dn)&(sub.length==ln)].sort_values("noise")
            if len(s2)==0: continue
            st={"sparse":":","medium":"--","dense":"-"}[dn]
            lw={"short":1.0,"mid":1.6,"long":2.4}[ln]
            axes[0].errorbar(s2.noise*100,s2.alpha_bias_mean,yerr=s2.alpha_bias_std,
                fmt=st,lw=lw,marker="o",ms=3,label=f"{fam}/{dn[:3]}/{ln[:4]}")
            axes[1].plot(s2.noise*100,s2.boundary_hit_rate,st,lw=lw,marker="s",ms=3)
axes[0].axhline(0,color="k",lw=0.8)
axes[0].set_xlabel("噪声水平 (%A)"); axes[0].set_ylabel("alpha_est - alpha_true")
axes[0].set_title("alpha 估计偏差 (mean±std)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=6,ncol=2)
axes[1].set_xlabel("噪声水平 (%A)"); axes[1].set_ylabel("边界命中率")
axes[1].set_title("P(|α-0.02|≤1e-3 或 |α-3.0|≤1e-3)"); axes[1].grid(alpha=0.3)
fig.suptitle("图3 凸族 alpha 估计偏差与边界命中率")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig3_alpha_bias.png"),dpi=150); plt.close(fig)

# 图4
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
sub=cells[cells.family=="accel_relax"]
im=None
for ax,ln in zip(axes,["short","mid","long"]):
    s2=sub[sub.length==ln]
    pv=s2.pivot(index="density",columns="noise",values="P_M3_AIC").reindex(["sparse","medium","dense"])
    im=ax.imshow(pv.values,aspect="auto",cmap="Reds",vmin=0,vmax=1)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([f"{int(c*100)}%" for c in pv.columns])
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
    ax.set_title(f"记录长度={ln}")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v=pv.values[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8)
axes[0].set_ylabel("采样密度")
fig.suptitle("图4 加速-松弛族 P(M3)「分数阶误判率」热图 (AIC)")
fig.colorbar(im,ax=axes,fraction=0.025,label="P(M3)")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig4_misjudge.png"),dpi=150); plt.close(fig)
print("4 张图已生成")
