# -*- coding: utf-8 -*-
"""
实验A：合成数据下的适用边界验证
================================================================
论文《分数阶 Burgers 模型在滑坡失稳预警中的适用边界》大修——审稿证据。

复用框架约定：本脚本中的 M1/M2/M3 模型、bounds、初值、noise 估计、
AIC/BIC 公式、least_squares 参数，与
  G:\\2026分数阶\\AcceleratingCreep\\code\\model_selection.py
逐行一致（只读副本见 code\\model_selection_source.py）。
本脚本不 import 该源文件（其导入即读真实数据并拟合）。

设计：
  4 主族 (+1 补充族 strong2p5) × 3 噪声 × 3 密度 × 3 长度 = 135 格。
  阶段(i)：每格 30 重复；阶段(ii)：关键格加密至 100 重复。
  种子：(族,噪声,密度,长度,重复) 的确定性 md5 函数，可复现。

收敛设置偏差说明（本脚本唯一相对源框架的改动）：
  收敛容差相对源框架放宽（xtol/ftol 1e-12→1e-8, max_nfev 20000→5000）：
  实测在 multi-scale ill-conditioned 参数空间下 1e-12 容差不可达，拟合烧满
  nfev 截断（稀疏41点单作业~40s），放宽后按容差真正收敛且总耗时可控；
  该差异不影响模型选择结论（RSS 相对精度 ~1e-8，AIC 差异远大于此）。
  模型定义、bounds、初值、AIC/BIC、种子、网格、统计全部与源框架逐行一致，
  未改动。
"""
import os, sys, json, time, hashlib, platform
# 限制底层线性代数线程，避免 10 进程并行时过度订阅（须在 import numpy 前设置）
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as Gamma_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
OUT  = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 全局常数（与源框架逐行一致）
# ----------------------------------------------------------------------
SIGMA0 = 1.0
T = 9000.0                      # 与真实案例 t∈[0,9210]d 同量级
# bounds（与 model_selection.py 一致）
b2 = ([1e-2, 1e-2, 1e0, 1e0], [1e8, 1e8, 1e13, 1e13])
b3 = ([1e-2, 1e-2, 1e0, 1e0, 0.02], [1e8, 1e8, 1e13, 1e13, 3.0])
M2_INITS = [(1e4,1e4,1e5,1e5), (1e3,1e3,1e4,1e4), (5e4,5e4,1e6,1e6)]
M3_ALPHA_INITS = [0.3, 0.7, 1.0, 1.5, 2.2, 2.8]
LSQ_KW = dict(max_nfev=5000, xtol=1e-8, ftol=1e-8)

# 网格配置
NOISE_P = [0.01, 0.03, 0.10]                 # 低/中/高
DENSITIES = {"sparse": 41, "medium": 201, "dense": 3001}   # 3001 为任务允许的降采样
LENGTH_FRACS = {"short": 1/3, "mid": 2/3, "long": 1.0}
REPS_BASE = 30
REPS_KEY  = 100
N_WORKERS = 10

# ----------------------------------------------------------------------
# 合成曲线族（无噪声 clean）
# ----------------------------------------------------------------------
def burgers_frac(t, EH, E1, eta1, alpha, etaa):
    """M3 数学形式（无噪声）。"""
    return (SIGMA0/EH
            + SIGMA0/E1*(1.0 - np.exp(-E1*t/eta1))
            + SIGMA0*t**alpha/(etaa*Gamma_fn(alpha+1.0)))

def curve_near_linear(t):
    a = 0.11/T
    b = 0.005
    return a*t + b

def curve_weak_conv(t):
    # alpha_true=1.3, eta_a=1e6
    return burgers_frac(t, 2e6, 1e2, 5e5, 1.3, 1e6)

def curve_strong_conv(t):
    # alpha_true=3.0, eta_a 使 t=T 处幂项≈0.10（在代码内标定）
    eta_a = _etaa_for_power(T, 3.0, 0.10)
    return burgers_frac(t, 2e6, 1e2, 5e5, 3.0, eta_a), eta_a

def curve_strong2p5(t):
    eta_a = _etaa_for_power(T, 2.5, 0.10)
    return burgers_frac(t, 2e6, 1e2, 5e5, 2.5, eta_a), eta_a

def curve_accel_relax(t):
    # 线性背景 + 单峰脉冲 g(t)=(t/tau_p)*exp(1-t/tau_p)
    tau_p = 1800.0
    h = 0.04
    a = 0.07/T
    b = 0.005
    g = (t/tau_p)*np.exp(1.0 - t/tau_p)
    return a*t + b + h*g

def _etaa_for_power(t_end, alpha, target=0.10):
    """幂项 s*t^alpha/(eta_a*Gamma(alpha+1)) 在 t=t_end 处等于 target -> eta_a。"""
    return (t_end**alpha)/(target*Gamma_fn(alpha+1.0))

# ----------------------------------------------------------------------
# 单格拟合（顶层函数，供 multiprocessing 调用）
# ----------------------------------------------------------------------
def aic_bic(n, k, rss, noise):
    ll  = -0.5*n*(np.log(2*np.pi*noise**2) + 1.0) - rss/(2*noise**2)
    aic = 2*k - 2*ll
    bic = np.log(n)*k - 2*ll
    return aic, bic

def fit_one(job):
    """job = (t, eps_clean, sigma, rep_seed)。返回一行 dict。"""
    t, eps_clean, sigma, rep_seed = job
    rng = np.random.default_rng(rep_seed)
    eps = eps_clean + rng.normal(0.0, sigma, size=len(t))
    n = len(t)
    # noise：每个数据集各自从一阶差分估计（与源框架一致）
    noise = max(np.std(np.diff(eps))/np.sqrt(2.0), 1e-12)

    fit_ok = True
    try:
        # ---- M1 线性（闭式最小二乘）----
        A = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, eps, rcond=None)
        pred1 = A @ coef
        rss1 = float(np.sum((eps-pred1)**2))
        aic1, bic1 = aic_bic(n, 2, rss1, noise)

        # ---- M2 整数 Burgers（3 初值）----
        def m2(p):
            EH, E1, eta1, etaa = p
            return (SIGMA0/EH + SIGMA0/E1*(1-np.exp(-E1*t/eta1))
                    + SIGMA0*t/etaa)
        def resid2(p): return (m2(p)-eps)/noise
        best2 = None
        for init in M2_INITS:
            sol = least_squares(resid2, list(init), bounds=b2, **LSQ_KW)
            if best2 is None or sol.cost < best2.cost:
                best2 = sol
        if not best2.success:
            fit_ok = False
        pred2 = m2(best2.x); rss2 = float(np.sum((eps-pred2)**2))
        aic2, bic2 = aic_bic(n, 4, rss2, noise)

        # ---- M3 分数 Burgers（6 alpha 初值）----
        def m3(p):
            EH, E1, eta1, etaa, alpha = p
            return (SIGMA0/EH + SIGMA0/E1*(1-np.exp(-E1*t/eta1))
                    + SIGMA0*t**alpha/(etaa*Gamma_fn(alpha+1.0)))
        def resid3(p): return (m3(p)-eps)/noise
        best3 = None
        for a0 in M3_ALPHA_INITS:
            init = [1e4, 1e4, 1e5, 1e5, a0]
            sol = least_squares(resid3, init, bounds=b3, **LSQ_KW)
            if best3 is None or sol.cost < best3.cost:
                best3 = sol
        if not best3.success:
            fit_ok = False
        alpha_est = float(best3.x[4])
        pred3 = m3(best3.x); rss3 = float(np.sum((eps-pred3)**2))
        aic3, bic3 = aic_bic(n, 5, rss3, noise)
    except Exception:
        return dict(aic_choice=-1, bic_choice=-1, alpha_est=np.nan, fit_ok=False)

    aics = [aic1, aic2, aic3]; bics = [bic1, bic2, bic3]
    aic_choice = int(np.argmin(aics))   # 0=M1,1=M2,2=M3
    bic_choice  = int(np.argmin(bics))
    return dict(aic_choice=aic_choice, bic_choice=bic_choice,
                alpha_est=alpha_est, fit_ok=bool(fit_ok))

# ----------------------------------------------------------------------
# 种子（确定性 md5）
# ----------------------------------------------------------------------
def make_seed(family, noise_p, density, length, rep):
    s = f"{family}|{noise_p:.4f}|{density}|{length}|{rep}"
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")

# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    t_start = time.time()
    rt = []   # runtime 记录
    def log(msg):
        dt = time.time()-t_start
        line = f"[{dt:8.1f}s] {msg}"
        print(line, flush=True)
        rt.append(line)

    # ---- 1. 标定四族参数与峰峰幅度 ----
    tfine = np.linspace(0, T, 20001)
    params = {"T": T, "sigma0": SIGMA0,
              "densities": DENSITIES, "length_fracs": LENGTH_FRACS,
              "noise_p": NOISE_P, "reps_base": REPS_BASE, "reps_key": REPS_KEY,
              "bounds_M2": [list(b2[0]), list(b2[1])],
              "bounds_M3": [list(b3[0]), list(b3[1])]}

    eta_strong = _etaa_for_power(T, 3.0, 0.10)
    eta_s2p5   = _etaa_for_power(T, 2.5, 0.10)
    curves = {}
    curves["near_linear"]  = curve_near_linear(tfine)
    curves["weak_conv"]    = curve_weak_conv(tfine)
    sc3, _ = curve_strong_conv(tfine); curves["strong_conv"] = sc3
    ss2, _ = curve_strong2p5(tfine);   curves["strong2p5"]   = ss2
    curves["accel_relax"]  = curve_accel_relax(tfine)

    # 幂项在 t=T 处的值（验证）
    def power_term(alpha, eta_a, t=T):
        return t**alpha/(eta_a*Gamma_fn(alpha+1.0))
    params["families"] = {
        "near_linear": dict(true_model="M1", a=0.11/T, b=0.005),
        "weak_conv":   dict(true_model="M3", alpha_true=1.3, EH=2e6, E1=1e2,
                            eta1=5e5, eta_a=1e6,
                            power_at_T=power_term(1.3, 1e6)),
        "strong_conv": dict(true_model="M3", alpha_true=3.0, EH=2e6, E1=1e2,
                            eta1=5e5, eta_a=eta_strong,
                            power_at_T=power_term(3.0, eta_strong)),
        "strong2p5":   dict(true_model="M3", alpha_true=2.5, EH=2e6, E1=1e2,
                            eta1=5e5, eta_a=eta_s2p5,
                            power_at_T=power_term(2.5, eta_s2p5)),
        "accel_relax": dict(true_model="none", a=0.07/T, b=0.005,
                            tau_p=1800.0, h=0.04,
                            note="线性背景+单峰脉冲 g(t)=(t/tau_p)exp(1-t/tau_p)，"
                                 "外部驱动(降雨/融雪)诱发的加速-回落事件，非幂律"),
    }
    # 峰峰幅度
    Afam = {}
    for fam, c in curves.items():
        Afam[fam] = float(c.max()-c.min())
    params["A_family"] = Afam
    for fam, p in params["families"].items():
        p["A_family"] = Afam[fam]
        p["sigma_per_noise"] = {f"{int(pp*100)}%": pp*Afam[fam] for pp in NOISE_P}
    with open(os.path.join(CODE, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    log("参数标定完成: " + json.dumps({k: round(v,5) for k,v in Afam.items()}))

    # ---- 2. 构建全部格的任务清单 ----
    FAMILIES = ["near_linear","weak_conv","strong_conv","accel_relax","strong2p5"]
    def clean_curve(fam, t):
        if fam=="near_linear": return curve_near_linear(t)
        if fam=="weak_conv":    return curve_weak_conv(t)
        if fam=="strong_conv":  return curve_strong_conv(t)[0]
        if fam=="strong2p5":    return curve_strong2p5(t)[0]
        if fam=="accel_relax":  return curve_accel_relax(t)
        raise ValueError(fam)

    # 关键格定义
    key_cells = set()
    for fam in ["weak_conv","strong_conv","accel_relax"]:
        for np_ in NOISE_P:
            for dn in DENSITIES:
                for ln in LENGTH_FRACS:
                    key_cells.add((fam,np_,dn,ln))
    # 近线性代表性：3 噪声 × 3 密度 × 长记录
    for np_ in NOISE_P:
        for dn in DENSITIES:
            key_cells.add(("near_linear",np_,dn,"long"))

    # 预构建每格的 t 与 eps_clean（只算一次）
    cell_data = {}
    for fam in FAMILIES:
        for dn, N in DENSITIES.items():
            t_full = np.linspace(0, T, N)
            for ln, frac in LENGTH_FRACS.items():
                ni = int(round(N*frac))
                tg = t_full[:ni]
                ec = clean_curve(fam, tg)
                for np_ in NOISE_P:
                    sigma = np_*Afam[fam]
                    cell_data[(fam,np_,dn,ln)] = (tg, ec, sigma)
    log(f"格总数={len(cell_data)} (5族x3x3x3)")

    # 阶段(i)：每格 30 重复
    jobs_i = []
    for key,(tg,ec,sigma) in cell_data.items():
        fam,np_,dn,ln = key
        for rep in range(REPS_BASE):
            jobs_i.append((tg, ec, sigma, make_seed(fam,np_,dn,ln,rep)))
    log(f"阶段(i) 任务数={len(jobs_i)} (=格数x{REPS_BASE})")

    t0 = time.time()
    log(f"阶段(i) 开始: {time.strftime('%H:%M:%S')}")
    with Pool(N_WORKERS) as pool:
        res_i = pool.map(fit_one, jobs_i, chunksize=8)
    log(f"阶段(i) 结束, 用时 {time.time()-t0:.1f}s")

    # 组装 raw 行（阶段 i）
    raw_rows = []
    idx = 0
    for key in cell_data.keys():
        fam,np_,dn,ln = key
        for rep in range(REPS_BASE):
            r = res_i[idx]; idx += 1
            raw_rows.append(dict(
                rep=rep, family=fam, noise=np_, density=dn, length=ln,
                seed=make_seed(fam,np_,dn,ln,rep),
                true_model=params["families"][fam]["true_model"],
                AIC_choice=r["aic_choice"], BIC_choice=r["bic_choice"],
                alpha_est=r["alpha_est"], fit_ok=r["fit_ok"]))

    # 阶段(i) 结果先增量存盘（防中断丢失）
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(os.path.join(OUT,"raw_results_stage1.csv"), index=False, encoding="utf-8-sig")
    log(f"阶段(i) raw 已存盘 行数={len(raw)}")

    # 阶段(ii)：关键格加密到 100
    jobs_ii = []
    job_meta = []
    for key in key_cells:
        if key not in cell_data: continue
        fam,np_,dn,ln = key
        tg,ec,sigma = cell_data[key]
        for rep in range(REPS_BASE, REPS_KEY):
            jobs_ii.append((tg, ec, sigma, make_seed(fam,np_,dn,ln,rep)))
            job_meta.append(key)
    log(f"阶段(ii) 加密任务数={len(jobs_ii)} (关键格每格补到{REPS_KEY})")
    t0 = time.time()
    log(f"阶段(ii) 开始: {time.strftime('%H:%M:%S')}")
    with Pool(N_WORKERS) as pool:
        res_ii = pool.map(fit_one, jobs_ii, chunksize=8)
    log(f"阶段(ii) 结束, 用时 {time.time()-t0:.1f}s")

    # 追加阶段(ii) 结果（rep 编号按格递增）
    raw_rows_ii = []
    counter = {k: REPS_BASE for k in key_cells}
    for key, r in zip(job_meta, res_ii):
        fam,np_,dn,ln = key
        rep = counter[key]; counter[key] += 1
        raw_rows_ii.append(dict(
            rep=rep, family=fam, noise=np_, density=dn, length=ln,
            seed=make_seed(fam,np_,dn,ln,rep),
            true_model=params["families"][fam]["true_model"],
            AIC_choice=r["aic_choice"], BIC_choice=r["bic_choice"],
            alpha_est=r["alpha_est"], fit_ok=r["fit_ok"]))
    raw_rows.extend(raw_rows_ii)

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(os.path.join(OUT,"raw_results.csv"), index=False, encoding="utf-8-sig")
    log(f"raw_results.csv 行数={len(raw)}")

    # ---- 3. 聚合每格统计 ----
    def agg_cell(df):
        n = len(df)
        ok = df["fit_ok"].astype(bool)
        fail_rate = 1.0 - ok.mean()
        # 模型选择频率（仅对 fit_ok 的重复统计选择；失败的计入失败率但不计入频率）
        d = df[ok]
        if len(d)==0:
            return dict(reps=n, AIC_acc=np.nan, BIC_acc=np.nan,
                        P_M1_AIC=np.nan, P_M2_AIC=np.nan, P_M3_AIC=np.nan,
                        P_M1_BIC=np.nan, P_M2_BIC=np.nan, P_M3_BIC=np.nan,
                        alpha_est_mean=np.nan, alpha_est_std=np.nan,
                        alpha_bias_mean=np.nan, alpha_bias_std=np.nan,
                        boundary_hit_rate=np.nan, fit_fail_rate=fail_rate)
        p = d["AIC_choice"].value_counts(normalize=True)
        pm1a,pm2a,pm3a = p.get(0,0.0),p.get(1,0.0),p.get(2,0.0)
        q = d["BIC_choice"].value_counts(normalize=True)
        pm1b,pm2b,pm3b = q.get(0,0.0),q.get(1,0.0),q.get(2,0.0)
        ae = d["alpha_est"].dropna()
        aem = ae.mean(); aes = ae.std(ddof=1) if len(ae)>1 else 0.0
        # 边界命中
        bh = ((ae-0.02).abs()<=1e-3) | ((ae-3.0).abs()<=1e-3)
        bhr = bh.mean() if len(ae)>0 else np.nan
        return dict(reps=n,
                    P_M1_AIC=pm1a, P_M2_AIC=pm2a, P_M3_AIC=pm3a,
                    P_M1_BIC=pm1b, P_M2_BIC=pm2b, P_M3_BIC=pm3b,
                    alpha_est_mean=aem, alpha_est_std=aes,
                    alpha_bias_mean=np.nan, alpha_bias_std=np.nan,
                    boundary_hit_rate=bhr, fit_fail_rate=fail_rate)

    rows = []
    for key in cell_data.keys():
        fam,np_,dn,ln = key
        sub = raw[(raw.family==fam)&(raw.noise==np_)&(raw.density==dn)&(raw.length==ln)]
        a = agg_cell(sub)
        # AIC_acc = 正确率（真模型）
        tm = params["families"][fam]["true_model"]
        if tm=="M1":
            a["AIC_acc"]=a["P_M1_AIC"]; a["BIC_acc"]=a["P_M1_BIC"]
        elif tm=="M3":
            a["AIC_acc"]=a["P_M3_AIC"]; a["BIC_acc"]=a["P_M3_BIC"]
            at = params["families"][fam]["alpha_true"]
            ok = sub[sub.fit_ok.astype(bool)]
            ae = ok["alpha_est"].dropna()
            bias = (ae-at)
            a["alpha_bias_mean"]=bias.mean() if len(bias)>0 else np.nan
            a["alpha_bias_std"]=bias.std(ddof=1) if len(bias)>1 else 0.0
        else:  # accel_relax 无真模型：正确率=NaN，P(M3) 作为误判率（已在 P_M3_*）
            a["AIC_acc"]=np.nan; a["BIC_acc"]=np.nan
        row = dict(family=fam, noise=np_, density=dn, length=ln)
        row.update(a)
        rows.append(row)
    cells = pd.DataFrame(rows)
    cols = ["family","noise","density","length","reps",
            "AIC_acc","BIC_acc",
            "P_M1_AIC","P_M2_AIC","P_M3_AIC",
            "P_M1_BIC","P_M2_BIC","P_M3_BIC",
            "alpha_est_mean","alpha_est_std","alpha_bias_mean","alpha_bias_std",
            "boundary_hit_rate","fit_fail_rate"]
    cells = cells[cols]
    cells.to_csv(os.path.join(OUT,"results_all_cells.csv"), index=False, encoding="utf-8-sig")
    log(f"results_all_cells.csv 行数={len(cells)}")

    # ---- 4. 图 ----
    make_figures(curves, Afam, cells, raw, OUT)
    log("图已生成")

    # ---- 5. runtime / env ----
    total = time.time()-t_start
    rt.append(f"总用时 {total:.1f}s = {total/60:.1f} min")
    with open(os.path.join(OUT,"runtime.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(rt)+"\n")
    with open(os.path.join(OUT,"env_versions.txt"),"w",encoding="utf-8") as f:
        f.write(f"python {platform.python_version()}\n")
        f.write(f"numpy {np.__version__}\nscipy {__import__('scipy').__version__}\n")
        f.write(f"pandas {pd.__version__}\nmatplotlib {matplotlib.__version__}\n")
        f.write(f"OS {platform.platform()}\nworkers={N_WORKERS} dense_N={DENSITIES['dense']}\n")
    log(f"全部完成，总用时 {total/60:.1f} min")

# ----------------------------------------------------------------------
# 图
# ----------------------------------------------------------------------
def make_figures(curves, Afam, cells, raw, OUT):
    tfine = np.linspace(0, T, 20001)
    fams = ["near_linear","weak_conv","strong_conv","accel_relax"]
    titles = {"near_linear":"近线性 (真M1)","weak_conv":"弱凸 α=1.3 (真M3)",
              "strong_conv":"强凸 α=3.0 (真M3)","accel_relax":"加速-松弛 (无真模型)"}

    # 图1：四族 clean 曲线 + 噪声带
    fig, axes = plt.subplots(2,2, figsize=(11,7), sharex=True)
    for ax, fam in zip(axes.ravel(), fams):
        c = curves[fam]
        A = Afam[fam]
        ax.plot(tfine, c, "k-", lw=1.5, label="clean")
        for pp, col in zip([0.01,0.03,0.10], ["tab:blue","tab:orange","tab:red"]):
            s = pp*A
            ax.fill_between(tfine, c-s, c+s, color=col, alpha=0.25,
                            label=f"σ={int(pp*100)}%A")
        ax.set_title(titles[fam]); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_ylabel("strain")
    axes[1,0].set_xlabel("t / day"); axes[1,1].set_xlabel("t / day")
    fig.suptitle("图1 四族合成曲线（clean + 1%/3%/10% 噪声带）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig1_families.png"), dpi=150)
    plt.close(fig)

    # 图2：模型选择正确率（真模型），按族分面，x=噪声，线=密度×长度
    fig, axes = plt.subplots(1,2, figsize=(13,5), sharey=True)
    for ax, metric in zip(axes, ["AIC_acc","BIC_acc"]):
        for fam in ["near_linear","weak_conv","strong_conv"]:
            sub = cells[cells.family==fam]
            for dn in ["sparse","medium","dense"]:
                for ln in ["short","mid","long"]:
                    s2 = sub[(sub.density==dn)&(sub.length==ln)].sort_values("noise")
                    if len(s2)==0: continue
                    style = {"sparse":":","medium":"--","dense":"-"}[dn]
                    lw = {"short":1.0,"mid":1.6,"long":2.4}[ln]
                    ax.plot(s2.noise*100, s2[metric], style, lw=lw,
                            marker="o", ms=3,
                            label=f"{fam[:6]}/{dn[:3]}/{ln[:4]}")
        ax.set_xlabel("噪声水平 (% A)"); ax.set_title(metric.replace("_acc"," 正确率"))
        ax.grid(alpha=0.3); ax.set_xticks([1,3,10])
    axes[0].set_ylabel("模型选择正确率")
    fig.suptitle("图2 真模型选择正确率（线：密度-记录长度；族在同一图例）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig2_accuracy.png"), dpi=150)
    plt.close(fig)

    # 图3：凸族 α 估计偏差 + 边界命中率
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    for fam, at in [("weak_conv",1.3),("strong_conv",3.0),("strong2p5",2.5)]:
        sub = cells[cells.family==fam]
        for dn in ["sparse","medium","dense"]:
            for ln in ["short","mid","long"]:
                s2 = sub[(sub.density==dn)&(sub.length==ln)].sort_values("noise")
                if len(s2)==0: continue
                style = {"sparse":":","medium":"--","dense":"-"}[dn]
                lw = {"short":1.0,"mid":1.6,"long":2.4}[ln]
                axes[0].errorbar(s2.noise*100, s2.alpha_bias_mean, yerr=s2.alpha_bias_std,
                                 fmt=style, lw=lw, marker="o", ms=3,
                                 label=f"{fam}/{dn[:3]}/{ln[:4]}")
                axes[1].plot(s2.noise*100, s2.boundary_hit_rate, style, lw=lw,
                             marker="s", ms=3)
    axes[0].axhline(0,color="k",lw=0.8)
    axes[0].set_xlabel("噪声水平 (%A)"); axes[0].set_ylabel("alpha_est - alpha_true")
    axes[0].set_title("alpha 估计偏差 (mean±std)"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_xlabel("噪声水平 (%A)"); axes[1].set_ylabel("边界命中率")
    axes[1].set_title("P(|α-0.02|≤1e-3 或 |α-3.0|≤1e-3)"); axes[1].grid(alpha=0.3)
    fig.suptitle("图3 凸族 alpha 估计偏差与边界命中率")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig3_alpha_bias.png"), dpi=150)
    plt.close(fig)

    # 图4：加速-松弛族 P(M3) 误判率热图（x=噪声, y=密度, 按长度分面）
    fig, axes = plt.subplots(1,3, figsize=(13,4), sharey=True)
    sub = cells[cells.family=="accel_relax"]
    for ax, ln in zip(axes, ["short","mid","long"]):
        s2 = sub[sub.length==ln]
        pivot = s2.pivot(index="density", columns="noise", values="P_M3_AIC")
        pivot = pivot.reindex(["sparse","medium","dense"])
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{int(c*100)}%" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_title(f"记录长度={ln}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i,j]
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8)
    axes[0].set_ylabel("采样密度")
    fig.suptitle("图4 加速-松弛族 P(M3)「分数阶误判率」热图 (AIC)")
    fig.colorbar(im, ax=axes, fraction=0.025, label="P(M3)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig4_misjudge.png"), dpi=150)
    plt.close(fig)

if __name__=="__main__":
    main()
