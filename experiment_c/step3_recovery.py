# -*- coding: utf-8 -*-
"""
实验③ 参数恢复模拟：
  真实 alpha = 1.0 / 2.5 / 4.0，在各案例对应的采样时间点 t、样本量 n、
  噪声水平 noise=std(diff(eps))/sqrt(2) 下生成 M3 合成数据，跑多起点反演，
  统计 alpha 估计偏差（均值/中位数/标准差）与边界命中率。
  每设定重复 NREP=50 次，随机种子固定。
真参数设定：取各案例 M3 拟合最优解 p_fit 的 (E_H,E_1,eta_1) 不变，
  把 alpha 换成目标 a_true，并按"端点 Abel 应变幅值守恒"重定 eta_a：
      A_fit = t_end^a_fit/(eta_a_fit Gamma(a_fit+1))
      eta_a_true = t_end^a_true/(Gamma(a_true+1) A_fit)
  理由：使不同 a_true 下分数阶项端点位移量级一致，曲线形态可比。
输出: recovery_summary.csv, recovery_detail.csv, recovery_dist.png
"""
import os, time, sys
import numpy as np, pandas as pd
from scipy.special import gamma as Gamma_fn
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import common as C

OUT = os.path.dirname(os.path.abspath(__file__))
NREP = 50
TRUE_ALPHAS = [1.0, 2.5, 4.0]
# 恢复模拟用较少 alpha 起点（覆盖低/中/高），保持 basin 覆盖
REC_ALPHA_INITS = [0.5, 1.5, 2.5, 3.5]


def build_true_params(c, p_fit, a_true):
    """由 p_fit 构造 a_true 下的真参数（端点 Abel 幅值守恒）。"""
    p = p_fit.copy()
    t_end = c["t"][-1]
    A_fit = t_end ** p_fit[4] / (p_fit[3] * Gamma_fn(p_fit[4] + 1.0))
    p[4] = a_true
    p[3] = t_end ** a_true / (Gamma_fn(a_true + 1.0) * A_fit)
    return p


def one_recovery(args):
    """worker：生成一条合成数据并反演，返回 dict。"""
    case_name, a_true, rep, p_fit, seed = args
    c = C.load_case(case_name)
    t, noise = c["t"], c["noise"]
    p_true = build_true_params(c, p_fit, a_true)
    rng = np.random.default_rng(seed)
    eps_syn = C.m3(p_true, t) + rng.normal(0.0, noise, size=len(t))
    # 临时 case：换 eps，alpha 起点用恢复网格
    tmp = dict(c)
    tmp["eps"] = eps_syn
    tmp["alpha_inits"] = REC_ALPHA_INITS
    # 恢复模拟不使用真实数据 anchor（避免泄露），但保留大 eta_a 尺度起点以覆盖高 alpha 盆
    tmp.pop("anchor", None)
    sol = C.fit_m3(tmp, max_nfev=5000, xtol=1e-8, ftol=1e-8)
    a_hat = sol.x[4]
    lo_a, hi_a = c["lo"][4], c["hi"][4]
    hit_lo = (a_hat - lo_a) <= 0.005 * (hi_a - lo_a)
    hit_hi = (hi_a - a_hat) <= 0.005 * (hi_a - lo_a)
    return dict(case=case_name, a_true=a_true, rep=rep, seed=seed,
                a_hat=a_hat, eta_a_hat=sol.x[3],
                hit_lo=bool(hit_lo), hit_hi=bool(hit_hi),
                cost=sol.cost)


def main():
    t_start = time.time()
    # 先求各案例 p_fit（串行）
    pfits = {}
    for name in C.select_cases():
        c = C.load_case(name)
        sol = C.fit_m3(c, max_nfev=5000, xtol=1e-8, ftol=1e-8)
        pfits[name] = sol.x.copy()
        print(f"[p_fit] {name}: alpha={sol.x[4]:.3f} "
              f"E_H={sol.x[0]:.3g} E_1={sol.x[1]:.3g} "
              f"eta_1={sol.x[2]:.3g} eta_a={sol.x[3]:.3g}", flush=True)

    # 构造任务列表
    tasks = []
    for name in C.select_cases():
        c = C.load_case(name)
        for a_true in TRUE_ALPHAS:
            for rep in range(NREP):
                seed = 100000 + hash((name, a_true, rep)) % 1000000
                tasks.append((name, a_true, rep, pfits[name], int(seed)))
    print(f"总任务数={len(tasks)}，启动 {mp.cpu_count()} 进程...", flush=True)

    # 多进程
    nproc = min(12, mp.cpu_count())
    with mp.Pool(nproc) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(one_recovery, tasks, chunksize=4)):
            results.append(r)
            if (i + 1) % 25 == 0:
                print(f"  完成 {i+1}/{len(tasks)}  "
                      f"({time.time()-t_start:.0f}s)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "recovery_detail.csv"),
              index=False, encoding="utf-8-sig")

    # 汇总
    summ = []
    for name in C.select_cases():
        c = C.load_case(name)
        lo_a, hi_a = c["lo"][4], c["hi"][4]
        for a_true in TRUE_ALPHAS:
            sub = df[(df.case == name) & (df.a_true == a_true)]
            ah = sub["a_hat"].values
            summ.append(dict(
                case=name, n=c["n"], noise=c["noise"], a_true=a_true,
                nrep=len(sub),
                a_mean=np.mean(ah), a_median=np.median(ah),
                a_std=np.std(ah), a_bias=np.mean(ah) - a_true,
                a_min=np.min(ah), a_max=np.max(ah),
                hit_lo_frac=sub["hit_lo"].mean(),
                hit_hi_frac=sub["hit_hi"].mean(),
                wall_hit_frac=((sub["hit_lo"]) | (sub["hit_hi"])).mean(),
                alpha_lo=lo_a, alpha_hi=hi_a))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(OUT, "recovery_summary.csv"),
               index=False, encoding="utf-8-sig")
    print("\n" + sdf.to_string(index=False), flush=True)

    # 分布图
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    for ax, name in zip(axes, ["Gilman", "Achoma", "Gradenbach", "Baijiabao"]):
        for a_true in TRUE_ALPHAS:
            sub = df[(df.case == name) & (df.a_true == a_true)]
            ax.hist(sub["a_hat"], bins=15, alpha=0.5, label=f"true={a_true}")
            ax.axvline(a_true, ls="--", lw=1)
        ax.set_title(f"{name} (n={C.load_case(name)['n']})")
        ax.set_xlabel("α̂"); ax.set_ylabel("频数"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("参数恢复模拟：α̂ 分布（每设定 N=%d）" % NREP)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recovery_dist.png"), dpi=140)
    plt.close(fig)
    print(f"\n总耗时 {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
