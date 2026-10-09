# Fractional-order Burgers model for landslide creep: applicability boundary and early-warning implications

**Code for:** "Fractional-order Burgers model for landslide creep and failure warning: applicability boundary and its evidence" (Wang & Jin, manuscript in preparation; targeted at *Natural Hazards and Earth System Sciences*).

This repository contains the three controlled numerical experiments that establish **when** the fractional-order Burgers model is (and is not) trustworthy for landslide creep analysis and failure-time estimation, using five observational records:

- **Gilman Creek** (Colorado, USA) — near-linear creep, `n = 62`
- **Achoma** (Colca valley, Peru) — accelerating creep to failure on 2020-06-18, `n = 45`
- **Gradenbach** (Austria) — long accelerating-creep record with episodic accelerations, `n = 6746`
- **Baijiabao** and **Shuping** (Three Gorges Reservoir, China) — creep modulated by reservoir-level seasonality

The experiments address the three questions reviewers most often raise for this type of study:

| # | Experiment | Question answered |
|---|---|---|
| A | Synthetic-data applicability boundary | Does the advantage of the fractional model grow with creep nonlinearity, or is it an artifact of sampling/noise/length? |
| B | Achoma rolling-origin forecasts | Is the near-perfect retrospective failure-time estimate a real warning capability, or an artifact of selecting the window after the fact? |
| C | Parameter identifiability | Are the fitted fractional order and viscosity distinguishable from simpler alternatives, and are they stable against parameter bounds? |

## Repository layout

```
fractional-burgers-landslide/
├── experiment_a/        # synthetic-data boundary experiment (Exp. A)
│   └── code/
│       ├── run_experiment.py        # main: 135-cell full-factorial grid, 10k+ fits
│       ├── postprocess.py           # aggregation → results_all_cells.csv + fig1-4
│       ├── params.json              # grid configuration
│       └── model_selection_source.py# read-only reference of the M1/M2/M3 framework
├── experiment_b/        # Achoma rolling-origin failure-time forecast (Exp. B)
│   ├── experiment_b_rolling.py      # main: rolling forecasts, Fukuzono vs inverse-velocity
│   └── make_plots.py                # error-vs-lead, representative fits
├── experiment_c/        # parameter identifiability (Exp. C)
│   ├── common.py                    # M3 inversion, profile likelihood, AIC/BIC, correlation
│   ├── step0_reproduce_bench.py     # reproduce four-case optimum + timing benchmark
│   ├── step1_profile.py             # profile likelihood + 95% CI per case
│   ├── step1b_fine_gradenbach.py    # fine profile near the Gradenbach basin
│   ├── step2_achoma_upper.py        # alpha upper-bound relaxation 5→8→10
│   └── step3_recovery.py            # parameter-recovery simulation (NREP=50)
├── preprocessing/       # (optional) rebuild clean time series from raw data
│   ├── gilman_preprocess.py         # USGS InSAR → t, strain
│   └── achoma_preprocess.py         # PlanetScope offset-tracking GeoTIFF → t, strain
├── data/                # see data/README.md (data NOT bundled here)
├── requirements.txt
└── LICENSE              # MIT (code only)
```

## Requirements

Python 3.10+ with `numpy`, `scipy`, `pandas`, `matplotlib`:

```bash
pip install -r requirements.txt
```

Optional (only for rebuilding raw data): `pyshp` (Gilman), `rasterio` (Achoma).

## Data

Three of the four time series **are already included** under `data/` (Gilman — USGS public domain; Achoma — Zenodo CC BY 4.0; Gradenbach — Zenodo CC BY 4.0), so **Experiments B and C run out of the box for these cases**. Only the **Baijiabao** series is excluded because it is covered by an NCDC data-use agreement that prohibits redistribution — request it via the NCDC portal (DOI 10.12072/ncdc.sanxia.db6815.2025) and place it at `data/Baijiabao/timeseries_clean.csv`.

- Per-dataset sources, DOIs, licenses and file layout: **[data/README.md](data/README.md)**.
- Until the Baijiabao file is available, run the Experiment C scripts with `--skip Baijiabao`:

```bash
cd experiment_c
python step0_reproduce_bench.py --skip Baijiabao
python step1_profile.py         --skip Baijiabao
python step3_recovery.py        --skip Baijiabao
```

To point the scripts at a custom data location:

```bash
export CREEP_DATA_ROOT=/path/to/your/data      # Linux/macOS
$env:CREEP_DATA_ROOT = "D:\your\data"          # Windows PowerShell
```

## Reproducing the experiments

### Experiment A — synthetic-data boundary (~2 h, 10 parallel workers)

```bash
cd experiment_a/code
python run_experiment.py     # 135 cells × 30–100 reps ≈ 10,350 fits → out/raw_results.csv
python postprocess.py        # → out/results_all_cells.csv + fig1_families/fig2_accuracy/fig3_alpha_bias/fig4_misjudge.png
```

Key result: the fractional model's advantage grows with creep nonlinearity; on truly linear data the 5-parameter M3 is selected only 4.0 % (AIC) / 0.3 % (BIC) of the time, and the acceleration–relaxation family shows a corner where M3 wins while `alpha` pins to the upper bound — the signal of external-driver misjudgment.

### Experiment B — Achoma rolling forecasts (seconds)

```bash
cd experiment_b
python experiment_b_rolling.py    # → results_table.csv, summary_by_cut.csv
python make_plots.py              # → fig1/fig2/fig3
```

Key result: with strictly causal truncation only the two latest cut points (88/89 d, i.e. 3–4 d lead) give usable estimates (bias −3.5/−2.5 d); the retrospective full-data fit (t_f ≈ 91.2 d vs true 92.0 d) is reported as retrospective, not as operational warning validation.

### Experiment C — identifiability (minutes)

```bash
cd experiment_c
python step0_reproduce_bench.py       # reproduce_bench.csv
python step1_profile.py               # profile_ci_summary.csv, corr_matrix.csv, boundary_hits.csv, fig
python step1b_fine_gradenbach.py      # fine profile near basin
python step2_achoma_upper.py          # achoma_upper_relax.csv (alpha bound 5→8→10)
python step3_recovery.py              # recovery_summary.csv, recovery_detail.csv, fig
```

Key results: `alpha = 1` is **not** statistically excluded for Gilman (profile-likelihood CI [0.40, 1.50]); Achoma's `alpha ≈ 4.46` is far above 1 and stable when the upper bound is raised 5→8→10 (interior solution); strong `corr(alpha, eta_a)` ridges (up to +1.000) expose weak joint identifiability in long records.

## Citation

```bibtex
@unpublished{wang2026fractional,
  title  = {Fractional-order Burgers model for landslide creep and failure warning:
            applicability boundary and its evidence},
  author = {Wang, Fuchang and Jin, Zhitong},
  year   = {2026},
  note   = {Manuscript in preparation; School of Science,
            Emergency Management University, Sanhe, Hebei, China}
}
```

## License

- **Code**: MIT (see [LICENSE](LICENSE)).
- **Data**: third-party licenses as described in [data/README.md](data/README.md) — do not redistribute the NCDC datasets (Baijiabao/Shuping) without permission.

## Funding

Hebei Provincial Natural Science Foundation of China (grant D2025512035).
