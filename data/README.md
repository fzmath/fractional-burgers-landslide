# Data

**Status: 3 of the 4 time series used by the experiments are already included in
this repository** (see below); only the NCDC-restricted Baijiabao series must be
requested separately.

All experiment scripts expect the same simple two-column format:

| column | meaning | unit |
|---|---|---|
| `t` | time since first observation | days |
| `strain` | normalized displacement (displacement / reference length) | dimensionless |

By default the scripts look for the files under this `data/` directory
(`<repo>/data`). To point them elsewhere, set the environment variable
`CREEP_DATA_ROOT` to the directory that contains the four case folders, e.g.
on Windows:

```powershell
$env:CREEP_DATA_ROOT = "D:\my_data"
```

## Case datasets

### 1. Gilman Creek landslide, Colorado, USA — data/Gilman/timeseries_clean.csv (INCLUDED)

- **Source**: USGS InSAR time series (timeseries_ERA5_demErr_Tintmasked). US Government work, public domain; redistribution permitted.
- **Reference / download**: USGS EROS InSAR time-series product; request from the USGS.
- **Reconstruction**: `preprocessing/gilman_preprocess.py` (set `GILMAN_BASE` to the directory containing `unzipped/timeseries_ERA5_demErr_Tintmasked.shp`). Requires `pyshp` (`pip install pyshp`).
- 62 observations; near-linear creep.

### 2. Achoma landslide, Colca valley, Peru — data/Achoma/timeseries_clean.csv (INCLUDED)

- **Source**: Lacroix et al. (2023), PlanetScope offset-tracking displacement; **CC BY 4.0** (redistribution permitted with attribution).
- **DOI**: https://doi.org/10.5281/zenodo.7866962  (accompanying paper: Lacroix et al., GRL 2023, doi:10.1029/2022GL101961)
- **Reconstruction**: `preprocessing/achoma_preprocess.py` (set `ACHOMA_RAW` to the folder with the NS/EW GeoTIFF stack, `ACHOMA_PROC` for outputs). Requires `rasterio` (`pip install rasterio`).
- 45 observations over the pre-failure acceleration segment; failed on 2020-06-18, true failure time `t_f = 92.0` d in this time base.

### 3. Gradenbach landslide, Austria — data/Gradenbach/timeseries_clean.csv (INCLUDED)

- **Source**: borehole extensometer record; **CC BY 4.0** (redistribution permitted with attribution).
- **DOI**: https://doi.org/10.5281/zenodo.21033260
- 6746 daily observations; long accelerating-creep record with episodic accelerations (the "T2b" pulse experiment).

### 4. Baijiabao landslide, Three Gorges Reservoir, China — NOT INCLUDED (apply for access)

- **Source**: National Cryosphere Desert Data Center (NCDC), obtained under a signed **data-use agreement**; **redistribution is not permitted**, so the processed file is not published here either.
- **DOI**: https://doi.org/10.12072/ncdc.sanxia.db6815.2025  (metadata: https://www.ncdc.ac.cn/portal/metadata/3768727b-13b2-4675-8a00-2d661ec96229)
- Request access through the NCDC portal, place the file at `data/Baijiabao/timeseries_clean.csv`, then run the Experiment C scripts **without** `--skip Baijiabao`. Until then, use `python step0_reproduce_bench.py --skip Baijiabao` (same for step1/step3) to reproduce the other three cases.

## Notes

- The reservoir-level time series used for the hydrological-forcing analysis of
  Baijiabao is also NCDC-restricted and therefore not included.
- Experiment A needs no data at all: it is fully synthetic (see
  `experiment_a/run_experiment.py` for the generative model and the
  deterministic seed function).
