# Huber 1D MRF preliminary comparison

This experiment compares one raw vertical line from one LHAR TIFF image. It is
kept separate from the production pipeline and does **not** identify any peak as
a PD boundary.

## Scope

- Raw single line (no 21-line median reduction)
- Existing-style Median (kernel 5), then Gaussian (sigma 5)
- Huber 1D MRF at weak, medium, and strong regularization
- A readable two-axis Raw/Denoise/absolute-derivative plot by default
- Optional profile, derivative, candidate-zoom, and residual comparison plot
- CSV metrics and candidate details

It does not run the 61-angle analysis, anchor correction, Pure/Recovered/Missing
classification, interpolation, or existing PD detection. It never uses `argmax`
as a fallback to manufacture a boundary candidate. A numerical absolute maximum
is reported as a descriptive metric only and is separate from thresholded
candidate detection.

## Run

From the repository root:

```powershell
python compare_mrf_profile.py
```

The default is the 0-degree image in `data/raw/260115-1-20um50cyc`, at `x=878`
and `y=1400..200`. Examples of explicit options:

```powershell
python compare_mrf_profile.py --image "data/raw/260115-1-20um50cyc/img114142_  0_  0.tiff" --x 878
python compare_mrf_profile.py --plot-method median_gaussian
python compare_mrf_profile.py --plot-method mrf --plot-regularization 10
python compare_mrf_profile.py --comparison-plot
```

Output is written only below:

```text
mrf_comparison/<dataset>/<image>/x<coordinate>/
```

`comparison_overview.png` shows only Raw, the selected denoising result, its
absolute first derivative, and the main candidate marker. The default selected
result is Huber MRF with lambda 1. `--plot-method` accepts `median`, `gaussian`,
`median_gaussian`, or `mrf`. The former five-panel view is generated separately
as `comparison_overview_multi_method.png` only with `--comparison-plot`.

`metrics.csv` distinguishes the
numerical maximum from a significant candidate; `significant_peaks.csv` records
prominence, width, and whether each filtered candidate has a nearby Raw local
maximum. Absence of a candidate is retained as `undetectable`.

## All-image batch comparison

To process every TIFF in a dataset while retaining the same single-line scope:

```powershell
python compare_mrf_batch.py --dataset 20260512
python compare_mrf_batch.py --dataset 20260512 --plot-method gaussian
python compare_mrf_batch.py --dataset 20260512 --comparison-plots
```

Batch output is isolated below `mrf_comparison_batch/<dataset>/x<coordinate>/`.
It includes per-angle profile CSVs and single-method figures, aggregate metrics,
and a compressed NumPy profile archive. Legacy five-panel per-angle figures are
written to the separate `comparison_plots` directory only when
`--comparison-plots` is supplied. The same flag enables the multi-method
all-angle heatmaps and summary plots. This remains a comparison experiment and
does not connect to the
production all-angle correction or PD classification.
