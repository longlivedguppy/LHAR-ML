# Huber 1D MRF preliminary comparison

This experiment compares a configurable parallel-line mean profile from one
LHAR TIFF image. It is
kept separate from the production pipeline and does **not** identify any peak as
a PD boundary.

## Scope

- Raw mean of 1, 5, 10, or 20 parallel lines
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
python compare_mrf_profile.py --plot-method mrf --plot-regularization 10 --line-count 10
python compare_mrf_profile.py --comparison-plot
```

Output is separated by the selected denoiser and its non-default key parameters:

```text
mrf_comparison/<dataset>/<image>/x<coordinate>/<denoise_tag>/line_<count>/
```

Examples include `median`, `median_k3`, `gaussian_sigma_1`,
`median_gaussian_k3_sigma_1`, and `mrf_lambda_10`. Different denoisers or
parameters therefore never overwrite one another; rerunning the same setting
continues to replace the same files.

Line counts are selected with any positive integer such as `--line-count 1`,
`5`, `10`, `20`, or `100`. Odd counts
use integer x offsets centered on the nominal line. Even counts use symmetric
half-pixel offsets with linear interpolation in x. All Raw lines are averaged
first; the selected denoiser, derivative, and peak assessment then run once on
the averaged Raw profile.

The default line spacing is one pixel, so 20 lines span only 19 pixels. Use
`--line-spacing 2` or `4` to distinguish the effect of sampling width from the
effect of sample count. Non-default spacing is included in the output tag, for
example `line_20_spacing_4`.

By default, peak prominence retains the existing adaptive MAD threshold. For a
controlled line-averaging comparison, use a fixed value such as
`--peak-prominence 100` for every line count. Fixed-threshold results are stored
below a separate directory such as `peak_prominence_100` and do not overwrite
adaptive results.

`comparison_overview.png` shows only Raw, the selected denoising result, its
absolute first derivative, and the main candidate marker. The default selected
result is Huber MRF with lambda 1. `--plot-method` accepts `median`, `gaussian`,
`median_gaussian`, or `mrf`. The former five-panel view is generated separately
as `comparison_overview_multi_method.png` only with `--comparison-plot`.

The two y-axes are dynamically scaled for visual separation. The assessed
absolute derivative occupies roughly the lower 35% and Raw/Denoise intensity
roughly the upper 45% by default. The derivative scale uses the assessment
region's robust 99th percentile plus the selected peak, so a large derivative
inside the excluded region does not flatten the useful signal. These display
fractions can be adjusted without changing analysis results:

```powershell
python compare_mrf_profile.py --derivative-height-fraction 0.4 --intensity-height-fraction 0.5
```

`metrics.csv` distinguishes the
numerical maximum from a significant candidate; `significant_peaks.csv` records
prominence, width, and whether each filtered candidate has a nearby Raw local
maximum. Absence of a candidate is retained as `undetectable`.

## All-image batch comparison

To process every TIFF in a dataset while retaining the same single-line scope:

```powershell
python compare_mrf_batch.py --dataset 20260512
python compare_mrf_batch.py --dataset 20260512 --plot-method gaussian
python compare_mrf_batch.py --dataset 20260512 --plot-method mrf --plot-regularization 10 --regularization 10 --line-count 20
python compare_mrf_batch.py --dataset 20260512 --plot-method mrf --plot-regularization 10 --regularization 10 --line-count 20 --line-spacing 4 --peak-prominence 100
python compare_mrf_batch.py --dataset 20260512 --plot-method mrf --plot-regularization 10 --regularization 10 --line-count 100 --peak-prominence 100
python compare_mrf_batch.py --dataset 20260512 --comparison-plots
```

Batch output is isolated below
`mrf_comparison_batch/<dataset>/x<coordinate>/<denoise_tag>/line_<count>/`.
It includes per-angle profile CSVs and single-method figures, aggregate metrics,
and a compressed NumPy profile archive. Legacy five-panel per-angle figures are
written to the separate `comparison_plots` directory only when
`--comparison-plots` is supplied. The same flag enables the multi-method
all-angle heatmaps and summary plots. This remains a comparison experiment and
does not connect to the
production all-angle correction or PD classification.
