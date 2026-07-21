# Huber 1D MRF preliminary comparison

This experiment compares one raw vertical line from one LHAR TIFF image. It is
kept separate from the production pipeline and does **not** identify any peak as
a PD boundary.

## Scope

- Raw single line (no 21-line median reduction)
- Existing-style Median (kernel 5), then Gaussian (sigma 5)
- Huber 1D MRF at weak, medium, and strong regularization
- Profile, derivative, absolute derivative, candidate zoom, and residual plots
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
python compare_mrf_profile.py --regularization 0.1 1 10
```

Output is written only below:

```text
mrf_comparison/<dataset>/<image>/x<coordinate>/
```

The overview PNG is intended for visual review. `metrics.csv` distinguishes the
numerical maximum from a significant candidate; `significant_peaks.csv` records
prominence, width, and whether each filtered candidate has a nearby Raw local
maximum. Absence of a candidate is retained as `undetectable`.

## All-image batch comparison

To process every TIFF in a dataset while retaining the same single-line scope:

```powershell
python compare_mrf_batch.py --dataset 20260512
```

Batch output is isolated below `mrf_comparison_batch/<dataset>/x<coordinate>/`.
It includes per-angle profile CSVs and comparison figures, aggregate metrics, a
compressed NumPy profile archive, all-angle profile/derivative heatmaps, and
summary plots. This remains a comparison experiment and does not connect to the
production all-angle correction or PD classification.
