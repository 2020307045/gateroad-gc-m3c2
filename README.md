# Gateroad surface extraction and GC-M3C2 deformation analysis

Python code for curvature-constrained point-cloud denoising and geometrically constrained multiscale model-to-model cloud comparison (GC-M3C2) in underground gateroads.

This code accompanies the manuscript **Gateroad Deformation Monitoring Based on 3D Laser Scanning Technology Under Complex Mining Environment: A Case Study**. The repository contains source code and documentation only. Field point clouds, monitoring measurements, figures, generated results, and Python environments are excluded.

[中文说明](README_zh.md) · [Method correspondence](docs/METHOD_MAPPING.md) · [GitHub upload instructions](docs/GITHUB_UPLOAD.md)

## Repository contents

| Script | Purpose |
|---|---|
| `scripts/reproduce_gateroad_denoising.py` | Curvature and statistical-distance constraints, followed by statistical and radius-density filtering |
| `scripts/generate_gc_m3c2_deformation_cloudmap.py` | Two-epoch GC-M3C2 estimates, validity flags, colored point clouds, and visualizations |
| `scripts/split_gateroad_sections.py` | Axis- or centerline-based section extraction |
| `scripts/evaluate_monitoring_accuracy.py` | Matched measurement errors and method-level accuracy summaries |
| `scripts/summarize_deformation.py` | Valid-point statistics and longitudinal interval summaries |
| `tests/test_workflow.py` | Numerical checks using small arrays generated in memory |

The first three scripts were selected from the supplied development archive. Their numerical routines are retained. The last two scripts provide additional reporting utilities for the manuscript's accuracy and spatial analyses. Source hashes and edit provenance are recorded in `docs/source_manifest.json`.

## Installation

Use Python 3.10–3.12. Local verification used Python 3.12.4 on Windows; other versions are not yet verified.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt, `./.venv/Scripts/Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then install:

```bash
python -m pip install -r requirements.txt
```

`requirements-tested.txt` lists the direct dependency versions used during local verification. For optional LAS/LAZ input to the deformation script, install `laspy[lazrs]` separately. LAS/LAZ support is not included in the local verification.

## Input preparation

1. Independently place both monitoring epochs in the same surveyed coordinate system.
2. Use metres for XYZ coordinates and geometric parameters.
3. Supply an explicit roadway axis when the longest PCA axis is not the longitudinal direction.
4. Use field measurements from matching locations, intervals, and displacement directions for accuracy evaluation.

Control-point surveying, target-assisted segment registration, and total-station-based coordinate transformation are described in the manuscript but are not implemented by the selected source files. Registration is an input prerequisite. The deformation script's optional `--do_icp` is an experimental cross-epoch alignment option; leave it off for the manuscript workflow because deforming roadway surfaces are not stable alignment references.

## Run the workflow

Run all commands from the repository root. Input paths below refer to files supplied by the user. The parameter values illustrate usage; they are not a recovered field-experiment configuration.

### 1. Extract the roadway surface

```bash
python scripts/reproduce_gateroad_denoising.py --input data/epoch1_registered.ply --out_dir results/epoch1_surface --kp 30 --curvature_quantile 0.93 --attached_alpha 3.0 --stat_k 30 --stat_alpha 2.5 --radius 0.15 --min_neighbors 8
python scripts/reproduce_gateroad_denoising.py --input data/epoch2_registered.ply --out_dir results/epoch2_surface --kp 30 --curvature_quantile 0.93 --attached_alpha 3.0 --stat_k 30 --stat_alpha 2.5 --radius 0.15 --min_neighbors 8
```

Main outputs include `surface_denoised_used.ply`, classified noise clouds, `denoise_summary.csv`, and `denoise_report.txt`. Labels are 0 for retained surface, 1 for attached structured noise, and 2 for isolated outliers. Radius-neighbor counts exclude the query point itself. `--voxel_size` and `--map_to_original` enable optional downsampling and label transfer.

### 2. Compute two-epoch deformation

```bash
python scripts/generate_gc_m3c2_deformation_cloudmap.py --baseline results/epoch1_surface/surface_denoised_used.ply --target results/epoch2_surface/surface_denoised_used.ply --out_dir results/deformation --axis y --section_step 1.0 --section_width 0.5 --projection_radius 0.20 --half_length 0.80 --min_neighbors 8 --peak_percentile 95 --peak_alpha 0.40 --voxel_size 0 --dpi 150
```

Set cylinder dimensions according to point spacing and expected displacement. In particular, the original half-length default of 0.30 m is shorter than the approximately 0.48–0.59 m maxima discussed in the manuscript. The example uses 0.80 m to illustrate a larger search support, not to prescribe a field-validated value. Inspect neighborhood counts and validity before interpreting the estimates.

The algorithm blends the difference in medians and the difference in the selected percentiles along inward radial directions:

```text
displacement = (1 - alpha) * (median(target) - median(baseline))
             + alpha * (percentile(target, p) - percentile(baseline, p))
```

`gc_m3c2_deformation_values.csv` contains all processed baseline/core points:

| Column | Meaning |
|---|---|
| `x`, `y`, `z` | Coordinates in metres |
| `signed_gc_m3c2_distance` | Signed displacement in metres; inward convergence is positive |
| `abs_gc_m3c2_distance` | Absolute magnitude in metres |
| `baseline_neighbor_count`, `target_neighbor_count` | Cylindrical neighborhood populations |
| `valid_flag` | 1 for a valid estimate; 0 for an invalid estimate |
| `mileage_m` | Axial distance from the processed baseline minimum, not surveyed chainage or curved-centerline arc length |

The retained visualization script colors **absolute magnitudes** and reports magnitude statistics. Its `--no-absolute` option does not switch the plot to signed values. Signed values remain in CSV/PLY; use the following summary utility for signed analysis. Color-range adjectives in the original legend are display labels, not validated stability thresholds. Invalid points remain NaN in the CSV and are omitted from the colored PLY; this script does not implement the visualization interpolation described in the manuscript.

If centerline construction fails, the original script warns and falls back to PCA normals. Such a run should be diagnosed before being reported as centerline-constrained GC-M3C2. See the method mapping for implementation details.

### 3. Summarize spatial deformation

```bash
python scripts/summarize_deformation.py --input results/deformation/gc_m3c2_deformation_values.csv --out_dir results/statistics --interval 10 --chainage_offset 850
```

The default summarizes **signed** convergence using valid points only. Add `--absolute` for magnitude statistics. Output displacements are in millimetres. Empty interior intervals remain present with count 0 and blank statistics. Intervals are left-closed and right-open, except an exact final boundary belongs to the preceding final interval. The full-field mean is computed directly over valid points, so it agrees with interval means weighted by their valid counts. `--chainage_offset` labels the axial origin; it does not perform surveying or registration.

### 4. Evaluate monitoring accuracy

Prepare a CSV with these columns, one row per method and monitoring location:

```text
chainage_m,location,method,measured_mm,estimated_mm
```

Use location names such as `roof`, `floor`, `left_rib`, and `right_rib`. Use one monitoring interval per input file. Methods must have the same location set and measured reference values.

```bash
python scripts/evaluate_monitoring_accuracy.py --input data/matched_monitoring.csv --out_dir results/accuracy
```

Outputs include location-level signed/absolute/relative errors and method-level MAE, RMSE, mean bias, and maximum absolute error. Relative errors are blank for zero reference displacement. The script does not compute conventional M3C2 or infer the field measurement correspondence; supply externally obtained, matched estimates for each method. Cross-sectional maxima must be selected over the same monitored location set, not over unrelated point-cloud maxima.

### 5. Optional section extraction

```bash
python scripts/split_gateroad_sections.py --input results/epoch1_surface/surface_denoised_used.ply --out_dir results/sections --slice_mode axis --axis y --total_length 100 --section_length 10 --min_points 100
```

These are contiguous slabs. For curved roadways, choose `--slice_mode centerline` and provide `--centerline_csv` or configure `--centerline_axis` and `--centerline_bin_width`. Centerline stationing in this utility differs from the deformation script's axial `mileage_m`.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests construct small synthetic arrays at runtime; no dataset is shipped. They check displacement sign, identical epochs, missing neighborhoods, denoising constraints, monitoring correspondence, interval boundaries, and valid-point weighting. See `docs/VALIDATION.md` for the completed local checks. Field results cannot be reproduced from source code alone; the aligned field inputs and original parameter settings are not part of this release.

## Citation and license

The manuscript title is recorded above for identification. Add the final publication details and DOI when available. No DOI, publication status, repository URL, or software license has been assumed. A reuse license has not yet been selected; add the chosen license before describing the repository as open-source software.
