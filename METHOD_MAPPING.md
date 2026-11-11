# Correspondence with the manuscript

| Manuscript component | Released implementation | Status |
|---|---|---|
| 2.2 Target-assisted segment registration and total-station coordinate transformation | No dedicated source selected from the supplied archive | Inputs must already be registered |
| 2.3, Eqs. 6–7: neighborhood covariance and curvature | `compute_curvature` in `reproduce_gateroad_denoising.py` | Implemented |
| 2.3, Eqs. 8–9: curvature AND distance constraint | `detect_attached_noise` | Implemented; threshold selected by a configurable quantile |
| 2.3.3 statistical and radius refinement | `detect_isolated_noise`, `radius_density_filter` | Implemented; radius counts exclude the point itself |
| 2.4.1: section centroids and inward radial directions | `build_centerline`, `nearest_centerline_fast`, `compute_gc_m3c2` | Smoothed, discretely sampled centerline; nearest centerline sample used |
| 2.4.2: cylindrical projection | `cylinder_projection` | Same support geometry applied to each epoch |
| 2.4.3, Eq. 16: median-plus-percentile blend | `compute_gc_m3c2` | Implemented using `peak_alpha` and `peak_percentile` |
| 3.3.1: independent field accuracy | `evaluate_monitoring_accuracy.py` | New reporting utility; requires matched measurements and estimates |
| 3.3.2: valid-point and longitudinal statistics | `summarize_deformation.py` | New reporting utility; signed by default, optional absolute magnitudes |
| Conventional M3C2 baseline used in the manuscript comparison | No verified standalone baseline in this code selection | Supply external baseline estimates |
| Invalid-point interpolation for visualization | Not present in selected script | Invalid CSV rows retain NaN; omitted from colored PLY |

## Differences among the archived deformation scripts

- `generate_gc_m3c2_deformation_cloudmap.py` was selected because it implements the two-epoch median-plus-percentile formula and retains signed values.
- `generate_gc_m3c2_colored_deformation_cloud.py` selects a median or percentile statistic rather than the weighted combination in Eq. 16. It also includes additional denoising logic. It is not the selected manuscript entry point.
- `run_field_gc_m3c2_section_analysis.py` requires four epochs and summarizes absolute displacements. It is not the paper's two-epoch full-field entry point.
- `generate_3d_deformation_cloudmap_analysis.py` concerns a separate four-epoch visualization workflow with hard-coded dates. It is excluded.
- Random-forest training, manual-region classification, alternative denoising experiments, figure-specific plotting scripts, output folders, environments, and cached binaries are excluded from this code-only manuscript release.

## Reproducibility details

The release preserves existing numerical routines rather than inferring a missing production configuration. The archive does not establish which exact settings generated every manuscript figure. Percentiles, blend weights, cylinder dimensions, downsampling, centerline step, and axis must be documented for a field rerun.

The original deformation script includes an optional cross-epoch ICP and a PCA fallback if the centerline fails. Its plots/statistics use absolute magnitudes even when `--no-absolute` is supplied; the script emits a warning. These behaviors are retained and identified rather than silently changing the archived algorithm. Use signed output columns and the added summary utility for the manuscript's sign convention.

The reported nearest-neighbor RMSE around optional ICP is a cloud-to-cloud discrepancy, not independently measured survey registration accuracy. Plot legend severity labels are not validated engineering thresholds.

The manuscript's 940 m values show a 0.1 mm difference between the displayed subtraction and Table 1. No field values or corrections are embedded in this release. The accuracy utility computes errors from the input measurements provided by the user.
