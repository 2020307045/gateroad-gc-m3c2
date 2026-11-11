#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Reproduce a gateroad point-cloud denoising workflow described as:

    curvature-constrained identification + statistical-density filtering

The method is designed for mobile laser scanning gateroad point clouds and
separates:
    0 = retained gateroad surface point
    1 = attached structured noise
    2 = isolated outlier

If scipy is missing, install dependencies with:
    pip install numpy open3d scipy
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Please install Open3D with: pip install open3d")
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("This reproduction uses scipy.spatial.cKDTree for efficient neighbor search.")
    print("Please install scipy with: pip install scipy")
    sys.exit(1)


LABEL_SURFACE = 0
LABEL_ATTACHED = 1
LABEL_ISOLATED = 2

COLOR_SURFACE = np.array([0.55, 0.55, 0.55], dtype=np.float64)
COLOR_ATTACHED = np.array([1.00, 0.00, 0.00], dtype=np.float64)
COLOR_ISOLATED = np.array([0.00, 0.25, 1.00], dtype=np.float64)

DEFAULT_CHUNK_SIZE = 50000


def str2bool(value):
    """Parse flexible boolean command line values."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def timed_message(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def normalize_colors(colors, point_count):
    """Normalize optional RGB colors to Open3D's 0-1 range."""
    if colors is None:
        return None
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim != 2 or colors.shape != (point_count, 3):
        return None
    if colors.size > 0 and np.nanmax(colors) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def load_text_point_cloud(input_path):
    """Load .xyz or .txt point cloud files with at least x/y/z columns."""
    input_path = Path(input_path)

    data = None
    for delimiter in (None, ","):
        try:
            candidate = np.genfromtxt(input_path, delimiter=delimiter, comments="#")
            if candidate.ndim == 1:
                candidate = candidate.reshape((1, -1))
            if candidate.size > 0 and candidate.shape[1] >= 3 and np.isfinite(candidate[:, :3]).all():
                data = candidate
                break
        except Exception:
            continue

    if data is None:
        # Try files with a header row such as x y z r g b.
        for delimiter in (None, ","):
            try:
                named = np.genfromtxt(
                    input_path,
                    delimiter=delimiter,
                    comments="#",
                    names=True,
                    dtype=float,
                    encoding="utf-8",
                )
                names = [name.lower() for name in named.dtype.names or []]
                if {"x", "y", "z"}.issubset(set(names)):
                    points = np.column_stack([named["x"], named["y"], named["z"]])
                    colors = None
                    if {"r", "g", "b"}.issubset(set(names)):
                        colors = np.column_stack([named["r"], named["g"], named["b"]])
                    elif {"red", "green", "blue"}.issubset(set(names)):
                        colors = np.column_stack([named["red"], named["green"], named["blue"]])
                    return points.astype(np.float64), normalize_colors(colors, len(points))
            except Exception:
                continue

    if data is None:
        raise ValueError(f"Failed to read text point cloud: {input_path}")

    points = data[:, :3].astype(np.float64)
    colors = None
    if data.shape[1] >= 6:
        colors = normalize_colors(data[:, 3:6], len(points))
    return points, colors


def load_point_cloud(input_path):
    """
    Step 1: read input point cloud and keep optional colors.

    Supported formats: .ply, .pcd, .xyz, .txt.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    ext = input_path.suffix.lower()
    if ext in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(input_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
        colors = normalize_colors(colors, len(points))
    elif ext in {".xyz", ".txt"}:
        points, colors = load_text_point_cloud(input_path)
    else:
        raise ValueError(f"Unsupported input format: {ext}. Use .ply, .pcd, .xyz, or .txt")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input point cloud must contain Nx3 coordinates.")
    if len(points) == 0:
        raise ValueError("Input point cloud is empty.")
    if not np.isfinite(points).all():
        raise ValueError("Input point cloud contains NaN or infinite coordinates.")

    return points, colors


def make_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = normalize_colors(colors, len(points))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_point_cloud(path, points, colors=None):
    """Save a PLY point cloud, including valid empty PLY files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    colors = normalize_colors(colors, len(points))

    if len(points) == 0:
        with path.open("w", encoding="ascii") as file_obj:
            file_obj.write("ply\n")
            file_obj.write("format ascii 1.0\n")
            file_obj.write("element vertex 0\n")
            file_obj.write("property float x\n")
            file_obj.write("property float y\n")
            file_obj.write("property float z\n")
            if colors is not None:
                file_obj.write("property uchar red\n")
                file_obj.write("property uchar green\n")
                file_obj.write("property uchar blue\n")
            file_obj.write("end_header\n")
        return

    ok = o3d.io.write_point_cloud(str(path), make_point_cloud(points, colors), write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud: {path}")


def voxel_downsample_if_needed(points, colors, voxel_size):
    """
    Step 2: optional voxel downsampling.

    Downsampling is only used for computation. If --map_to_original is enabled,
    labels are later mapped back to the original full-resolution cloud.
    """
    if voxel_size <= 0:
        return points.copy(), None if colors is None else colors.copy()

    pcd = make_point_cloud(points, colors)
    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    used_points = np.asarray(down.points, dtype=np.float64)
    used_colors = np.asarray(down.colors, dtype=np.float64) if down.has_colors() else None
    used_colors = normalize_colors(used_colors, len(used_points))

    if len(used_points) == 0:
        raise ValueError("Voxel downsampling produced an empty point cloud.")
    return used_points, used_colors


def adjusted_neighbor_count(requested_k, point_count, name):
    """Use min(k, n_points - 1), with warning when adjustment is needed."""
    if point_count < 2:
        return 0
    adjusted = min(int(requested_k), point_count - 1)
    if adjusted < requested_k:
        warnings.warn(f"{name} is larger than point count; using {adjusted}.")
    if adjusted < 1:
        warnings.warn(f"{name} is too small after adjustment.")
    return adjusted


def kd_query(tree, query_points, k):
    """cKDTree.query wrapper compatible with older scipy versions."""
    try:
        return tree.query(query_points, k=k, workers=-1)
    except TypeError:
        return tree.query(query_points, k=k)


def compute_curvature(points, kp, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Step 3: compute local PCA curvature.

    kappa_i = lambda_min / (lambda1 + lambda2 + lambda3 + eps)

    The smallest eigenvalue is deliberately used as the numerator.
    """
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    curvature = np.zeros(point_count, dtype=np.float64)
    k_eff = adjusted_neighbor_count(kp, point_count, "kp")

    if point_count < 3 or k_eff < 3:
        warnings.warn("Too few points/neighbors for reliable curvature; curvature is set to 0.")
        return curvature

    tree = cKDTree(points)
    query_k = k_eff + 1

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        _, indices = kd_query(tree, points[start:end], k=query_k)
        if indices.ndim == 1:
            indices = indices.reshape((-1, query_k))
        neighbor_indices = indices[:, 1:]
        neighborhoods = points[neighbor_indices]
        centers = np.mean(neighborhoods, axis=1, keepdims=True)
        centered = neighborhoods - centers
        covariances = np.einsum("nki,nkj->nij", centered, centered) / float(k_eff)
        eigvals = np.linalg.eigvalsh(covariances)
        eigvals = np.maximum(eigvals, 0.0)
        denom = np.sum(eigvals, axis=1) + 1e-12
        curvature[start:end] = eigvals[:, 0] / denom

        timed_message(f"Curvature progress: {end}/{point_count}")

    return curvature


def compute_average_neighbor_distance(points, k, chunk_size=DEFAULT_CHUNK_SIZE):
    """Compute mean distance to k nearest neighbors, excluding the point itself."""
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    avg_dist = np.zeros(point_count, dtype=np.float64)
    k_eff = adjusted_neighbor_count(k, point_count, "stat_k")

    if point_count < 2 or k_eff < 1:
        warnings.warn("Too few points for average neighbor distance; distances are set to 0.")
        return avg_dist

    tree = cKDTree(points)
    query_k = k_eff + 1

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        distances, _ = kd_query(tree, points[start:end], k=query_k)
        if distances.ndim == 1:
            distances = distances.reshape((-1, query_k))
        neighbor_distances = distances[:, 1:]
        neighbor_distances[~np.isfinite(neighbor_distances)] = 0.0
        avg_dist[start:end] = np.mean(neighbor_distances, axis=1)

    return avg_dist


def detect_attached_noise(curvature, avg_dist, curvature_quantile, attached_alpha):
    """
    Steps 4-5: high curvature candidates constrained by statistical distance.

    Curvature alone is not used for deletion. A point must satisfy both:
        curvature > quantile(curvature, q)
        avg_dist > mean(avg_dist) + alpha * std(avg_dist)
    """
    curvature_threshold = float(np.quantile(curvature, curvature_quantile))
    attached_threshold = float(np.mean(avg_dist) + attached_alpha * np.std(avg_dist))
    attached_mask = (curvature > curvature_threshold) & (avg_dist > attached_threshold)
    return attached_mask, curvature_threshold, attached_threshold


def detect_isolated_noise(points, stat_k, stat_alpha, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Step 6: statistical isolated outliers after removing attached noise.

    Returns a mask for the provided points and the isolated distance threshold.
    """
    if len(points) < 2:
        warnings.warn("Too few remaining points for statistical isolated filtering.")
        return np.zeros(len(points), dtype=bool), 0.0, np.zeros(len(points), dtype=np.float64)

    avg_dist = compute_average_neighbor_distance(points, stat_k, chunk_size=chunk_size)
    threshold = float(np.mean(avg_dist) + stat_alpha * np.std(avg_dist))
    return avg_dist > threshold, threshold, avg_dist


def radius_density_filter(points, radius, min_neighbors, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Step 7: radius density filtering.

    Neighbor count excludes the query point itself.
    """
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    if point_count == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)

    tree = cKDTree(points)
    neighbor_counts = np.zeros(point_count, dtype=np.int64)

    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        query = points[start:end]
        try:
            counts = tree.query_ball_point(query, r=radius, return_length=True, workers=-1)
        except TypeError:
            try:
                counts = tree.query_ball_point(query, r=radius, return_length=True)
            except TypeError:
                neighbors = tree.query_ball_point(query, r=radius)
                counts = np.asarray([len(item) for item in neighbors], dtype=np.int64)

        counts = np.asarray(counts, dtype=np.int64)
        neighbor_counts[start:end] = np.maximum(counts - 1, 0)

    return neighbor_counts < min_neighbors, neighbor_counts


def make_labels(attached_mask, statistical_iso_mask_full, radius_iso_mask_full):
    """
    Step 8: final labels with priority:
      attached structured noise > isolated outlier > retained surface.
    """
    labels = np.full(len(attached_mask), LABEL_SURFACE, dtype=np.int8)
    isolated_mask = (statistical_iso_mask_full | radius_iso_mask_full) & (~attached_mask)
    labels[isolated_mask] = LABEL_ISOLATED
    labels[attached_mask] = LABEL_ATTACHED
    return labels


def map_labels_to_original(original_points, used_points, used_labels, chunk_size=DEFAULT_CHUNK_SIZE):
    """Step 9: map used-point labels back to original resolution by nearest neighbor."""
    tree = cKDTree(used_points)
    original_labels = np.zeros(len(original_points), dtype=np.int8)

    for start in range(0, len(original_points), chunk_size):
        end = min(start + chunk_size, len(original_points))
        _, indices = kd_query(tree, original_points[start:end], k=1)
        original_labels[start:end] = used_labels[np.asarray(indices, dtype=np.int64)]

    return original_labels


def make_colored_cloud(labels):
    """Create class colors: gray surface, red attached noise, blue isolated outlier."""
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    colors[labels == LABEL_SURFACE] = COLOR_SURFACE
    colors[labels == LABEL_ATTACHED] = COLOR_ATTACHED
    colors[labels == LABEL_ISOLATED] = COLOR_ISOLATED
    return colors


def colors_for_subset(source_colors, labels, mask):
    """Use original colors if present; otherwise use semantic class colors."""
    if source_colors is not None:
        return source_colors[mask]
    return make_colored_cloud(labels[mask])


def save_classification_outputs(
    out_dir,
    suffix,
    points,
    labels,
    colors,
    save_colored=True,
):
    """Save surface/noise/classification PLY outputs for used or original points."""
    out_dir = Path(out_dir)
    surface_mask = labels == LABEL_SURFACE
    attached_mask = labels == LABEL_ATTACHED
    isolated_mask = labels == LABEL_ISOLATED
    noise_mask = attached_mask | isolated_mask

    output_paths = {}
    output_paths[f"surface_denoised_{suffix}"] = out_dir / f"surface_denoised_{suffix}.ply"
    output_paths[f"attached_noise_{suffix}"] = out_dir / f"attached_noise_{suffix}.ply"
    output_paths[f"isolated_noise_{suffix}"] = out_dir / f"isolated_noise_{suffix}.ply"
    output_paths[f"noise_removed_{suffix}"] = out_dir / f"noise_removed_{suffix}.ply"
    output_paths[f"label_colored_{suffix}"] = out_dir / f"label_colored_{suffix}.ply"

    save_point_cloud(
        output_paths[f"surface_denoised_{suffix}"],
        points[surface_mask],
        colors_for_subset(colors, labels, surface_mask),
    )
    save_point_cloud(
        output_paths[f"attached_noise_{suffix}"],
        points[attached_mask],
        colors_for_subset(colors, labels, attached_mask),
    )
    save_point_cloud(
        output_paths[f"isolated_noise_{suffix}"],
        points[isolated_mask],
        colors_for_subset(colors, labels, isolated_mask),
    )
    save_point_cloud(
        output_paths[f"noise_removed_{suffix}"],
        points[noise_mask],
        colors_for_subset(colors, labels, noise_mask),
    )

    if save_colored:
        save_point_cloud(
            output_paths[f"label_colored_{suffix}"],
            points,
            make_colored_cloud(labels),
        )

    return output_paths


def label_counts(labels):
    surface = int(np.sum(labels == LABEL_SURFACE))
    attached = int(np.sum(labels == LABEL_ATTACHED))
    isolated = int(np.sum(labels == LABEL_ISOLATED))
    return surface, attached, isolated, attached + isolated


def write_summary(
    out_dir,
    total_points,
    used_points_count,
    final_labels,
    curvature_threshold,
    attached_distance_threshold,
    isolated_distance_threshold,
    args,
):
    """Write denoise_summary.csv with required statistics."""
    out_dir = Path(out_dir)
    surface, attached, isolated, total_noise = label_counts(final_labels)
    denominator = max(len(final_labels), 1)

    summary = {
        "total_points": int(total_points),
        "used_points": int(used_points_count),
        "surface_points": surface,
        "attached_noise_points": attached,
        "isolated_noise_points": isolated,
        "total_noise_points": total_noise,
        "surface_ratio": surface / denominator,
        "noise_ratio": total_noise / denominator,
        "attached_ratio": attached / denominator,
        "isolated_ratio": isolated / denominator,
        "curvature_threshold": curvature_threshold,
        "attached_distance_threshold": attached_distance_threshold,
        "isolated_distance_threshold": isolated_distance_threshold,
        "radius": args.radius,
        "min_neighbors": args.min_neighbors,
    }

    summary_path = out_dir / "denoise_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary_path, summary


def write_report(
    out_dir,
    args,
    summary,
    used_counts,
    output_paths,
    summary_path,
    curvature_threshold,
    attached_distance_threshold,
    isolated_distance_threshold,
):
    """Write a text report with parameters, output paths, and step statistics."""
    out_dir = Path(out_dir)
    report_path = out_dir / "denoise_report.txt"
    used_surface, used_attached, used_isolated, used_noise = used_counts

    with report_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("Gateroad point-cloud denoising reproduction report\n")
        file_obj.write("=" * 72 + "\n\n")
        file_obj.write("Method:\n")
        file_obj.write(
            "curvature-constrained identification + statistical-density filtering\n\n"
        )

        file_obj.write("Input and parameters:\n")
        for key, value in vars(args).items():
            file_obj.write(f"  {key}: {value}\n")
        file_obj.write("\n")

        file_obj.write("Thresholds:\n")
        file_obj.write(f"  curvature_threshold: {curvature_threshold:.10f}\n")
        file_obj.write(f"  attached_distance_threshold: {attached_distance_threshold:.10f}\n")
        file_obj.write(f"  isolated_distance_threshold: {isolated_distance_threshold:.10f}\n")
        file_obj.write("\n")

        file_obj.write("Used-point classification statistics:\n")
        file_obj.write(f"  surface_points_used: {used_surface}\n")
        file_obj.write(f"  attached_noise_points_used: {used_attached}\n")
        file_obj.write(f"  isolated_noise_points_used: {used_isolated}\n")
        file_obj.write(f"  total_noise_points_used: {used_noise}\n")
        used_total = max(used_surface + used_noise, 1)
        file_obj.write(f"  denoising_rate_used: {used_noise / used_total:.6f}\n\n")

        file_obj.write("Final output statistics:\n")
        for key, value in summary.items():
            file_obj.write(f"  {key}: {value}\n")
        file_obj.write("\n")

        file_obj.write("Output files:\n")
        file_obj.write(f"  summary: {summary_path}\n")
        for key, path in sorted(output_paths.items()):
            file_obj.write(f"  {key}: {path}\n")

    return report_path


def run_denoising(points, args):
    """Run all algorithmic steps on used_points and return labels + thresholds."""
    timed_message("Computing curvature...")
    curvature = compute_curvature(points, kp=args.kp)

    timed_message("Computing neighbor statistics for attached-noise detection...")
    avg_dist = compute_average_neighbor_distance(points, k=args.stat_k)

    attached_mask, curvature_threshold, attached_threshold = detect_attached_noise(
        curvature=curvature,
        avg_dist=avg_dist,
        curvature_quantile=args.curvature_quantile,
        attached_alpha=args.attached_alpha,
    )
    timed_message(f"Attached noise points: {int(np.sum(attached_mask))}")

    remaining_after_attached = ~attached_mask
    remaining_indices = np.where(remaining_after_attached)[0]
    remaining_points = points[remaining_after_attached]

    timed_message("Detecting statistical isolated outliers...")
    stat_iso_remaining, isolated_threshold, _ = detect_isolated_noise(
        remaining_points,
        stat_k=args.stat_k,
        stat_alpha=args.stat_alpha,
    )
    stat_iso_full = np.zeros(len(points), dtype=bool)
    stat_iso_full[remaining_indices] = stat_iso_remaining
    timed_message(f"Statistical isolated points: {int(np.sum(stat_iso_full))}")

    remaining_after_stat = (~attached_mask) & (~stat_iso_full)
    radius_indices = np.where(remaining_after_stat)[0]
    radius_points = points[remaining_after_stat]

    timed_message("Running radius density filter...")
    radius_iso_remaining, neighbor_counts = radius_density_filter(
        radius_points,
        radius=args.radius,
        min_neighbors=args.min_neighbors,
    )
    radius_iso_full = np.zeros(len(points), dtype=bool)
    radius_iso_full[radius_indices] = radius_iso_remaining
    timed_message(f"Radius isolated points: {int(np.sum(radius_iso_full))}")

    labels = make_labels(
        attached_mask=attached_mask,
        statistical_iso_mask_full=stat_iso_full,
        radius_iso_mask_full=radius_iso_full,
    )

    surface, attached, isolated, total_noise = label_counts(labels)
    timed_message(f"Final surface points: {surface}")
    timed_message(f"Final attached noise points: {attached}")
    timed_message(f"Final isolated outlier points: {isolated}")
    timed_message(f"Final total noise points: {total_noise}")

    return labels, curvature_threshold, attached_threshold, isolated_threshold


def validate_args(args):
    if not Path(args.input).is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.voxel_size < 0:
        raise ValueError("--voxel_size must be >= 0")
    if args.radius <= 0:
        raise ValueError("--radius must be > 0")
    if args.min_neighbors < 1:
        raise ValueError("--min_neighbors must be >= 1")
    if args.kp < 1:
        raise ValueError("--kp must be >= 1")
    if args.stat_k < 1:
        raise ValueError("--stat_k must be >= 1")
    if not (0.0 <= args.curvature_quantile <= 1.0):
        raise ValueError("--curvature_quantile must be in [0, 1]")
    if args.attached_alpha < 0:
        raise ValueError("--attached_alpha must be >= 0")
    if args.stat_alpha < 0:
        raise ValueError("--stat_alpha must be >= 0")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce gateroad denoising by curvature-constrained attached-noise "
            "identification and statistical-density isolated-outlier filtering."
        )
    )
    parser.add_argument("--input", required=True, help="Input point cloud: .ply, .pcd, .xyz, .txt")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--map_to_original", action="store_true")
    parser.add_argument("--kp", type=int, default=30)
    parser.add_argument("--curvature_quantile", type=float, default=0.93)
    parser.add_argument("--attached_alpha", type=float, default=3.0)
    parser.add_argument("--stat_k", type=int, default=30)
    parser.add_argument("--stat_alpha", type=float, default=2.5)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--min_neighbors", type=int, default=8)
    parser.add_argument("--save_colored", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no_save_colored", dest="save_colored", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timed_message(f"Loading point cloud: {args.input}")
    original_points, original_colors = load_point_cloud(args.input)
    timed_message(f"Loaded points: {len(original_points)}")

    used_points, used_colors = voxel_downsample_if_needed(
        original_points,
        original_colors,
        voxel_size=args.voxel_size,
    )
    timed_message(f"Used points for computation: {len(used_points)}")

    if len(used_points) < 100:
        warnings.warn("Used point count is small. Denoising thresholds may be unstable.")

    used_labels, curvature_threshold, attached_threshold, isolated_threshold = run_denoising(
        used_points,
        args,
    )

    output_paths = {}
    timed_message("Saving used-point outputs...")
    output_paths.update(
        save_classification_outputs(
            out_dir=out_dir,
            suffix="used",
            points=used_points,
            labels=used_labels,
            colors=used_colors,
            save_colored=args.save_colored,
        )
    )

    final_labels = used_labels
    final_point_count = len(used_points)

    if args.map_to_original and args.voxel_size > 0:
        timed_message("Mapping labels back to original point cloud...")
        original_labels = map_labels_to_original(original_points, used_points, used_labels)
        timed_message("Saving original-resolution outputs...")
        output_paths.update(
            save_classification_outputs(
                out_dir=out_dir,
                suffix="original",
                points=original_points,
                labels=original_labels,
                colors=original_colors,
                save_colored=args.save_colored,
            )
        )
        final_labels = original_labels
        final_point_count = len(original_points)
    elif args.map_to_original and args.voxel_size <= 0:
        warnings.warn("--map_to_original was set, but voxel_size is 0. Original mapping is not needed.")

    summary_path, summary = write_summary(
        out_dir=out_dir,
        total_points=len(original_points),
        used_points_count=len(used_points),
        final_labels=final_labels,
        curvature_threshold=curvature_threshold,
        attached_distance_threshold=attached_threshold,
        isolated_distance_threshold=isolated_threshold,
        args=args,
    )

    report_path = write_report(
        out_dir=out_dir,
        args=args,
        summary=summary,
        used_counts=label_counts(used_labels),
        output_paths=output_paths,
        summary_path=summary_path,
        curvature_threshold=curvature_threshold,
        attached_distance_threshold=attached_threshold,
        isolated_distance_threshold=isolated_threshold,
    )

    surface, attached, isolated, total_noise = label_counts(final_labels)
    timed_message(f"Final output point count: {final_point_count}")
    timed_message(f"Final surface points: {surface}")
    timed_message(f"Final total noise points: {total_noise}")
    timed_message(f"Final noise ratio: {total_noise / max(final_point_count, 1):.6f}")
    timed_message(f"Saved outputs to: {out_dir}")
    timed_message(f"Summary CSV: {summary_path}")
    timed_message(f"Report TXT: {report_path}")


if __name__ == "__main__":
    main()

