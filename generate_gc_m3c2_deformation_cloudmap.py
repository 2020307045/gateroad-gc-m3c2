#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-epoch GC-M3C2 deformation cloud-map generator for gateroad point clouds.

Dependencies:
    pip install numpy pandas scipy matplotlib tqdm open3d laspy

Open3D is recommended for PLY/PCD reading and optional ICP.  LAS/LAZ reading
requires laspy.  Input point clouds are never modified; any downsampling is
performed in memory only.

Example (Windows cmd.exe):
    python generate_gc_m3c2_deformation_cloudmap.py ^
      --baseline "data/epoch1_clean.ply" ^
      --target "data/epoch2_clean.ply" ^
      --out_dir "data/gc_m3c2_cloudmap_outputs" ^
      --section_step 1.0 ^
      --section_width 0.5 ^
      --projection_radius 0.20 ^
      --half_length 0.30 ^
      --min_neighbors 8 ^
      --peak_percentile 95 ^
      --peak_alpha 0.40 ^
      --voxel_size 0.02 ^
      --vmin 0 --vmax 0.80 --cmap turbo
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


EPS = 1e-12
PLY_SCALAR_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return None if not np.isfinite(length) or length < EPS else vector / length


def read_ply_header(path):
    """Read a standard point-cloud PLY header without loading all vertices."""
    vertex_count, properties, current = None, [], None
    with open(path, "rb") as handle:
        if handle.readline().decode("ascii", errors="ignore").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        file_format = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            tokens = line.decode("ascii", errors="ignore").strip().split()
            if len(tokens) >= 3 and tokens[0] == "format":
                file_format = tokens[1]
            elif len(tokens) == 3 and tokens[0] == "element":
                current = tokens[1]
                if current == "vertex":
                    vertex_count = int(tokens[2])
            elif tokens and tokens[0] == "property" and current == "vertex":
                if len(tokens) >= 5 and tokens[1] == "list":
                    raise ValueError("PLY vertex list properties are unsupported.")
                if len(tokens) == 3:
                    properties.append((tokens[2], tokens[1].lower()))
            elif tokens == ["end_header"]:
                offset = handle.tell()
                break
    if file_format is None or vertex_count is None or not properties:
        raise ValueError(f"Invalid PLY vertex definition: {path}")
    return file_format, vertex_count, properties, offset


def uniformly_spaced_indices(count, maximum):
    if maximum is None or maximum <= 0 or count <= maximum:
        return None
    return np.linspace(0, count - 1, int(maximum), dtype=np.int64)


def xyz_property_names(properties):
    names = {name.lower(): name for name, _ in properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY must provide x, y, and z vertex properties.")
    return names["x"], names["y"], names["z"]


def load_ply_fallback(path, maximum):
    """Memory-bounded standard ASCII/binary PLY reader used when Open3D is absent."""
    file_format, count, properties, offset = read_ply_header(path)
    x_name, y_name, z_name = xyz_property_names(properties)
    samples = uniformly_spaced_indices(count, maximum)
    names = [name for name, _ in properties]
    if file_format == "ascii":
        column = {name: index for index, name in enumerate(names)}
        rows, sample_position = [], 0
        with open(path, "rb") as handle:
            handle.seek(offset)
            for row_index in range(count):
                line = handle.readline()
                if not line:
                    break
                keep = samples is None or (sample_position < len(samples) and row_index == samples[sample_position])
                if not keep:
                    continue
                if samples is not None:
                    sample_position += 1
                parts = line.split()
                try:
                    rows.append([float(parts[column[x_name]]), float(parts[column[y_name]]), float(parts[column[z_name]])])
                except (IndexError, ValueError):
                    continue
        return np.asarray(rows, dtype=float)
    if file_format not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {file_format}")
    endian = "<" if file_format == "binary_little_endian" else ">"
    dtype = np.dtype([(name, endian + PLY_SCALAR_DTYPES[type_name]) for name, type_name in properties])
    blocks = []
    with open(path, "rb") as handle:
        handle.seek(offset)
        for start in range(0, count, 1_000_000):
            length = min(1_000_000, count - start)
            records = np.fromfile(handle, dtype=dtype, count=length)
            if len(records) != length:
                break
            if samples is not None:
                lo = np.searchsorted(samples, start, side="left")
                hi = np.searchsorted(samples, start + length, side="left")
                local = samples[lo:hi] - start
                if not len(local):
                    continue
                records = records[local]
            blocks.append(np.column_stack([records[x_name], records[y_name], records[z_name]]))
    return np.vstack(blocks).astype(float, copy=False) if blocks else np.empty((0, 3), dtype=float)


def load_point_cloud(path, maximum=None):
    """Load XYZ data from PLY/PCD/XYZ/TXT/CSV/LAS/LAZ, preserving no input state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Point-cloud file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".ply", ".pcd"}:
        try:
            import open3d as o3d
            cloud = o3d.io.read_point_cloud(str(path))
            points = np.asarray(cloud.points, dtype=float)
            if len(points) == 0:
                raise ValueError(f"Open3D could not read points from {path}")
            if maximum and len(points) > maximum:
                points = points[uniformly_spaced_indices(len(points), maximum)]
        except ImportError:
            if suffix == ".pcd":
                raise ImportError("PCD reading requires Open3D. Install it with: pip install open3d")
            points = load_ply_fallback(path, maximum)
    elif suffix in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError("LAS/LAZ reading requires laspy. Install it with: pip install laspy") from exc
        las = laspy.read(path)
        points = np.column_stack([las.x, las.y, las.z])
        if maximum and len(points) > maximum:
            points = points[uniformly_spaced_indices(len(points), maximum)]
    elif suffix in {".txt", ".xyz"}:
        points = np.loadtxt(path, usecols=(0, 1, 2))
    elif suffix == ".csv":
        frame = pd.read_csv(path, header=None)
        points = frame.iloc[:, :3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported point-cloud type: {suffix}. Supported: .ply, .pcd, .xyz, .txt, .csv, .las, .laz")
    points = np.asarray(points, dtype=float)
    if points.ndim == 1 and points.size >= 3:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} does not contain an Nx3 point array.")
    return points[:, :3]


def voxel_downsample(points, voxel_size):
    """Retain one observed point per voxel; this avoids modifying source files."""
    if voxel_size is None or voxel_size <= 0:
        return points
    grid = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return points[np.sort(indices)]


def preprocess_point_cloud(points, voxel_size, max_points_compute, seed):
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty after removing NaN/Inf coordinates.")
    points = voxel_downsample(points, voxel_size)
    if len(points) == 0:
        raise ValueError("Point cloud is empty after voxel downsampling.")
    if max_points_compute and len(points) > max_points_compute:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points_compute, replace=False)]
    return points


def global_axial_stratified_cap(points, axis, max_points, seed, bins=120):
    """Cap compute points while preserving coverage along the entire roadway.

    This is not a section extraction.  Every populated axial bin contributes
    reference points, so the output remains a full-roadway deformation field.
    """
    if not max_points or len(points) <= max_points:
        return points
    axial = points @ axis
    bin_count = max(12, min(int(bins), int(max_points)))
    edges = np.linspace(float(axial.min()), float(axial.max()), bin_count + 1)
    assignment = np.clip(np.digitize(axial, edges) - 1, 0, bin_count - 1)
    per_bin = max(1, int(math.ceil(max_points / bin_count)))
    rng = np.random.default_rng(seed)
    selected = []
    for bin_index in range(bin_count):
        indices = np.flatnonzero(assignment == bin_index)
        if len(indices) <= per_bin:
            selected.append(indices)
        else:
            selected.append(rng.choice(indices, size=per_bin, replace=False))
    selected = np.concatenate(selected) if selected else np.empty(0, dtype=int)
    if len(selected) > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)
    return points[np.sort(selected)]


def estimate_roadway_axis(points, mode):
    if mode != "auto":
        axes = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
        return axes[mode]
    centred = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    axis = unit(vectors[0])
    if axis is None:
        raise ValueError("Could not estimate the roadway longitudinal axis using PCA.")
    component = int(np.argmax(np.abs(axis)))
    return -axis if axis[component] < 0 else axis


def make_frame(axis):
    """Construct vertical and transverse axes perpendicular to roadway axis."""
    world_vertical = np.array([0.0, 0.0, 1.0])
    vertical = unit(world_vertical - np.dot(world_vertical, axis) * axis)
    if vertical is None:
        fallback = np.array([0.0, 1.0, 0.0])
        vertical = unit(fallback - np.dot(fallback, axis) * axis)
    if vertical is None:
        raise ValueError("Could not construct a vertical local direction.")
    side = unit(np.cross(vertical, axis))
    if side is None:
        raise ValueError("Could not construct a transverse local direction.")
    return vertical, side


def smooth_rows(values, window=5):
    if len(values) < 3:
        return values.copy()
    window = max(1, min(int(window), len(values)))
    half = window // 2
    output = values.copy()
    for index in range(len(values)):
        output[index] = values[max(0, index - half): min(len(values), index + half + 1)].mean(axis=0)
    return output


def build_centerline(points, axis, section_step, section_width):
    """Build a centreline from continuous axial sections of the baseline cloud."""
    origin = points.mean(axis=0)
    axial = (points - origin) @ axis
    minimum, maximum = float(axial.min()), float(axial.max())
    edges = np.arange(minimum, maximum + section_step, section_step)
    if len(edges) < 3:
        raise ValueError("Roadway axial extent is too short to build a centreline.")
    centers, centre_t = [], []
    half_width = max(section_width / 2.0, section_step / 2.0)
    for centre in np.arange(minimum + section_step / 2.0, maximum, section_step):
        group = points[np.abs(axial - centre) <= half_width]
        if len(group) >= 3:
            centers.append(group.mean(axis=0))
            centre_t.append(centre)
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 3:
        raise ValueError("Fewer than three populated roadway sections; centreline estimation failed.")
    return smooth_rows(centers, window=5), origin, minimum, np.asarray(centre_t)


def pca_normal(neighbours):
    if len(neighbours) < 3:
        return None
    centred = neighbours - neighbours.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred)
    return unit(vectors[:, np.argmin(values)])


def cylinder_projection(points, tree, center, direction, radius, half_length):
    outer_radius = math.sqrt(radius * radius + half_length * half_length)
    indices = tree.query_ball_point(center, outer_radius)
    if not indices:
        return np.empty(0, dtype=float)
    neighbours = points[np.asarray(indices, dtype=int)]
    delta = neighbours - center
    projection = delta @ direction
    lateral = np.linalg.norm(delta - projection[:, None] * direction, axis=1)
    return projection[(np.abs(projection) <= half_length) & (lateral <= radius)]


def optional_icp(baseline, target, threshold, enabled):
    """Optionally refine target alignment; GC-M3C2 remains the deformation method."""
    baseline_tree = cKDTree(baseline)
    before = float(np.sqrt(np.mean(baseline_tree.query(target, k=1)[0] ** 2)))
    if not enabled:
        return target, {"used": False, "rmse_before": before, "rmse_after": before, "fitness": np.nan}
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("--do_icp requires Open3D. Install it with: pip install open3d") from exc
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    destination = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(baseline))
    result = o3d.pipelines.registration.registration_icp(
        source, destination, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformed = (target @ result.transformation[:3, :3].T) + result.transformation[:3, 3]
    after = float(np.sqrt(np.mean(baseline_tree.query(transformed, k=1)[0] ** 2)))
    return transformed, {"used": True, "rmse_before": before, "rmse_after": after, "fitness": float(result.fitness)}


def compute_gc_m3c2(baseline, target, axis, centerline, origin, axial_min, args):
    """Evaluate the proposed GC-M3C2 median-plus-peak deformation at all baseline points."""
    baseline_tree, target_tree = cKDTree(baseline), cKDTree(target)
    valid_centerline = centerline is not None and len(centerline) >= 3
    if valid_centerline:
        nearest_centers, _ = nearest_centerline_fast(baseline, centerline)
        directions = nearest_centers - baseline
        norms = np.linalg.norm(directions, axis=1)
        usable_direction = norms > EPS
        directions[usable_direction] /= norms[usable_direction, None]
    else:
        print("Warning: centerline estimation failed, fallback to PCA-based projection.")
        directions = np.full_like(baseline, np.nan)
        usable_direction = np.zeros(len(baseline), dtype=bool)
    signed = np.full(len(baseline), np.nan)
    baseline_count = np.zeros(len(baseline), dtype=np.int32)
    target_count = np.zeros(len(baseline), dtype=np.int32)
    normal_radius = max(args.projection_radius * 2.0, args.voxel_size * 3.0 if args.voxel_size > 0 else 0.05)
    straight_axis_origin = origin + ((baseline - origin) @ axis)[:, None] * axis

    for index, point in enumerate(tqdm(baseline, desc="GC-M3C2", unit="point")):
        direction = directions[index] if usable_direction[index] else None
        if direction is None or not np.all(np.isfinite(direction)):
            neighbourhood = baseline_tree.query_ball_point(point, normal_radius)
            normal = pca_normal(baseline[np.asarray(neighbourhood, dtype=int)]) if len(neighbourhood) else None
            toward_axis = straight_axis_origin[index] - point
            if normal is None:
                direction = unit(toward_axis)
            else:
                if np.dot(normal, toward_axis) < 0:
                    normal = -normal
                direction = normal
        if direction is None:
            continue
        base_projection = cylinder_projection(baseline, baseline_tree, point, direction, args.projection_radius, args.half_length)
        target_projection = cylinder_projection(target, target_tree, point, direction, args.projection_radius, args.half_length)
        baseline_count[index], target_count[index] = len(base_projection), len(target_projection)
        if len(base_projection) < args.min_neighbors or len(target_projection) < args.min_neighbors:
            continue
        median_distance = np.median(target_projection) - np.median(base_projection)
        peak_distance = np.percentile(target_projection, args.peak_percentile) - np.percentile(base_projection, args.peak_percentile)
        signed[index] = (1.0 - args.peak_alpha) * median_distance + args.peak_alpha * peak_distance
    chainage = (baseline - origin) @ axis - axial_min
    section_number = np.floor(np.maximum(chainage, 0.0) / args.section_step).astype(int) + 1
    return signed, baseline_count, target_count, chainage, section_number


def nearest_centerline_fast(points, centerline):
    tree = cKDTree(centerline)
    _, indices = tree.query(points, k=1)
    return centerline[indices], indices


def automatic_vmax(values):
    maximum = float(np.nanmax(values))
    if maximum <= 0.1:
        return max(0.01, math.ceil(maximum * 100.0) / 100.0)
    if maximum <= 1.0:
        return math.ceil(maximum * 10.0) / 10.0
    return math.ceil(maximum * 2.0) / 2.0


def colorbar_range(values, vmin, vmax):
    lower = 0.0 if vmin is None else float(vmin)
    upper = automatic_vmax(values) if vmax is None else float(vmax)
    if upper <= lower:
        raise ValueError(f"Invalid colourbar range: vmin={lower}, vmax={upper}")
    return lower, upper


def range_legend_text(vmax):
    if vmax >= 0.40:
        bounds = [0.00, 0.10, 0.20, 0.40, vmax]
    else:
        bounds = [0.00, vmax * 0.25, vmax * 0.50, vmax * 0.75, vmax]
    bounds = np.unique(np.round(bounds, 6))
    labels = ["slight deformation", "moderate deformation", "large deformation", "severe deformation"]
    lines = ["Deformation range legend (colour mapping uses abs_gc_m3c2_distance):"]
    for index in range(len(bounds) - 1):
        right_symbol = ">=" if index == len(bounds) - 2 else ""
        lines.append(f"{bounds[index]:.2f}-{bounds[index + 1]:.2f} m: {labels[min(index, len(labels) - 1)]}{' (' + right_symbol + ' upper range)' if right_symbol else ''}")
    return "\n".join(lines) + "\n"


def map_colors(values, vmin, vmax, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    normalised = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.rint(cmap(normalised)[:, :3] * 255).astype(np.uint8)


def save_colored_point_cloud(path, points, signed, absolute, colors):
    """Write a standards-friendly ASCII PLY with colours and scalar properties."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        for name in ("x", "y", "z", "signed_gc_m3c2_distance", "abs_gc_m3c2_distance"):
            handle.write(f"property float {name}\n")
        for name in ("red", "green", "blue"):
            handle.write(f"property uchar {name}\n")
        handle.write("end_header\n")
        for point, s_value, a_value, color in zip(points, signed, absolute, colors):
            handle.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {s_value:.8f} {a_value:.8f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def save_statistics(path, absolute, valid, vmin, vmax, icp_info):
    values = absolute[valid]
    lines = [
        "Method: GC-M3C2 (Geometrically Constrained M3C2)",
        f"Valid point count: {int(valid.sum())}",
        f"Invalid point count: {int((~valid).sum())}",
        f"Valid point ratio: {valid.mean() * 100:.2f}%",
        f"Minimum deformation: {np.min(values):.6f} m",
        f"Maximum deformation: {np.max(values):.6f} m",
        f"Mean deformation: {np.mean(values):.6f} m",
        f"Median deformation: {np.median(values):.6f} m",
        f"Standard deviation: {np.std(values):.6f} m",
        f"95th percentile deformation: {np.percentile(values, 95):.6f} m",
        f"Colorbar range: {vmin:.2f}-{vmax:.2f} m",
        f"ICP used: {icp_info['used']}",
        f"ICP RMSE before: {icp_info['rmse_before']:.6f} m",
        f"ICP RMSE after: {icp_info['rmse_after']:.6f} m",
    ]
    if icp_info["used"]:
        lines.append(f"ICP fitness: {icp_info['fitness']:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}


def plot_setup():
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.unicode_minus": False, "svg.fonttype": "none"})


def plotted_subset(points, absolute, max_points_plot, seed):
    if not max_points_plot or len(points) <= max_points_plot:
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    return rng.choice(len(points), size=max_points_plot, replace=False)


def add_colorbar(figure, artist, axes, vmin, vmax):
    colorbar = figure.colorbar(artist, ax=axes, pad=0.02, shrink=0.88)
    colorbar.set_label("Deformation (m)")
    colorbar.set_ticks(np.linspace(vmin, vmax, 9))
    return colorbar


def plot_3d_cloudmap(points, absolute, output, vmin, vmax, args):
    figure = plt.figure(figsize=(args.figure_width, args.figure_height), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    artist = axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=absolute, cmap=args.cmap, vmin=vmin, vmax=vmax, s=1.0, linewidths=0, rasterized=True)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.view_init(elev=args.elev, azim=args.azim)
    add_colorbar(figure, artist, axis, vmin, vmax)
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_projection(points, absolute, axis_vector, vertical, side, origin, output, vmin, vmax, args, mode, paper=False):
    delta = points - origin
    longitudinal = delta @ axis_vector
    transverse = delta @ (vertical if mode == "side" else side)
    figure, axis = plt.subplots(figsize=(args.figure_width, args.figure_height), facecolor="white")
    artist = axis.scatter(longitudinal, transverse, c=absolute, cmap=args.cmap, vmin=vmin, vmax=vmax, s=1.2, linewidths=0, rasterized=True)
    axis.set_xlabel("Roadway longitudinal coordinate / m")
    axis.set_ylabel("Vertical coordinate / m" if mode == "side" else "Transverse coordinate / m")
    axis.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.55)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
    add_colorbar(figure, artist, axis, vmin, vmax)
    figure.tight_layout()
    if paper:
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(output.with_suffix("." + suffix), dpi=args.dpi, facecolor="white", bbox_inches="tight")
    else:
        figure.savefig(output, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_histogram(values, output, vmin, vmax):
    figure, axis = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
    axis.hist(values, bins=60, range=(vmin, vmax), color="#2E75B6", edgecolor="white", linewidth=0.35)
    axis.set_xlabel("Deformation (m)")
    axis.set_ylabel("Point count")
    axis.grid(True, axis="y", color="#D7D7D7", linewidth=0.5, alpha=0.65)
    figure.tight_layout()
    figure.savefig(output, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description="GC-M3C2 gateroad deformation cloud-map generator.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline", required=True, help="Baseline/reference point cloud.")
    parser.add_argument("--target", required=True, help="Target point cloud.")
    parser.add_argument("--out_dir", required=True, help="Directory for GC-M3C2 outputs.")
    parser.add_argument("--section_step", type=float, default=1.0, help="Axial bin length used only to build the full-roadway centreline (m); it does not restrict GC-M3C2 output to one section.")
    parser.add_argument("--section_width", type=float, default=0.5, help="Width used only while estimating centreline section centroids (m).")
    parser.add_argument("--projection_radius", type=float, default=0.20)
    parser.add_argument("--half_length", type=float, default=0.30)
    parser.add_argument("--min_neighbors", type=int, default=8)
    parser.add_argument("--peak_percentile", type=float, default=95.0)
    parser.add_argument("--peak_alpha", type=float, default=0.40)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--do_icp", action="store_true", help="Apply optional point-to-point ICP to the target cloud.")
    parser.add_argument("--icp_threshold", type=float, default=0.50)
    parser.add_argument("--compute_voxel_size", action="store_true", help="Estimate a coarse voxel size from nearest-neighbour spacing when --voxel_size <= 0.")
    parser.add_argument("--max_points_compute", type=int, default=0, help="Maximum globally stratified points per epoch for full-roadway GC-M3C2; 0 keeps all processed points.")
    parser.add_argument("--max_points_plot", type=int, default=0, help="Maximum points displayed in raster figures only; 0 displays all computed valid points.")
    parser.add_argument("--figure_width", type=float, default=16.0, help="Output figure width in inches.")
    parser.add_argument("--figure_height", type=float, default=9.0, help="Output figure height in inches.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster-output resolution in dots per inch.")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--absolute", action=argparse.BooleanOptionalAction, default=True, help="Use absolute distance for visualisation (recommended).")
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    for name in ("section_step", "section_width", "projection_radius", "half_length", "min_neighbors"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be greater than zero.")
    if args.max_points_compute < 0 or args.max_points_plot < 0:
        raise ValueError("--max_points_compute and --max_points_plot must be zero or positive.")
    if args.figure_width <= 0 or args.figure_height <= 0 or args.dpi <= 0:
        raise ValueError("--figure_width, --figure_height, and --dpi must be greater than zero.")
    if not 0 < args.peak_percentile < 100:
        raise ValueError("--peak_percentile must be strictly between 0 and 100.")
    if not 0 <= args.peak_alpha <= 1:
        raise ValueError("--peak_alpha must be between 0 and 1.")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    plot_setup()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_limit = args.max_points_compute * 3 if args.max_points_compute else None
    baseline_raw = load_point_cloud(args.baseline, raw_limit)
    target_raw = load_point_cloud(args.target, raw_limit)
    voxel_size = args.voxel_size
    if args.compute_voxel_size and voxel_size <= 0:
        sample = baseline_raw[: min(len(baseline_raw), 50000)]
        distances = cKDTree(sample).query(sample, k=2)[0][:, 1]
        voxel_size = float(np.median(distances) * 1.5)
        print(f"Estimated voxel size: {voxel_size:.6f} m")
    # Voxel processing is applied to the whole loaded roadway.  The optional
    # compute cap is imposed only afterwards with axial stratification, so all
    # portions of the roadway contribute to the global deformation cloud map.
    baseline_voxel = preprocess_point_cloud(baseline_raw, voxel_size, 0, args.seed)
    target_voxel = preprocess_point_cloud(target_raw, voxel_size, 0, args.seed + 1)
    axis = estimate_roadway_axis(baseline_voxel, args.axis)
    baseline = global_axial_stratified_cap(baseline_voxel, axis, args.max_points_compute, args.seed)
    target = global_axial_stratified_cap(target_voxel, axis, args.max_points_compute, args.seed + 1)
    target, icp_info = optional_icp(baseline, target, args.icp_threshold, args.do_icp)
    vertical, side = make_frame(axis)
    try:
        centerline, origin, axial_min, _ = build_centerline(baseline, axis, args.section_step, args.section_width)
        centerline_ok = True
    except ValueError as exc:
        warnings.warn(f"Warning: centerline estimation failed, fallback to PCA-based projection. ({exc})", RuntimeWarning)
        centerline, origin, axial_min, centerline_ok = None, baseline.mean(axis=0), float(np.min((baseline - baseline.mean(axis=0)) @ axis)), False
    signed, base_count, target_count, chainage, section_number = compute_gc_m3c2(baseline, target, axis, centerline, origin, axial_min, args)
    absolute = np.abs(signed)
    valid = np.isfinite(signed)
    if valid.sum() == 0:
        raise RuntimeError("No valid GC-M3C2 points: increase neighbourhood dimensions or inspect registration/data density.")
    if not args.absolute:
        warnings.warn("--no-absolute does not alter cloud-map colouring: the paper cloud maps always use abs_gc_m3c2_distance. Signed values remain available in CSV/PLY output.", RuntimeWarning)
    vmin, vmax = colorbar_range(absolute[valid], args.vmin, args.vmax)
    colors = map_colors(absolute[valid], vmin, vmax, args.cmap)

    table = pd.DataFrame(
        {
            "point_id": np.arange(len(baseline), dtype=int), "x": baseline[:, 0], "y": baseline[:, 1], "z": baseline[:, 2],
            "signed_gc_m3c2_distance": signed, "abs_gc_m3c2_distance": absolute,
            "baseline_neighbor_count": base_count, "target_neighbor_count": target_count,
            "valid_flag": valid.astype(int), "section_id": [f"S{number:04d}" for number in section_number], "mileage_m": chainage,
        }
    )
    csv_path = output_dir / "gc_m3c2_deformation_values.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    coloured_ply_path = output_dir / "gc_m3c2_deformation_colored.ply"
    save_colored_point_cloud(coloured_ply_path, baseline[valid], signed[valid], absolute[valid], colors)
    statistics_path = output_dir / "gc_m3c2_deformation_statistics.txt"
    statistics = save_statistics(statistics_path, absolute, valid, vmin, vmax, icp_info)
    legend_path = output_dir / "deformation_range_legend.txt"
    legend_path.write_text(range_legend_text(vmax), encoding="utf-8")
    generated = [csv_path, coloured_ply_path, statistics_path, legend_path]
    if args.save_npy:
        npy_path = output_dir / "gc_m3c2_deformation_values.npy"
        np.save(npy_path, table.to_records(index=False))
        generated.append(npy_path)
    display_indices = plotted_subset(baseline[valid], absolute[valid], args.max_points_plot, args.seed)
    display_points, display_abs = baseline[valid][display_indices], absolute[valid][display_indices]
    if not args.max_points_plot and len(display_points) > 2_000_000:
        warnings.warn(
            f"Plotting all {len(display_points):,} valid points. This can require substantial RAM and time at {args.figure_width} x {args.figure_height} in, {args.dpi} dpi.",
            RuntimeWarning,
        )
    plot_3d_path = output_dir / "gc_m3c2_cloudmap_3d.png"
    side_path = output_dir / "gc_m3c2_cloudmap_side_view.png"
    plan_path = output_dir / "gc_m3c2_cloudmap_plan_view.png"
    paper_path = output_dir / "gc_m3c2_deformation_cloudmap_paper.png"
    plot_3d_cloudmap(display_points, display_abs, plot_3d_path, vmin, vmax, args)
    plot_projection(display_points, display_abs, axis, vertical, side, origin, side_path, vmin, vmax, args, "side")
    plot_projection(display_points, display_abs, axis, vertical, side, origin, plan_path, vmin, vmax, args, "plan")
    plot_projection(display_points, display_abs, axis, vertical, side, origin, paper_path, vmin, vmax, args, "side", paper=True)
    histogram_path = output_dir / "gc_m3c2_deformation_histogram.png"
    plot_histogram(absolute[valid], histogram_path, vmin, vmax)
    generated.extend([plot_3d_path, side_path, plan_path, paper_path, paper_path.with_suffix(".svg"), paper_path.with_suffix(".pdf"), histogram_path])
    if centerline_ok:
        centerline_path = output_dir / "gc_m3c2_centerline.npy"
        np.save(centerline_path, centerline)
        generated.append(centerline_path)

    print("\nGC-M3C2 deformation analysis complete")
    print("Analysis scope: full roadway surface (all retained baseline points; no representative cross-section extraction)")
    print(f"Compute point cap: {'all processed points' if not args.max_points_compute else args.max_points_compute}")
    print(f"Plot display cap: {'all valid points' if not args.max_points_plot else args.max_points_plot}")
    print(f"Baseline file: {Path(args.baseline).resolve()}")
    print(f"Target file: {Path(args.target).resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print("Method: GC-M3C2")
    print(f"Estimated roadway axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
    print(f"ICP used: {icp_info['used']}; RMSE before/after: {icp_info['rmse_before']:.6f}/{icp_info['rmse_after']:.6f} m")
    print(f"Valid point count: {int(valid.sum())}; invalid point count: {int((~valid).sum())}; valid ratio: {valid.mean() * 100:.2f}%")
    print(f"Minimum/maximum/mean deformation: {statistics['min']:.6f}/{statistics['max']:.6f}/{statistics['mean']:.6f} m")
    print(f"95th percentile deformation: {statistics['p95']:.6f} m")
    print(f"Colorbar range: {vmin:.2f}-{vmax:.2f} m")
    print("Generated files:")
    for path in generated:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()

