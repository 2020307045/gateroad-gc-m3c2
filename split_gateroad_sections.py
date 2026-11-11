#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split a gateroad cloud into axial or centerline-based sections.

See README.md for portable command-line examples.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import open3d as o3d


AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}

PROJECTION_BLOCK_SIZE = 50000


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Split a gateroad point cloud by axis coordinate or centerline station."
    )
    parser.add_argument(
        "--input",
        required=True,
        help='Input point cloud, for example data/epoch1.ply',
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help='Output directory, for example results/sections',
    )
    parser.add_argument(
        "--slice_mode",
        choices=("axis", "centerline"),
        default="centerline",
        help="切片模式：axis 为旧版固定坐标轴切片，centerline 为中心线里程切片，默认 centerline",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="y",
        help="axis 模式下的巷道轴向，可选 x、y、z，默认 y",
    )
    parser.add_argument(
        "--start_coord",
        type=float,
        default=None,
        help="切片起始坐标；axis 模式表示坐标值，centerline 模式表示起始里程 s；不设置时自动取最小值",
    )
    parser.add_argument(
        "--total_length",
        type=float,
        default=60.0,
        help="总切片长度，默认 60.0",
    )
    parser.add_argument(
        "--section_length",
        type=float,
        default=2.0,
        help="每个切片长度，默认 2.0",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=100,
        help="每个切片最小点数，默认 100",
    )
    parser.add_argument(
        "--prefix",
        default="section",
        help='输出文件名前缀，默认 "section"',
    )
    parser.add_argument(
        "--centerline_csv",
        default=None,
        help="可选：中心线 CSV 文件，至少包含 x,y,z 三列；提供时优先使用该中心线",
    )
    parser.add_argument(
        "--centerline_axis",
        choices=("x", "y", "z"),
        default="y",
        help="未提供 centerline_csv 时，自动估计中心线使用的粗略轴向，默认 y",
    )
    parser.add_argument(
        "--centerline_bin_width",
        type=float,
        default=1.0,
        help="自动估计中心线时的粗切片宽度，默认 1.0 m",
    )
    parser.add_argument(
        "--centerline_smooth_window",
        type=int,
        default=5,
        help="自动估计中心线时的移动平均平滑窗口，默认 5",
    )
    parser.add_argument(
        "--max_projection_distance",
        type=float,
        default=10.0,
        help="点到中心线最大允许投影距离，超过该距离的点不参与切片，默认 10.0 m",
    )
    return parser.parse_args()


def validate_args(args):
    """检查参数是否合理，尽早给出清晰错误。"""
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入点云不存在: {input_path}")

    if args.centerline_csv is not None and not Path(args.centerline_csv).is_file():
        raise FileNotFoundError(f"中心线 CSV 不存在: {args.centerline_csv}")

    if args.total_length <= 0:
        raise ValueError("--total_length 必须大于 0")
    if args.section_length <= 0:
        raise ValueError("--section_length 必须大于 0")
    if args.min_points < 0:
        raise ValueError("--min_points 不能小于 0")
    if args.centerline_bin_width <= 0:
        raise ValueError("--centerline_bin_width 必须大于 0")
    if args.centerline_smooth_window < 1:
        raise ValueError("--centerline_smooth_window 必须大于等于 1")
    if args.max_projection_distance <= 0:
        raise ValueError("--max_projection_distance 必须大于 0")


def section_count_from_lengths(total_length, section_length):
    """根据总长度和切片长度计算切片数量。60/2 时得到 30 个切片。"""
    return int(math.ceil(total_length / section_length - 1e-12))


def make_section_point_cloud(source_pcd, section_points, mask):
    """
    根据布尔 mask 生成切片点云。

    坐标直接来自原始点云，不做去噪、下采样、平移、归一化或任何变换。
    如果原点云带颜色或法向量，则保留对应点的属性。
    """
    section_pcd = o3d.geometry.PointCloud()
    section_pcd.points = o3d.utility.Vector3dVector(section_points)

    if source_pcd.has_colors():
        colors = np.asarray(source_pcd.colors)
        section_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    if source_pcd.has_normals():
        normals = np.asarray(source_pcd.normals)
        section_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return section_pcd


def cumulative_centerline_s(centerline_points):
    """计算中心线点的累计弧长 s。"""
    if len(centerline_points) < 2:
        raise ValueError("中心线至少需要 2 个点")

    segment_lengths = np.linalg.norm(np.diff(centerline_points, axis=0), axis=1)
    centerline_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    if centerline_s[-1] <= 0:
        raise ValueError("中心线总长度为 0，请检查中心线点是否重复")

    return centerline_s


def read_centerline_csv(csv_path):
    """从 CSV 读取中心线点，要求至少包含 x,y,z 三列，按文件行顺序构建折线。"""
    points = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"中心线 CSV 没有表头: {csv_path}")

        normalized_fields = {field.strip().lower() for field in reader.fieldnames}
        missing_fields = {"x", "y", "z"} - normalized_fields
        if missing_fields:
            raise ValueError(f"中心线 CSV 缺少字段: {', '.join(sorted(missing_fields))}")

        for row in reader:
            normalized_row = {
                key.strip().lower(): value
                for key, value in row.items()
                if key is not None
            }
            points.append(
                [
                    float(normalized_row["x"]),
                    float(normalized_row["y"]),
                    float(normalized_row["z"]),
                ]
            )

    centerline_points = np.asarray(points, dtype=np.float64)
    centerline_points = centerline_points[np.isfinite(centerline_points).all(axis=1)]

    if len(centerline_points) < 2:
        raise ValueError("中心线 CSV 中有效点少于 2 个")

    return remove_consecutive_duplicate_points(centerline_points)


def remove_consecutive_duplicate_points(points, eps=1e-9):
    """删除相邻重复中心线点，避免产生零长度线段。"""
    if len(points) <= 1:
        return points

    keep = np.ones(len(points), dtype=bool)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep[1:] = segment_lengths > eps
    return points[keep]


def smooth_centerline_points(centerline_points, window):
    """
    对自动估计的中心线做简单移动平均平滑。

    这里不改变点云本身，只平滑用于计算里程的中心线。
    """
    if window <= 1 or len(centerline_points) <= 2:
        return centerline_points

    radius = window // 2
    smoothed = np.empty_like(centerline_points)

    for idx in range(len(centerline_points)):
        start = max(0, idx - radius)
        end = min(len(centerline_points), idx + radius + 1)
        smoothed[idx] = centerline_points[start:end].mean(axis=0)

    return smoothed


def estimate_centerline_from_points(points, axis, bin_width, smooth_window, min_points):
    """
    自动估计中心线。

    方法：沿粗略轴向分箱，每个箱内用横向坐标中位数作为中心线横向位置；
    轴向坐标使用 bin center。默认会删除点数过少的箱，减少孤立噪声对中心线的影响。
    """
    axis_index = AXIS_TO_INDEX[axis]
    transverse_indices = [idx for idx in range(3) if idx != axis_index]
    axis_values = points[:, axis_index]

    min_axis = float(np.min(axis_values))
    max_axis = float(np.max(axis_values))
    if max_axis <= min_axis:
        raise ValueError("自动估计中心线失败：所选 centerline_axis 上坐标范围为 0")

    min_bin_points = max(5, min(30, max(1, min_points // 10)))
    bin_edges = np.arange(min_axis, max_axis + bin_width, bin_width, dtype=np.float64)
    if len(bin_edges) < 2:
        bin_edges = np.array([min_axis, max_axis], dtype=np.float64)

    centerline_points = []
    for bin_idx in range(len(bin_edges) - 1):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]
        if bin_idx == len(bin_edges) - 2:
            mask = (axis_values >= bin_start) & (axis_values <= bin_end)
        else:
            mask = (axis_values >= bin_start) & (axis_values < bin_end)

        bin_points = points[mask]
        if len(bin_points) < min_bin_points:
            continue

        center_point = np.zeros(3, dtype=np.float64)
        center_point[axis_index] = (bin_start + bin_end) * 0.5
        center_point[transverse_indices] = np.median(bin_points[:, transverse_indices], axis=0)
        centerline_points.append(center_point)

    centerline_points = np.asarray(centerline_points, dtype=np.float64)
    if len(centerline_points) < 2:
        raise ValueError(
            "自动估计中心线失败：有效中心线点少于 2 个。"
            "可尝试增大 --centerline_bin_width、改 --centerline_axis，或提供 --centerline_csv。"
        )

    original_axis_values = centerline_points[:, axis_index].copy()
    centerline_points = smooth_centerline_points(centerline_points, smooth_window)
    # 自动中心线的轴向坐标来自 bin center，应保持不动；这里只平滑横向中心位置。
    centerline_points[:, axis_index] = original_axis_values
    centerline_points = remove_consecutive_duplicate_points(centerline_points)

    if len(centerline_points) < 2:
        raise ValueError("自动估计中心线失败：平滑后有效中心线点少于 2 个")

    return centerline_points


def project_points_to_centerline(points, centerline_points, block_size=PROJECTION_BLOCK_SIZE):
    """
    将点云投影到中心线折线，返回每个点的中心线里程和投影距离。

    输入：
        points: N x 3 点云坐标
        centerline_points: M x 3 中心线点，相邻点构成 M-1 条线段

    返回：
        point_s: N 个点在中心线上的累计里程
        projection_distance: N 个点到最近中心线线段的距离
        nearest_segment_id: N 个点最近的中心线线段编号，0 表示第 1 条线段

    为避免内存过大，点云按 block_size 分块处理。每个块内逐线段更新最近距离，
    不一次性构造 N x segment_count 的距离矩阵。
    """
    points = np.asarray(points, dtype=np.float64)
    centerline_points = np.asarray(centerline_points, dtype=np.float64)
    centerline_s = cumulative_centerline_s(centerline_points)

    segment_starts = centerline_points[:-1]
    segment_vectors = centerline_points[1:] - centerline_points[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_length_sq = segment_lengths ** 2
    valid_segment_ids = np.where(segment_length_sq > 1e-18)[0]

    if len(valid_segment_ids) == 0:
        raise ValueError("中心线没有有效线段")

    point_count = len(points)
    point_s = np.full(point_count, np.nan, dtype=np.float64)
    projection_distance = np.full(point_count, np.inf, dtype=np.float64)
    nearest_segment_id = np.full(point_count, -1, dtype=np.int32)

    for block_start in range(0, point_count, block_size):
        block_end = min(block_start + block_size, point_count)
        block = points[block_start:block_end]
        block_size_actual = block_end - block_start

        best_dist_sq = np.full(block_size_actual, np.inf, dtype=np.float64)
        best_s = np.full(block_size_actual, np.nan, dtype=np.float64)
        best_segment = np.full(block_size_actual, -1, dtype=np.int32)

        for segment_id in valid_segment_ids:
            seg_start = segment_starts[segment_id]
            seg_vector = segment_vectors[segment_id]
            seg_len = segment_lengths[segment_id]
            seg_len_sq = segment_length_sq[segment_id]

            t = ((block - seg_start) @ seg_vector) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            projection = seg_start + t[:, None] * seg_vector
            dist_sq = np.sum((block - projection) ** 2, axis=1)

            better = dist_sq < best_dist_sq
            if np.any(better):
                best_dist_sq[better] = dist_sq[better]
                best_s[better] = centerline_s[segment_id] + t[better] * seg_len
                best_segment[better] = segment_id

        point_s[block_start:block_end] = best_s
        projection_distance[block_start:block_end] = np.sqrt(best_dist_sq)
        nearest_segment_id[block_start:block_end] = best_segment

    return point_s, projection_distance, nearest_segment_id


def write_centerline_outputs(out_dir, centerline_points):
    """输出中心线点云 PLY 和带里程 s 的 CSV。"""
    centerline_s = cumulative_centerline_s(centerline_points)

    centerline_pcd = o3d.geometry.PointCloud()
    centerline_pcd.points = o3d.utility.Vector3dVector(centerline_points)
    centerline_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(centerline_points), 1))
    )

    centerline_ply_path = out_dir / "centerline_points.ply"
    ok = o3d.io.write_point_cloud(str(centerline_ply_path), centerline_pcd, write_ascii=False)
    if not ok:
        raise RuntimeError(f"中心线点云写入失败: {centerline_ply_path}")

    centerline_csv_path = out_dir / "centerline_points.csv"
    with centerline_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["centerline_id", "x", "y", "z", "s"])
        writer.writeheader()
        for idx, (point, station) in enumerate(zip(centerline_points, centerline_s), start=1):
            writer.writerow(
                {
                    "centerline_id": idx,
                    "x": f"{point[0]:.6f}",
                    "y": f"{point[1]:.6f}",
                    "z": f"{point[2]:.6f}",
                    "s": f"{station:.6f}",
                }
            )

    return centerline_ply_path, centerline_csv_path


def write_projection_summary(
    out_dir,
    total_points,
    valid_mask,
    point_s,
    projection_distance,
):
    """输出点到中心线投影的整体统计。"""
    valid_count = int(np.count_nonzero(valid_mask))
    invalid_count = int(total_points - valid_count)

    if valid_count > 0:
        valid_distances = projection_distance[valid_mask]
        valid_s = point_s[valid_mask]
        mean_distance = f"{float(np.mean(valid_distances)):.6f}"
        max_distance = f"{float(np.max(valid_distances)):.6f}"
        min_s = f"{float(np.min(valid_s)):.6f}"
        max_s = f"{float(np.max(valid_s)):.6f}"
    else:
        mean_distance = ""
        max_distance = ""
        min_s = ""
        max_s = ""

    summary_path = out_dir / "point_projection_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "total_points",
            "valid_projected_points",
            "invalid_projection_points",
            "mean_projection_distance",
            "max_projection_distance",
            "min_s",
            "max_s",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "total_points": int(total_points),
                "valid_projected_points": valid_count,
                "invalid_projection_points": invalid_count,
                "mean_projection_distance": mean_distance,
                "max_projection_distance": max_distance,
                "min_s": min_s,
                "max_s": max_s,
            }
        )

    return summary_path


def run_axis_mode(args, pcd, points, out_dir):
    """兼容旧版：按固定坐标轴 x/y/z 直接切片。"""
    axis_index = AXIS_TO_INDEX[args.axis]
    axis_values = points[:, axis_index]

    start_coord = args.start_coord
    if start_coord is None:
        start_coord = float(axis_values.min())

    total_end_coord = start_coord + args.total_length
    section_count = section_count_from_lengths(args.total_length, args.section_length)

    print(f"切片模式: axis")
    print(f"切片轴向: {args.axis}")
    print(f"起始坐标: {start_coord:.6f}")
    print(f"结束坐标: {total_end_coord:.6f}")
    print(f"切片长度: {args.section_length:.6f}")
    print(f"切片数量: {section_count}")
    print(f"输出文件夹: {out_dir}")
    print("-" * 80)

    summary_rows = []

    for section_idx in range(section_count):
        section_no = section_idx + 1
        section_start = start_coord + section_idx * args.section_length
        section_end = min(section_start + args.section_length, total_end_coord)

        # 区间采用 [start, end)，最后一个切片采用 [start, end]。
        # 这样边界点不会被重复写入，同时保留 total_end_coord 上的点。
        if section_idx == section_count - 1:
            mask = (axis_values >= section_start) & (axis_values <= section_end)
        else:
            mask = (axis_values >= section_start) & (axis_values < section_end)

        section_points = points[mask]
        point_count = int(section_points.shape[0])

        saved = point_count >= args.min_points
        file_path = out_dir / f"{args.prefix}_{section_no:03d}.ply"
        file_for_csv = ""

        if saved:
            section_pcd = make_section_point_cloud(pcd, section_points, mask)
            ok = o3d.io.write_point_cloud(str(file_path), section_pcd, write_ascii=False)
            if not ok:
                raise RuntimeError(f"切片写入失败: {file_path}")
            file_for_csv = str(file_path)

        summary_rows.append(
            {
                "section": f"{args.prefix}_{section_no:03d}",
                "start_coord": f"{section_start:.6f}",
                "end_coord": f"{section_end:.6f}",
                "point_count": point_count,
                "saved": saved,
                "file": file_for_csv,
            }
        )

        print(
            f"{args.prefix}_{section_no:03d}: "
            f"{section_start:.6f} -> {section_end:.6f}, "
            f"points={point_count}, saved={saved}"
        )

    summary_path = out_dir / "section_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = ["section", "start_coord", "end_coord", "point_count", "saved", "file"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 80)
    print(f"切片输出文件夹: {out_dir}")
    print(f"统计文件路径: {summary_path}")


def run_centerline_mode(args, pcd, points, out_dir):
    """按中心线里程 s 切片，适合弯曲巷道。"""
    if args.centerline_csv:
        print(f"读取中心线 CSV: {args.centerline_csv}")
        centerline_points = read_centerline_csv(args.centerline_csv)
        centerline_source = "csv"
    else:
        print("未提供中心线 CSV，开始从点云自动估计中心线")
        centerline_points = estimate_centerline_from_points(
            points=points,
            axis=args.centerline_axis,
            bin_width=args.centerline_bin_width,
            smooth_window=args.centerline_smooth_window,
            min_points=args.min_points,
        )
        centerline_source = "auto"

    centerline_s = cumulative_centerline_s(centerline_points)
    centerline_ply_path, centerline_csv_path = write_centerline_outputs(out_dir, centerline_points)

    print(f"切片模式: centerline")
    print(f"中心线来源: {centerline_source}")
    print(f"中心线点数: {len(centerline_points)}")
    print(f"中心线长度: {centerline_s[-1]:.6f}")
    print(f"中心线点云: {centerline_ply_path}")
    print(f"中心线 CSV: {centerline_csv_path}")
    print("开始计算点到中心线的最近投影，这一步在大点云上可能需要一些时间")

    point_s, projection_distance, nearest_segment_id = project_points_to_centerline(
        points, centerline_points
    )
    valid_projection_mask = (
        np.isfinite(point_s)
        & np.isfinite(projection_distance)
        & (nearest_segment_id >= 0)
        & (projection_distance <= args.max_projection_distance)
    )

    projection_summary_path = write_projection_summary(
        out_dir=out_dir,
        total_points=len(points),
        valid_mask=valid_projection_mask,
        point_s=point_s,
        projection_distance=projection_distance,
    )

    valid_count = int(np.count_nonzero(valid_projection_mask))
    if valid_count == 0:
        raise ValueError(
            "没有点满足中心线投影距离阈值。"
            "请检查中心线方向、--centerline_csv 或增大 --max_projection_distance。"
        )

    start_s = args.start_coord
    if start_s is None:
        start_s = float(np.min(point_s[valid_projection_mask]))

    end_s = start_s + args.total_length
    section_count = section_count_from_lengths(args.total_length, args.section_length)

    print(f"有效投影点数: {valid_count} / {len(points)}")
    print(f"最大允许投影距离: {args.max_projection_distance:.6f}")
    print(f"起始里程 s: {start_s:.6f}")
    print(f"结束里程 s: {end_s:.6f}")
    print(f"切片长度: {args.section_length:.6f}")
    print(f"切片数量: {section_count}")
    print(f"输出文件夹: {out_dir}")
    print("-" * 80)

    summary_rows = []

    for section_idx in range(section_count):
        section_no = section_idx + 1
        section_start = start_s + section_idx * args.section_length
        section_end = min(section_start + args.section_length, end_s)

        # 区间采用 [start, end)，最后一个切片采用 [start, end]。
        # 每个有效投影点只会进入一个切片。
        if section_idx == section_count - 1:
            mask = (
                valid_projection_mask
                & (point_s >= section_start)
                & (point_s <= section_end)
            )
        else:
            mask = (
                valid_projection_mask
                & (point_s >= section_start)
                & (point_s < section_end)
            )

        section_points = points[mask]
        point_count = int(section_points.shape[0])

        if point_count > 0:
            section_distances = projection_distance[mask]
            mean_projection_distance = float(np.mean(section_distances))
            max_projection_distance = float(np.max(section_distances))
            mean_projection_text = f"{mean_projection_distance:.6f}"
            max_projection_text = f"{max_projection_distance:.6f}"
        else:
            mean_projection_text = ""
            max_projection_text = ""

        saved = point_count >= args.min_points
        file_path = out_dir / f"{args.prefix}_{section_no:03d}.ply"
        file_for_csv = ""

        if saved:
            section_pcd = make_section_point_cloud(pcd, section_points, mask)
            ok = o3d.io.write_point_cloud(str(file_path), section_pcd, write_ascii=False)
            if not ok:
                raise RuntimeError(f"切片写入失败: {file_path}")
            file_for_csv = str(file_path)

        summary_rows.append(
            {
                "section": f"{args.prefix}_{section_no:03d}",
                "start_s": f"{section_start:.6f}",
                "end_s": f"{section_end:.6f}",
                "point_count": point_count,
                "saved": saved,
                "file": file_for_csv,
                "mean_projection_distance": mean_projection_text,
                "max_projection_distance": max_projection_text,
            }
        )

        print(
            f"{args.prefix}_{section_no:03d}: "
            f"s {section_start:.6f} -> {section_end:.6f}, "
            f"points={point_count}, saved={saved}, "
            f"mean_proj={mean_projection_text}, max_proj={max_projection_text}"
        )

    summary_path = out_dir / "section_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "section",
            "start_s",
            "end_s",
            "point_count",
            "saved",
            "file",
            "mean_projection_distance",
            "max_projection_distance",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 80)
    print(f"切片输出文件夹: {out_dir}")
    print(f"切片统计文件: {summary_path}")
    print(f"中心线点云: {centerline_ply_path}")
    print(f"中心线 CSV: {centerline_csv_path}")
    print(f"投影统计文件: {projection_summary_path}")


def main():
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取点云: {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)

    if points.size == 0:
        raise ValueError(f"点云为空或读取失败: {input_path}")

    print(f"点数: {len(points)}")

    if args.slice_mode == "axis":
        run_axis_mode(args, pcd, points, out_dir)
    else:
        run_centerline_mode(args, pcd, points, out_dir)


if __name__ == "__main__":
    main()

