import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata

from utils import read_lines, load_tsv

def get_not_in_tissue_coords(coords: np.ndarray, grid_xy):
    """找出规则网格中不属于组织的点及其索引"""
    grid_x, grid_y = grid_xy
    coords = coords.astype(grid_x.dtype)
    coords_list = [list(val) for val in coords]

    not_in_tissue_coords = []
    not_in_tissue_index = []

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            coord = [grid_x[i, j], grid_y[i, j]]
            if coord not in coords_list:
                not_in_tissue_coords.append(coord)
                not_in_tissue_index.append([i, j])

    return not_in_tissue_coords, np.array(not_in_tissue_index)


def build_test_grid(test_counts, test_coords):
    """将离散 spot 表达插值到规则网格"""
    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)

    delta_x, delta_y = 1, 2

    x_min = min(test_coords[:, 0]) - min(test_coords[:, 0]) % 2
    y_min = min(test_coords[:, 1]) - min(test_coords[:, 1]) % 2

    grid_x, grid_y = np.mgrid[
        x_min:max(test_coords[:, 0]) + delta_x:delta_x,
        y_min:max(test_coords[:, 1]) + delta_y:delta_y,
    ]

    # 构造六边形排列偏移
    for i in range(1, grid_y.shape[0], 2):
        grid_y[i] += delta_y / 2

    not_in_tissue_coords, not_in_tissue_idx = get_not_in_tissue_coords(
        test_coords, (grid_x, grid_y)
    )

    test_set = []
    for i in range(test_counts.shape[1]):
        data = griddata(
            test_coords,
            test_counts[:, i],
            (grid_x, grid_y),
            method="nearest",
        )
        data[not_in_tissue_idx[:, 0], not_in_tissue_idx[:, 1]] = 0
        test_set.append(data)

    return np.array(test_set)


def upsample_gene_expression(gene_set):
    """将表达矩阵上采样到更高分辨率"""
    _, h, w = gene_set.shape

    gene_set = torch.tensor(gene_set, dtype=torch.float32).unsqueeze(1)

    hr_gene = F.interpolate(
        gene_set,
        size=(2 * h - 1, 2 * w - 1),
        mode="bilinear",
        align_corners=False,
    )

    return hr_gene.squeeze(1)


def build_hr_position_info(integral_coords):
    """生成高分辨率网格坐标和组织外点"""
    integral_coords = np.array(integral_coords)

    delta_x, delta_y = 1, 2

    x_min = min(integral_coords[:, 0]) - min(integral_coords[:, 0]) % 2
    y_min = min(integral_coords[:, 1]) - min(integral_coords[:, 1]) % 2

    grid_x, grid_y = np.mgrid[
        x_min:max(integral_coords[:, 0]) + delta_x:delta_x / 2,
        y_min:max(integral_coords[:, 1]) + delta_y:delta_y / 2,
    ]

    # 调整六边形排列
    for i in range(1, grid_y.shape[0], 2):
        grid_y[i] -= delta_y / 4
    for i in range(2, grid_y.shape[0], 4):
        grid_y[i:i + 2] += delta_y / 2

    integral_coords = integral_coords.astype(np.float32)

    not_in_tissue_coords = []
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            coord = [grid_x[i, j], grid_y[i, j]]
            if list(coord) not in integral_coords.tolist():
                not_in_tissue_coords.append(coord)

    return [grid_x, grid_y, not_in_tissue_coords]


def get_locs(prefix):
    """读取并缩放 spot 坐标"""
    locs = load_tsv(f"{prefix}locs.csv")
    locs = np.stack([locs["x"] // 200, locs["y"] // 100], axis=-1)
    return locs.round().astype(int)


def load_data(prefix):
    """读取表达矩阵和坐标"""
    gene_names = read_lines(f"{prefix}gene-names.txt")

    cnts = load_tsv(f"{prefix}cnts.csv")
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]

    locs = get_locs(prefix)
    return cnts, locs


def grid_to_expression(imputed_img, gene_ids, integral_coords, position_info):
    """将高分辨率网格转回表达矩阵"""
    grid_x, grid_y, not_in_tissue_coords = position_info
    imputed_img = imputed_img.numpy()

    total_spots = imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords)

    imputed_counts = pd.DataFrame(
        np.zeros((total_spots, imputed_img.shape[0])),
        columns=gene_ids,
    )

    imputed_coords = pd.DataFrame(
        np.zeros((total_spots, 2)),
        columns=["x", "y"],
    )

    idx = 0
    for i in range(imputed_img.shape[1]):
        for j in range(imputed_img.shape[2]):
            coord = [grid_x[i, j], grid_y[i, j]]

            if coord in not_in_tissue_coords:
                continue

            imputed_counts.iloc[idx, :] = imputed_img[:, i, j]
            imputed_coords.iloc[idx, :] = coord
            idx += 1

    return imputed_counts, imputed_coords


def main(data_dir):
    """主函数：生成高分辨率表达"""
    cnts, locs = load_data(data_dir)

    test_set = build_test_grid(cnts, locs)
    hr_test_set = upsample_gene_expression(test_set)

    position_info = build_hr_position_info(locs)

    # padding 保持尺寸一致
    hr_test_set = F.pad(hr_test_set, (0, 1, 0, 1), value=0)

    imputed_counts, imputed_coords = grid_to_expression(
        hr_test_set,
        cnts.columns,
        locs,
        position_info,
    )

    imputed_counts.to_csv(f"{data_dir}HRcnts.csv")

    # 坐标恢复到原像素空间
    imputed_coords["x"] = imputed_coords["x"] * 200 + 55
    imputed_coords["y"] = imputed_coords["y"] * 100 + 55
    imputed_coords.to_csv(f"{data_dir}HRlocs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene expression super-resolution")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/MBHD/",
        help="data directory",
    )

    args = parser.parse_args()

    main(args.data_dir)