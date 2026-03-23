import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata

from utils import load_tsv, read_lines


def get_not_in_tissue_coords(coords: np.ndarray, img_xy):
    """找出规则网格中不属于组织区域的坐标及其索引。"""
    img_x, img_y = img_xy
    coords = coords.astype(img_x.dtype)
    coords_list = [list(val) for val in np.array(coords)]

    not_in_tissue_coords = []
    not_in_tissue_index = []

    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            ij_coord = [img_x[i, j], img_y[i, j]]
            if ij_coord not in coords_list:
                not_in_tissue_coords.append(ij_coord)
                not_in_tissue_index.append([int(i), int(j)])

    return not_in_tissue_coords, np.array(not_in_tissue_index)


def get_10x_testset(test_counts, test_coords):
    """将离散 spot 表达插值到规则网格，生成测试输入。"""
    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)

    delta_x = 1
    delta_y = 2

    x_min = min(test_coords[:, 0]) - min(test_coords[:, 0]) % 2
    y_min = min(test_coords[:, 1]) - min(test_coords[:, 1]) % 2

    test_input_x, test_input_y = np.mgrid[
        x_min:max(test_coords[:, 0]) + delta_x:delta_x,
        y_min:max(test_coords[:, 1]) + delta_y:delta_y,
    ]

    # 奇数行做半格偏移，构建六边形排列对应的规则坐标
    for i in range(1, test_input_y.shape[0], 2):
        test_input_y[i] = test_input_y[i] + delta_y / 2

    not_in_tissue_coords, not_in_tissue_xy = get_not_in_tissue_coords(
        test_coords,
        (test_input_x, test_input_y),
    )
    not_in_tissue_x = not_in_tissue_xy.T[0]
    not_in_tissue_y = not_in_tissue_xy.T[1]

    test_set = [None] * test_counts.shape[1]
    for i in range(test_counts.shape[1]):
        # 规则网格上每个点取最近邻 spot 的基因表达值
        test_data = griddata(
            test_coords,
            test_counts[:, i],
            (test_input_x, test_input_y),
            method="nearest",
        )
        test_data[not_in_tissue_x, not_in_tissue_y] = 0
        test_set[i] = test_data

    return np.array(test_set)


def get_hr_sge(gene_set):
    """将基因表达网格上采样到更高分辨率。"""
    _, h, w = gene_set.shape

    gene_set = torch.tensor(gene_set, dtype=torch.float32).unsqueeze(1)
    hr_gene_set = F.interpolate(
        gene_set,
        size=(2 * h - 1, 2 * w - 1),
        mode="bilinear",
        align_corners=False,
    )

    return hr_gene_set.squeeze(1)


def get_10x_position_info(integral_coords):
    """生成高密度网格坐标、邻接关系及组织外点信息。"""
    integral_coords = np.array(integral_coords)

    delta_x = 1
    delta_y = 2

    x_min = min(integral_coords[:, 0]) - min(integral_coords[:, 0]) % 2
    y_min = min(integral_coords[:, 1]) - min(integral_coords[:, 1]) % 2

    y = list(np.arange(y_min, max(integral_coords[:, 1]) + delta_y, delta_y))
    imputed_x, imputed_y = np.mgrid[
        x_min:max(integral_coords[:, 0]) + delta_x:delta_x / 2,
        y_min:y[-1] + delta_y:delta_y / 2,
    ]

    # 调整坐标排列，使其与原始六边形晶格一致
    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] -= delta_y / 4
    for i in range(2, imputed_y.shape[0], 4):
        imputed_y[i:i + 2] += delta_y / 2

    integral_coords = integral_coords.astype(np.float32)
    imputed_barcodes = [
        f"{val[0]}x{val[1]}"
        for val in np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).T
    ]

    imputed_coords = pd.DataFrame(
        np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).astype(np.float32).T,
        columns=["row", "col"],
        index=imputed_barcodes,
    )

    neighbor_matrix = pd.DataFrame(
        np.zeros((imputed_coords.shape[0], imputed_coords.shape[0]), dtype=np.int32),
        columns=imputed_barcodes,
        index=imputed_barcodes,
    )

    row1 = imputed_coords[imputed_coords["row"] == min(imputed_coords["row"])].sort_values("col")
    for i in range(len(row1) - 1):
        if row1["col"].iloc[i + 1] - row1["col"].iloc[i] == delta_y / 2:
            neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
            neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1

    for row in list(np.array(imputed_x).T[0])[:-1]:
        row0 = imputed_coords[imputed_coords["row"] == row].sort_values("col")
        row1 = imputed_coords[imputed_coords["row"] == row + delta_x / 2].sort_values("col")

        for i in range(len(row1) - 1):
            if row1["col"].iloc[i + 1] - row1["col"].iloc[i] == delta_y / 2:
                neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
                neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1

        for i in range(len(row0)):
            flag = 0
            for j in range(len(row1)):
                if abs(
                    imputed_coords.loc[row0.index[i], "col"]
                    - imputed_coords.loc[row1.index[j], "col"]
                ) == delta_y / 4:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = 1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = 1
                    flag += 1
                if flag >= 2:
                    continue

    neighbor_matrix = neighbor_matrix.loc[
        :,
        [f"{val[0]}x{val[1]}" for val in integral_coords],
    ]

    not_in_tissue_coords = []
    for i in range(len(imputed_coords)):
        if imputed_coords.index[i] in neighbor_matrix.columns:
            continue
        if sum(neighbor_matrix.iloc[i]) < 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))

    return [imputed_x, imputed_y, not_in_tissue_coords]


def get_locs(prefix):
    """读取原始 spot 坐标，并映射到离散晶格坐标系。"""
    locs = load_tsv(f"{prefix}locs.csv")
    locs = np.stack([locs["x"] // 200, locs["y"] // 100], axis=-1)
    return locs.round().astype(int)


def get_data(prefix):
    """读取表达矩阵和坐标信息。"""
    gene_names = read_lines(f"{prefix}gene-names.txt")

    cnts = load_tsv(f"{prefix}cnts.csv")
    # 按方差从大到小排序后，再按 gene_names 顺序重排
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]

    locs = get_locs(prefix)
    return cnts, locs


def img2expr(imputed_img, gene_ids, integral_coords, position_info):
    """将高分辨率网格表达重新转换为表达矩阵和坐标表。"""
    imputed_x, imputed_y, not_in_tissue_coords = position_info
    imputed_img = imputed_img.numpy()

    if isinstance(not_in_tissue_coords, np.ndarray):
        not_in_tissue_coords = [list(val) for val in not_in_tissue_coords]

    integral_barcodes = [f"{row[0]}x{row[1]}" for row in integral_coords]
    integral_barcodes = pd.Index(integral_barcodes)

    total_spots = imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords)

    imputed_counts = pd.DataFrame(
        np.zeros((total_spots, imputed_img.shape[0])),
        columns=gene_ids,
    )
    imputed_coords = pd.DataFrame(
        np.zeros((total_spots, 2)),
        columns=["array_row", "array_col"],
    )

    imputed_barcodes = [None] * total_spots
    integral_coords_list = [list(i.astype(np.float32)) for i in np.array(integral_coords)]

    flag = 0
    for i in range(imputed_img.shape[1]):
        for j in range(imputed_img.shape[2]):
            spot_coords = [imputed_x[i, j], imputed_y[i, j]]

            if spot_coords in not_in_tissue_coords:
                continue

            if spot_coords in integral_coords_list:
                imputed_barcodes[flag] = integral_barcodes[integral_coords_list.index(spot_coords)]
            else:
                x_id = str(int(imputed_x[i, j])) if int(imputed_x[i, j]) == imputed_x[i, j] else str(imputed_x[i, j])
                y_id = str(int(imputed_y[i, j])) if int(imputed_y[i, j]) == imputed_y[i, j] else str(imputed_y[i, j])
                imputed_barcodes[flag] = f"{x_id}x{y_id}"

            imputed_counts.iloc[flag, :] = imputed_img[:, i, j]
            imputed_coords.iloc[flag, :] = spot_coords
            flag += 1

    imputed_counts.index = imputed_barcodes
    imputed_coords.index = imputed_barcodes

    return imputed_counts, imputed_coords


def main(prefix):
    """主流程：构建规则网格、上采样并导出高分辨率表达与坐标。"""
    cnts, locs = get_data(prefix)

    test_set = get_10x_testset(cnts, locs)
    hr_test_set = get_hr_sge(test_set)
    hr_locs = get_10x_position_info(locs)

    # 右侧和下侧各补一行/列 0，保持输出尺寸与原实现一致
    hr_test_set = F.pad(hr_test_set, (0, 1, 0, 1, 0, 0), mode="constant", value=0)

    imputed_counts, imputed_coords = img2expr(
        hr_test_set,
        cnts.columns,
        locs,
        hr_locs,
    )

    imputed_counts.to_csv(f"{prefix}HRcnts.csv")

    imputed_coords.columns = ["x", "y"]
    imputed_coords["x"] = imputed_coords["x"] * 200 + 55
    imputed_coords["y"] = imputed_coords["y"] * 100 + 55
    imputed_coords.to_csv(f"{prefix}HRlocs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene expression enhancement script.")
    parser.add_argument(
        "--directory",
        type=str,
        default="./data/MBHD/",
        help="data directory",
    )
    args = parser.parse_args()

    main(args.directory)