import os
import pickle

import numpy as np
import pandas as pd

from utils import load_image, read_lines
from Preprocessing_Visium_HD import get_disk_mask


def load_predicted_data(data_dir: str, pred_subfolder: str = "cnts-super") -> np.ndarray:
    """读取所有基因的预测结果，并堆叠为 (G, H, W) 格式。"""
    pred_list = []
    folder_path = os.path.join(data_dir, pred_subfolder)

    gene_names = read_lines(os.path.join(data_dir, "gene-names.txt"))

    for gene_name in gene_names:
        file_path = os.path.join(folder_path, f"{gene_name}.pickle")
        with open(file_path, "rb") as f:
            matrix = pickle.load(f)
        pred_list.append(matrix)

    return np.array(pred_list, dtype=np.float32)


def load_actual_data(data_dir: str, gt_name: str = "genes_3D.pkl") -> np.ndarray:
    """读取真实基因表达数据，并转换为 (G, H, W) 格式。"""
    gt_path = os.path.join(data_dir, gt_name)

    with open(gt_path, "rb") as f:
        actual_data = pickle.load(f)

    actual_data = np.array(actual_data, dtype=np.float32)
    actual_data[actual_data < 0] = 0

    # 原始格式通常为 (H, W, G)，转换为 (G, H, W)
    actual_data = np.transpose(actual_data, (2, 0, 1))
    return actual_data


def load_spot_locations(
    data_dir: str,
    superpixel_size: int = 16,
    locs_name: str = "locs.csv",
) -> np.ndarray:
    """读取 locs.csv，并转换到 SR 网格坐标系，返回 (S, 2) 的 (y, x)。"""
    locs_df = pd.read_csv(os.path.join(data_dir, locs_name), header=0, index_col=0)

    locs_x = (locs_df["x"].to_numpy(dtype=np.float32) / superpixel_size).round().astype(int)
    locs_y = (locs_df["y"].to_numpy(dtype=np.float32) / superpixel_size).round().astype(int)

    return np.stack([locs_y, locs_x], axis=1)


def load_radius_mask(
    data_dir: str,
    superpixel_size: int = 16,
    radius_name: str = "radius.txt",
) -> np.ndarray:
    """读取 radius.txt，并生成 SR 网格上的圆形掩码。"""
    with open(os.path.join(data_dir, radius_name), "r", encoding="utf-8") as f:
        pseudo_radius_px = float(f.read().strip())

    radius_sr = pseudo_radius_px / superpixel_size
    return get_disk_mask(radius_sr)


def load_tissue_mask(data_dir: str, mask_name: str = "mask.png") -> np.ndarray:
    """读取组织区域 mask，并返回二维布尔矩阵。"""
    tissue_mask = load_image(os.path.join(data_dir, mask_name)) > 0
    return tissue_mask[:, :, 0]


def filter_spots_by_tissue(locs_yx: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    """只保留位于组织区域内的 spot 中心。"""
    in_bounds = (
        (locs_yx[:, 0] >= 0)
        & (locs_yx[:, 0] < tissue_mask.shape[0])
        & (locs_yx[:, 1] >= 0)
        & (locs_yx[:, 1] < tissue_mask.shape[1])
    )
    locs_yx = locs_yx[in_bounds]

    center_in_tissue = tissue_mask[locs_yx[:, 0], locs_yx[:, 1]]
    return locs_yx[center_in_tissue]


def safe_extract_patch(arr2d: np.ndarray, cy: int, cx: int, mask: np.ndarray) -> np.ndarray:
    """从二维图中截取以 (cy, cx) 为中心的 patch，越界区域补 0。"""
    patch_h, patch_w = mask.shape
    half_h, half_w = patch_h // 2, patch_w // 2

    y0, y1 = cy - half_h, cy - half_h + patch_h
    x0, x1 = cx - half_w, cx - half_w + patch_w

    patch = np.zeros((patch_h, patch_w), dtype=arr2d.dtype)

    src_y0 = max(0, y0)
    src_y1 = min(arr2d.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(arr2d.shape[1], x1)

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    patch[dst_y0:dst_y1, dst_x0:dst_x1] = arr2d[src_y0:src_y1, src_x0:src_x1]
    return patch


def aggregate_spot_sums(
    gene_map_hw: np.ndarray,
    locs_yx: np.ndarray,
    disk_mask: np.ndarray,
) -> np.ndarray:
    """对单个基因图在每个 spot 的圆形区域内求和，得到 spot-level 表达。"""
    spot_sums = np.zeros((locs_yx.shape[0],), dtype=np.float64)

    for i, (y, x) in enumerate(locs_yx):
        patch = safe_extract_patch(gene_map_hw, int(y), int(x), disk_mask)
        spot_sums[i] = patch[disk_mask].sum()

    return spot_sums


def pearsonr_safe(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """安全计算 Pearson 相关系数，异常情况返回 NaN。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    valid_mask = np.isfinite(a) & np.isfinite(b)
    if valid_mask.sum() < 2:
        return np.nan

    a = a[valid_mask]
    b = b[valid_mask]

    if a.std() < eps or b.std() < eps:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def compute_pcc_per_gene(
    predicted_data: np.ndarray,
    actual_data: np.ndarray,
    locs_yx: np.ndarray,
    disk_mask: np.ndarray,
    use_log1p: bool = False,
) -> np.ndarray:
    """计算每个基因的 spot-level PCC。"""
    # 对齐空间尺寸
    height = min(actual_data.shape[1], predicted_data.shape[1])
    width = min(actual_data.shape[2], predicted_data.shape[2])

    actual_data = actual_data[:, :height, :width]
    predicted_data = predicted_data[:, :height, :width]

    pcc_list = []
    for gene_idx in range(actual_data.shape[0]):
        gt_spot = aggregate_spot_sums(actual_data[gene_idx], locs_yx, disk_mask)
        pred_spot = aggregate_spot_sums(predicted_data[gene_idx], locs_yx, disk_mask)

        if use_log1p:
            gt_spot = np.log1p(gt_spot)
            pred_spot = np.log1p(pred_spot)

        pcc_list.append(pearsonr_safe(gt_spot, pred_spot))

    return np.array(pcc_list, dtype=np.float64)


def save_pcc_results(
    pcc_per_gene: np.ndarray,
    output_dir: str,
    output_name: str = "ours.txt",
) -> None:
    """保存每个基因的 PCC 结果。"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    np.savetxt(output_path, pcc_per_gene)


if __name__ == "__main__":
    data_dir = "./data/MBHD"
    pcc_dir = os.path.join(data_dir, "PCC_spot")

    predicted_data = load_predicted_data(data_dir, pred_subfolder="cnts-super")
    actual_data = load_actual_data(data_dir, gt_name="genes_3D.pkl")

    locs_yx = load_spot_locations(
        data_dir=data_dir,
        superpixel_size=16,
        locs_name="locs.csv",
    )
    disk_mask = load_radius_mask(
        data_dir=data_dir,
        superpixel_size=16,
        radius_name="radius.txt",
    )
    tissue_mask = load_tissue_mask(data_dir, mask_name="mask.png")
    locs_yx = filter_spots_by_tissue(locs_yx, tissue_mask)

    pcc_per_gene = compute_pcc_per_gene(
        predicted_data=predicted_data,
        actual_data=actual_data,
        locs_yx=locs_yx,
        disk_mask=disk_mask,
        use_log1p=True,
    )

    print("Mean PCC:", np.nanmean(pcc_per_gene))
    print("Var PCC:", np.nanvar(pcc_per_gene))

    save_pcc_results(
        pcc_per_gene=pcc_per_gene,
        output_dir=pcc_dir,
        output_name="ours.txt",
    )