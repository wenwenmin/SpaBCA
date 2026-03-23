import os
import pickle

import numpy as np
from skimage.metrics import structural_similarity as ssim

from utils import load_image, read_lines


def normalize_data(data: np.ndarray) -> np.ndarray:
    """按每个基因分别归一化到 [0, 1]。"""
    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)

    # 防止分母为 0
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    normalized_data = (data - min_vals) / denom
    return normalized_data


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

    return np.array(pred_list)


def load_actual_data(data_dir: str, gt_name: str = "genes_3D.pkl") -> np.ndarray:
    """读取真实基因表达数据，并转换为 (G, H, W) 格式。"""
    gt_path = os.path.join(data_dir, gt_name)

    with open(gt_path, "rb") as f:
        actual_data = pickle.load(f)

    # 将无效区域的负值置 0
    actual_data[actual_data < 0] = 0

    # 原始格式通常为 (H, W, G)，转换为 (G, H, W)
    actual_data = np.transpose(actual_data, (2, 0, 1))
    return actual_data


def load_mask(data_dir: str, mask_name: str = "mask.png") -> np.ndarray:
    """读取 mask，并返回二维布尔矩阵。"""
    mask = load_image(os.path.join(data_dir, mask_name)) > 0
    return mask[:, :, 0]


def compute_ssim_per_gene(
    predicted_data: np.ndarray,
    actual_data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """计算每个基因在有效区域约束下的 SSIM。"""
    # 对齐空间尺寸
    predicted_data = predicted_data[:, :actual_data.shape[1], :actual_data.shape[2]]

    # 归一化
    predicted_data = normalize_data(predicted_data)
    actual_data = normalize_data(actual_data)

    # 无效区域置 0，保持与原始脚本逻辑一致
    predicted_data = np.where(mask, predicted_data, 0)
    actual_data = np.where(mask, actual_data, 0)

    ssim_list = []
    for i in range(predicted_data.shape[0]):
        ssim_index = ssim(actual_data[i], predicted_data[i], data_range=1.0)
        ssim_list.append(ssim_index)

    return np.array(ssim_list)


def save_ssim_results(
    ssim_per_gene: np.ndarray,
    output_dir: str,
    output_name: str = "ours.txt",
) -> None:
    """保存每个基因的 SSIM 结果。"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    np.savetxt(output_path, ssim_per_gene)


if __name__ == "__main__":
    data_dir = "./data/MBHD"
    ssim_dir = os.path.join(data_dir, "SSIM")

    predicted_data = load_predicted_data(data_dir, pred_subfolder="cnts-super")
    actual_data = load_actual_data(data_dir, gt_name="genes_3D.pkl")
    mask = load_mask(data_dir, mask_name="mask.png")

    ssim_per_gene = compute_ssim_per_gene(predicted_data, actual_data, mask)

    print("Mean SSIM:", np.nanmean(ssim_per_gene))
    print("Var SSIM:", np.nanvar(ssim_per_gene))

    save_ssim_results(
        ssim_per_gene=ssim_per_gene,
        output_dir=ssim_dir,
        output_name="ours.txt",
    )