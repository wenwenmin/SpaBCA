import argparse
import json
import os
import pickle
import tarfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile as tf
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from shapely.geometry import Polygon
from skimage.draw import polygon2mask

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
Image.MAX_IMAGE_PIXELS = 900000000


def load_10x_matrix_from_tar_gz(tar_file_path: str):
    """解压 10x 的 tar.gz 文件，并返回解压目录。"""
    tar_path = Path(tar_file_path)

    if not tar_path.exists():
        print(f"Error: file not found: {tar_file_path}")
        return None

    extract_dir = tar_path.parent / tar_path.stem.split(".")[0]
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    return extract_dir


def get_xenium_pixel_size(file_path: str):
    """从 Xenium 配置文件中读取 pixel_size。"""
    xenium_file = Path(file_path)

    if not xenium_file.exists():
        print(f"Error: file not found: {file_path}")
        return None

    try:
        with open(xenium_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pixel_size = data.get("pixel_size")
        if pixel_size is None:
            print(f"Warning: 'pixel_size' not found in {file_path}")
            return None

        return float(pixel_size)

    except json.JSONDecodeError:
        print(f"Error: invalid JSON file: {file_path}")
        return None
    except Exception as e:
        print(f"Error while reading {file_path}: {e}")
        return None


def cal_area_2poly(data1, data2):
    """计算两个多边形的交集面积。"""
    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        return 0.0

    return poly1.intersection(poly2).area


def unique_pixels(coord_list):
    """对二维坐标列表去重，保持原顺序。"""
    unique_coords = set()
    filtered_list = []

    for coord in coord_list:
        coord_tuple = tuple(coord)
        if coord_tuple not in unique_coords:
            unique_coords.add(coord_tuple)
            filtered_list.append(coord)

    return filtered_list


def pad_to_multiple_of_224(img: np.ndarray, fill_value: int = 255) -> np.ndarray:
    """将图像补齐到 224 的整数倍。"""
    h, w = img.shape[:2]
    new_h = ((h + 223) // 224) * 224
    new_w = ((w + 223) // 224) * 224

    pad_h = new_h - h
    pad_w = new_w - w

    img_padded = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )
    return img_padded


def get_disk_mask(radius, boundary_width=None):
    """生成圆形掩码；如果提供 boundary_width，则生成环形掩码。"""
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    distsq = (locs ** 2).sum(axis=-1)

    isin = distsq <= radius ** 2
    if boundary_width is not None:
        isin &= distsq >= (radius - boundary_width) ** 2

    return isin


def get_locs(data_dir, rescale_factor):
    """读取 locs.csv，并按比例缩放到网格坐标。"""
    locs = pd.read_csv(os.path.join(data_dir, "locs.csv"), header=0, index_col=0)
    locs = np.stack([locs["x"], locs["y"]], axis=-1).astype(float)
    locs /= rescale_factor
    return locs.round().astype(int)


def get_patches_flat(img, locs, mask):
    """提取每个位置对应的局部 patch。"""
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape - center], axis=-1)

    x_list = []
    for s in locs:
        patch = img[
            s[1] + r[0][0]: s[1] + r[0][1],
            s[0] + r[1][0]: s[0] + r[1][1],
        ]
        x = patch if mask.all() else patch[mask]
        x_list.append(x)

    return np.stack(x_list)


def scale_img(raw_image_path, output_image_path, align_path, scale, shape):
    """读取原图、缩放、配准并补齐输出。"""
    try:
        he = cv2.imread(raw_image_path)
        if he is None:
            print(f"Error: failed to load image: {raw_image_path}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    new_width = int(he.shape[1] * scale)
    new_height = int(he.shape[0] * scale)
    he_small = cv2.resize(he, (new_width, new_height), interpolation=cv2.INTER_AREA)

    m = np.loadtxt(align_path, delimiter=",", dtype=float)
    m_new = m.copy()
    m_new[0, 2] *= scale
    m_new[1, 2] *= scale

    dst_width = int(shape[1] * scale)
    dst_height = int(shape[0] * scale)

    img = cv2.warpPerspective(he_small, m_new, dsize=(dst_width, dst_height))
    img = img.astype(np.uint8)
    img = pad_to_multiple_of_224(img)

    # 将纯黑背景替换为浅灰色
    black_mask = np.all(img == [0, 0, 0], axis=-1)
    img[black_mask] = [235, 235, 235]

    cv2.imwrite(output_image_path, img)
    print(f"Saved padded image to: {output_image_path}, shape: {img.shape}")
    return img


def create_gene_matrix(
    folder_path,
    img_shape: Tuple[int, int, int],
    gene_matrix_path: str,
    gene_names_path: str,
    superpixel_size: int = 16,
):
    """根据细胞轮廓和表达矩阵构建 3D gene matrix。"""
    mtx_dir = os.path.join(folder_path, "cell_feature_matrix")

    if os.path.isdir(mtx_dir):
        print(f"[create_gene_matrix] use MTX dir: {mtx_dir}")
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols")
    else:
        tar_path = os.path.join(folder_path, "cell_feature_matrix.tar.gz")
        if os.path.exists(tar_path):
            print(f"[create_gene_matrix] extract from: {tar_path}")
            extracted_dir = load_10x_matrix_from_tar_gz(tar_path)
            if extracted_dir is None:
                raise FileNotFoundError(f"Failed to extract: {tar_path}")
            mtx_dir = os.path.join(extracted_dir, "cell_feature_matrix")
            adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols")
        else:
            raise FileNotFoundError(
                f"Cannot find 'cell_feature_matrix' or 'cell_feature_matrix.tar.gz' under {folder_path}"
            )

    adata.var_names_make_unique()

    pd.Series(adata.var_names).to_csv(gene_names_path, index=False, header=False)
    print(f"Saved gene names to: {gene_names_path}")

    counts = np.asarray(adata.X.todense())
    boundaries = pd.read_csv(
        os.path.join(folder_path, "cell_boundaries.csv.gz"),
        sep=",",
        index_col=0,
    )
    boundaries *= 2
    boundaries = boundaries.round().astype(int).reset_index()

    # 将原始 cell_id 重映射为连续编号，便于后续按行索引
    index_map = {}
    current_number = 1
    for value in boundaries.iloc[:, 0]:
        if value not in index_map:
            index_map[value] = current_number
            current_number += 1

    boundaries.iloc[:, 0] = boundaries.iloc[:, 0].map(index_map)
    boundaries = boundaries.set_index("cell_id")

    cells = pd.read_csv(
        os.path.join(folder_path, "cells.csv.gz"),
        sep=",",
        index_col=0,
    )

    gene_nums = adata.shape[1]
    cnts = np.zeros(
        (img_shape[0] // superpixel_size, img_shape[1] // superpixel_size, gene_nums),
        dtype=np.float32,
    )

    grid_size = superpixel_size

    for c in range(len(cells)):
        data1 = np.array(boundaries.loc[c + 1])

        coarse_pixels = []
        for point in data1:
            coarse_pixels.append(point // grid_size * grid_size)
        coarse_pixels = np.array(unique_pixels(coarse_pixels))

        min_x, min_y = (np.min(data1, axis=0) // grid_size * grid_size).astype(int)
        max_x, max_y = (np.max(data1, axis=0) // grid_size * grid_size).astype(int)

        all_pixels = []
        for x in range(min_x, max_x + grid_size, grid_size):
            for y in range(min_y, max_y + grid_size, grid_size):
                all_pixels.append([x, y])
        all_pixels = np.array(all_pixels)

        mask = polygon2mask(
            (max_y - min_y + grid_size, max_x - min_x + grid_size),
            (data1 - [min_x, min_y]) / grid_size,
        )

        valid_pixels = all_pixels[
            mask[
                all_pixels[:, 1] // grid_size - min_y // grid_size,
                all_pixels[:, 0] // grid_size - min_x // grid_size,
            ]
        ]

        pixels = np.unique(np.vstack((coarse_pixels, valid_pixels)), axis=0)

        # 计算细胞与每个 superpixel 的交集面积，再按比例分配表达量
        areas = []
        for p in pixels:
            x, y = p[0], p[1]
            data2 = np.array(
                [[x, y], [x, y + grid_size], [x + grid_size, y + grid_size], [x + grid_size, y]]
            )
            inter_area = cal_area_2poly(data1, data2)
            areas.append(inter_area)

        total_area = sum(areas)
        if total_area == 0:
            continue

        for i, p in enumerate(pixels):
            x, y = p[0], p[1]
            ratio = areas[i] / total_area
            gene = counts[c, :] * ratio

            w, h = int(x // grid_size), int(y // grid_size)
            if h >= cnts.shape[0] or w >= cnts.shape[1]:
                continue

            cnts[h, w, :] += gene

    with open(gene_matrix_path, "wb") as f:
        pickle.dump(cnts, f)

    print(f"Saved 3D gene matrix to: {gene_matrix_path}, shape: {cnts.shape}")
    return cnts


def get_mask(he, data_dir, block_size=16):
    """根据 H&E 图像生成组织区域 mask。"""
    h, w, _ = he.shape
    h_new, w_new = h // block_size, w // block_size

    mask = np.zeros((h_new, w_new), dtype=bool)

    for i in range(h_new):
        for j in range(w_new):
            block = he[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size,
                1,
            ]
            mask[i, j] = np.mean(block) <= 230

    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    plt.imsave(os.path.join(data_dir, "mask.png"), mask, cmap="gray")
    return mask


def get_pseudo_visium_locs(
    he,
    mask_raw,
    data_dir,
    distance: int = 100,
    pixel_size: float = 0.5,
    superpixel_size: int = 16,
    offset: int = 3,
):
    """生成 pseudo-Visium spot 坐标，并根据组织区域筛选。"""
    w_px, h_px = he.shape[1], he.shape[0]
    w_um, h_um = w_px * pixel_size, h_px * pixel_size

    locations_um = []
    for i in range(0, int(w_um - 27.5), distance):
        k = 0 if (i / distance) % 2 == 0 else 50
        for j in range(0, int(h_um - 27.5 - (127.5 - k)), distance):
            x = i + 27.5
            y = j + 27.5 + k
            locations_um.append([x, y])

    if not locations_um:
        print("Warning: no locations were generated.")
        return

    locations_um_np = np.array(locations_um, dtype=np.float32)
    locations_px = (locations_um_np / pixel_size).astype(np.int32)

    df = pd.DataFrame(locations_px, columns=["x", "y"])
    print(f"Generated {len(df)} initial locations.")

    mask = mask_raw > 0
    h_mask, w_mask = mask.shape

    # 将像素坐标映射到 mask 网格坐标
    df["mask_col"] = df["x"] // superpixel_size - offset
    df["mask_row"] = df["y"] // superpixel_size - offset

    df_filtered = df[
        (df["mask_row"] >= 0) & (df["mask_row"] < h_mask) &
        (df["mask_col"] >= 0) & (df["mask_col"] < w_mask)
    ].copy()

    mask_indices = df_filtered[["mask_row", "mask_col"]].values
    mask_values = mask[mask_indices[:, 0], mask_indices[:, 1]]
    df_final = df_filtered[mask_values].copy()

    df_final["spots"] = range(len(df_final))
    df_final[["spots", "x", "y"]].to_csv(
        os.path.join(data_dir, "locs.csv"),
        index=False,
        float_format="%.2f",
    )


def get_pseudo_visium_cnts(genes, radius, data_dir, superpixel_size):
    """统计 pseudo-Visium spot 对应局部区域的基因表达量。"""
    genes = genes.copy()
    genes[genes < 0] = 0

    mask = get_disk_mask(radius)
    locs = get_locs(data_dir, superpixel_size)
    x = get_patches_flat(genes, locs, mask)
    cnts = np.sum(x, axis=1)

    with open(os.path.join(data_dir, "gene-names.txt"), "r", encoding="utf-8") as file:
        gene_names = [line.strip() for line in file]

    cnts_df = pd.DataFrame(data=cnts, columns=gene_names)
    cnts_df.to_csv(os.path.join(data_dir, "cnts.csv"))


def process_xenium(
    data_dir: str,
    sample_subfolder: str,
    align_name: str,
    raw_image_name: str,
    output_image_name: str,
    gene_names_name: str,
    ground_truth_name: str,
    scale_pix_size: float = 0.5,
    superpixel_size: int = 16,
    pseudo_radius: int = 55,
):
    """Xenium 数据预处理主流程。"""
    os.makedirs(data_dir, exist_ok=True)

    folder_path = os.path.join(data_dir, sample_subfolder)
    raw_image_path = os.path.join(data_dir, raw_image_name)
    output_image_path = os.path.join(data_dir, output_image_name)
    gene_names_path = os.path.join(data_dir, gene_names_name)
    gene_matrix_path = os.path.join(data_dir, ground_truth_name)
    raw_pix_size_path = os.path.join(folder_path, "experiment.xenium")
    align_path = os.path.join(data_dir, align_name)

    print("--- Start preprocessing Xenium data ---")

    raw_pix_size = get_xenium_pixel_size(raw_pix_size_path)
    if raw_pix_size is None:
        print("Error: failed to get raw pixel size.")
        return

    with open(os.path.join(data_dir, "pixel-size-raw.txt"), "w", encoding="utf-8") as f:
        f.write(str(raw_pix_size))
    with open(os.path.join(data_dir, "pixel-size.txt"), "w", encoding="utf-8") as f:
        f.write(str(scale_pix_size))

    scale = raw_pix_size / scale_pix_size
    print(f"Original pixel size: {raw_pix_size:.4f} μm, scaling factor: {scale:.4f}")

    with tf.TiffFile(os.path.join(folder_path, "morphology.ome.tif")) as tif:
        morphology_ome_shape = tif.series[0].shape[1:]

    img_padded = scale_img(
        raw_image_path=raw_image_path,
        output_image_path=output_image_path,
        align_path=align_path,
        scale=scale,
        shape=morphology_ome_shape,
    )
    if img_padded is None:
        print("Error: failed to generate aligned image.")
        return

    genes_3d = create_gene_matrix(
        folder_path=folder_path,
        img_shape=img_padded.shape,
        gene_matrix_path=gene_matrix_path,
        gene_names_path=gene_names_path,
        superpixel_size=superpixel_size,
    )

    mask = get_mask(img_padded, data_dir, block_size=superpixel_size)

    get_pseudo_visium_locs(
        he=img_padded,
        mask_raw=mask,
        data_dir=data_dir,
        superpixel_size=superpixel_size,
    )

    get_pseudo_visium_cnts(
        genes=genes_3d,
        radius=pseudo_radius / superpixel_size,
        data_dir=data_dir,
        superpixel_size=superpixel_size,
    )

    with open(os.path.join(data_dir, "radius.txt"), "w", encoding="utf-8") as f:
        f.write(str(pseudo_radius))

    print("--- All processing steps completed successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Xenium preprocessing.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/sample",
        help="base directory containing all data",
    )
    parser.add_argument(
        "--sample_subfolder",
        type=str,
        default="outs",
        help="Xenium output subfolder name",
    )
    parser.add_argument(
        "--align_name",
        type=str,
        default="alignment.csv",
        help="alignment matrix file name",
    )
    parser.add_argument(
        "--raw_image_name",
        type=str,
        default="raw_image.ome.tif",
        help="raw image file name",
    )
    parser.add_argument(
        "--output_image_name",
        type=str,
        default="he.jpg",
        help="output image file name",
    )
    parser.add_argument(
        "--gene_names_name",
        type=str,
        default="gene-names.txt",
        help="gene names output file name",
    )
    parser.add_argument(
        "--ground_truth_name",
        type=str,
        default="genes_3D.pkl",
        help="3D gene matrix output file name",
    )
    parser.add_argument(
        "--scale_pix_size",
        type=float,
        default=0.5,
        help="target pixel size in microns",
    )
    parser.add_argument(
        "--superpixel_size",
        type=int,
        default=16,
        help="superpixel size in pixels",
    )
    parser.add_argument(
        "--pseudo_radius",
        type=int,
        default=55,
        help="pseudo Visium radius in pixels",
    )

    args = parser.parse_args()

    process_xenium(
        data_dir=args.data_dir,
        sample_subfolder=args.sample_subfolder,
        align_name=args.align_name,
        raw_image_name=args.raw_image_name,
        output_image_name=args.output_image_name,
        gene_names_name=args.gene_names_name,
        ground_truth_name=args.ground_truth_name,
        scale_pix_size=args.scale_pix_size,
        superpixel_size=args.superpixel_size,
        pseudo_radius=args.pseudo_radius,
    )