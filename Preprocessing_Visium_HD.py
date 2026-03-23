import argparse
import os
import pickle
from typing import Tuple

import bin2cell as b2c
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.transform import rescale

Image.MAX_IMAGE_PIXELS = 900000000


def pad_to_multiple_of_224(img: np.ndarray, fill_value: int = 255) -> np.ndarray:
    """Pad image to a multiple of 224."""
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
    """Generate a disk mask or ring mask."""
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


def create_gene_matrix(
    adata: sc.AnnData,
    img_shape: Tuple[int, int, int],
    gene_matrix_path: str,
    gene_names_path: str,
    superpixel_size: int = 16,
    num_top_genes: int = 1000,
):
    """根据高变基因构建 3D gene matrix。"""
    h, w, _ = img_shape
    grid_h = h // superpixel_size
    grid_w = w // superpixel_size

    # 只保留高变基因，降低后续处理维度
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=num_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    print(f"Selected {adata.n_vars} highly variable genes.")

    pd.Series(adata.var_names).to_csv(gene_names_path, index=False, header=False)
    print(f"Saved gene names to: {gene_names_path}")

    genes_3d = np.zeros((grid_h, grid_w, adata.n_vars), dtype=np.float32)
    counts = adata.X.toarray()

    pxl_row = adata.obs["pxl_row_in_fullres"]
    pxl_col = adata.obs["pxl_col_in_fullres"]

    # 将 spot 表达映射到规则网格
    for idx in range(adata.n_obs):
        row_idx = (pxl_row.iloc[idx] - (superpixel_size // 2)) // superpixel_size
        col_idx = (pxl_col.iloc[idx] - (superpixel_size // 2)) // superpixel_size

        if 0 <= row_idx < grid_h and 0 <= col_idx < grid_w:
            genes_3d[row_idx, col_idx, :] = -1 if np.all(counts[idx, :] == 0) else counts[idx, :]

    with open(gene_matrix_path, "wb") as f:
        pickle.dump(genes_3d, f)

    print(f"Saved 3D gene matrix to: {gene_matrix_path}, shape: {genes_3d.shape}")
    return genes_3d


def get_mask(genes, data_dir):
    """根据 gene matrix 生成有效区域 mask。"""
    mask = np.any(genes, axis=2).astype(np.uint8) * 255

    # 对有效区域做一次膨胀，扩大边界覆盖范围
    bool_mask = mask == 255
    dilated_mask = binary_dilation(bool_mask)
    new_mask = dilated_mask.astype(np.uint8) * 255

    plt.imsave(os.path.join(data_dir, "mask.png"), new_mask, cmap="gray")
    return new_mask


def get_pseudo_visium_locs(
    he,
    mask_raw,
    data_dir,
    distance: int = 100,
    pixel_size: float = 0.5,
    superpixel_size: int = 16,
    offset: int = 3,
):
    """生成 pseudo-Visium spot 坐标，并根据 mask 进行筛选。"""
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
        print("Warning: no pseudo spots were generated.")
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
    patches = get_patches_flat(genes, locs, mask)
    cnts = np.sum(patches, axis=1)

    with open(os.path.join(data_dir, "gene-names.txt"), "r", encoding="utf-8") as file:
        gene_names = [line.strip() for line in file]

    cnts_df = pd.DataFrame(data=cnts, columns=gene_names)
    cnts_df.to_csv(os.path.join(data_dir, "cnts.csv"))


def process_visium_hd(
    data_dir: str,
    sample_subfolder: str,
    raw_image_name: str,
    output_image_name: str,
    gene_names_name: str,
    ground_truth_name: str,
    scale_pix_size: float = 0.5,
    superpixel_size: int = 16,
    num_top_genes: int = 1000,
    pseudo_radius: int = 55,
):
    """Visium HD 数据预处理主流程。"""
    os.makedirs(data_dir, exist_ok=True)

    folder_path = os.path.join(data_dir, sample_subfolder)
    raw_image_path = os.path.join(data_dir, raw_image_name)
    output_image_path = os.path.join(data_dir, output_image_name)
    gene_names_path = os.path.join(data_dir, gene_names_name)
    gene_matrix_path = os.path.join(data_dir, ground_truth_name)

    print("--- Start preprocessing Visium HD data ---")

    try:
        adata = b2c.read_visium(folder_path)
    except Exception as e:
        print(f"Error loading Visium data: {e}")
        return

    # 过滤掉异常 spot
    adata = adata[(adata.obs >= 0).all(axis=1)].copy()

    visium_key = next(iter(adata.uns["spatial"]))
    raw_pix_size = adata.uns["spatial"][visium_key]["scalefactors"]["microns_per_pixel"]

    with open(os.path.join(data_dir, "pixel-size-raw.txt"), "w", encoding="utf-8") as f:
        f.write(str(raw_pix_size))
    with open(os.path.join(data_dir, "pixel-size.txt"), "w", encoding="utf-8") as f:
        f.write(str(scale_pix_size))

    scale = raw_pix_size / scale_pix_size
    print(f"Original pixel size: {raw_pix_size:.4f} μm, scaling factor: {scale:.4f}")

    # 将空间坐标缩放到目标分辨率
    adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]] = adata.obsm["spatial"] * scale
    adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]] = (
        adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]]
        .round()
        .astype(int)
    )

    try:
        img = np.array(Image.open(raw_image_path))
    except FileNotFoundError:
        print(f"Error: image file not found: {raw_image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 将原图缩放到目标像素尺寸
    img = rescale(img, [scale, scale, 1], preserve_range=True).astype(np.uint8)

    # 补齐到 224 的整数倍，方便下游模型处理
    img_padded = pad_to_multiple_of_224(img)
    Image.fromarray(img_padded).save(output_image_path)
    print(f"Saved padded image to: {output_image_path}, shape: {img_padded.shape}")

    genes_3d = create_gene_matrix(
        adata=adata,
        img_shape=img_padded.shape,
        gene_matrix_path=gene_matrix_path,
        gene_names_path=gene_names_path,
        superpixel_size=superpixel_size,
        num_top_genes=num_top_genes,
    )

    mask = get_mask(genes_3d, data_dir)

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
    parser = argparse.ArgumentParser(description="End-to-end Visium HD preprocessing.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/sample",
        help="base directory containing all data",
    )
    parser.add_argument(
        "--sample_subfolder",
        type=str,
        default="square_008um",
        help="Visium HD output subfolder name",
    )
    parser.add_argument(
        "--raw_image_name",
        type=str,
        default="raw_image.tif",
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
        "--num_top_genes",
        type=int,
        default=1000,
        help="number of highly variable genes",
    )
    parser.add_argument(
        "--pseudo_radius",
        type=int,
        default=55,
        help="pseudo Visium radius in pixels",
    )

    args = parser.parse_args()

    process_visium_hd(
        data_dir=args.data_dir,
        sample_subfolder=args.sample_subfolder,
        raw_image_name=args.raw_image_name,
        output_image_name=args.output_image_name,
        gene_names_name=args.gene_names_name,
        ground_truth_name=args.ground_truth_name,
        scale_pix_size=args.scale_pix_size,
        superpixel_size=args.superpixel_size,
        num_top_genes=args.num_top_genes,
        pseudo_radius=args.pseudo_radius,
    )