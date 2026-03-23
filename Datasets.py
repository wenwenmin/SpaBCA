import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from utils import get_disk_mask


def dimensionality_reduction(arr, target_size):
    """对最后一维进行插值压缩到 target_size"""
    x = np.arange(arr.shape[-1])
    f = interp1d(x, arr, axis=-1)
    return f(np.linspace(0, arr.shape[-1] - 1, target_size))


def _build_knn_indices(locs: np.ndarray, k: int):
    """构建每个点的KNN索引和距离"""
    locs = np.asarray(locs)
    n = locs.shape[0]
    k_query = min(k + 1, n)

    tree = cKDTree(locs)
    dists, indices = tree.query(locs, k=k_query)

    if indices.ndim == 1:
        indices = indices[:, None]
        dists = dists[:, None]

    return indices.astype(np.int64), dists.astype(np.float32)


class SpotDataset(Dataset):
    def __init__(
        self,
        x_all,
        y,
        enhance_y,
        locs,
        enhance_locs,
        radius,
        his_neighbor_k: int = 0,
        gene_neighbor_k: int = 5,
        return_his_ctx: bool = True,
        exclude_self: bool = True,
        weighted_his_ctx: bool = True,
        temperature: float = 10.0,
    ):
        super().__init__()

        # 构建局部patch掩码（用于确定patch大小）
        mask = get_disk_mask(radius)

        # 提取图像patch（展平）
        self.his = get_patches_flat(x_all, locs, mask)

        # 提取基因邻域特征
        self.gene = get_patches_genes(enhance_locs, enhance_y, k=gene_neighbor_k)

        self.y = y
        self.locs = np.asarray(locs)
        self.mask = mask

        # 邻域上下文参数
        self.his_neighbor_k = int(his_neighbor_k)
        self.return_his_ctx = bool(return_his_ctx)
        self.exclude_self = bool(exclude_self)
        self.weighted_his_ctx = bool(weighted_his_ctx)
        self.temperature = float(temperature)

        self.nei_idx = None
        self.nei_dist = None
        self.his_pool = None

        # 构建histology邻域上下文
        if self.his_neighbor_k > 0 and self.return_his_ctx:
            self.nei_idx, self.nei_dist = _build_knn_indices(self.locs, self.his_neighbor_k)
            # 对patch做平均池化作为邻域特征
            self.his_pool = self.his.mean(axis=1).astype(np.float32)

    def __len__(self):
        return len(self.his)

    def __getitem__(self, idx):
        x_item = {
            "his": self.his[idx],
            "gene": self.gene[idx],
        }

        # 构建邻域上下文特征（可选）
        if self.nei_idx is not None:
            nei = self.nei_idx[idx]
            dists = self.nei_dist[idx]

            # 去掉自身节点
            if self.exclude_self and nei.shape[0] > 1:
                nei = nei[1:]
                dists = dists[1:]

            nei_feats = self.his_pool[nei]

            # 距离加权 or 平均
            if self.weighted_his_ctx:
                weights = np.exp(-dists / self.temperature)
                weights /= weights.sum() + 1e-8
                ctx = (nei_feats * weights[:, None]).sum(axis=0)
            else:
                ctx = nei_feats.mean(axis=0)

            x_item["his_ctx"] = ctx.astype(np.float32)

        return x_item, self.y[idx]


def get_patches_flat(img, locs, mask):
    """
    提取每个坐标对应的局部patch并展平

    注意：当前实现中mask被全1替代，实际返回的是完整patch
    """
    shape = np.array(mask.shape)
    full_mask = np.ones_like(mask, dtype=bool)

    center = shape // 2
    r = np.stack([-center, shape - center], axis=-1)

    x_list = []
    for s in locs:
        patch = img[
            s[0] + r[0][0]: s[0] + r[0][1],
            s[1] + r[1][0]: s[1] + r[1][1]
        ]
        x = patch[full_mask]
        x_list.append(x)

    return np.stack(x_list)


def get_patches_genes(locs, y, k=20):
    """获取每个点的k近邻基因表达"""
    tree = cKDTree(locs)
    _, indices = tree.query(locs, k=k)
    genes_list = [y[idx] for idx in indices]
    return np.stack(genes_list)


def get_patches_genes_test(locs1, locs2, y, k=20):
    """locs1在locs2中的k近邻基因表达（测试用）"""
    tree = cKDTree(locs2)
    _, indices = tree.query(locs1, k=k)
    return [y[idx] for idx in indices]


def get_center_coordinates_rounded(h, w, block_size):
    """生成规则网格块中心坐标（四舍五入）"""
    step = block_size

    center_y = np.arange(step / 2, h, step)
    center_x = np.arange(step / 2, w, step)

    grid_x, grid_y = np.meshgrid(center_x, center_y)
    grid_x = np.round(grid_x).astype(int)
    grid_y = np.round(grid_y).astype(int)

    return np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)