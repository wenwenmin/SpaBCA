import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

from utils import get_disk_mask


# =========================
# Cross Attention Layer
# =========================
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, q_dim, kv_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = num_heads ** -0.5

        self.query_proj = nn.Linear(q_dim, embed_dim)
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, N1, _ = query.shape
        _, N2, _ = key.shape

        Q = self.query_proj(query).reshape(B, N1, self.num_heads, -1).permute(0, 2, 1, 3)
        K = self.key_proj(key).reshape(B, N2, self.num_heads, -1).permute(0, 2, 1, 3)
        V = self.value_proj(value).reshape(B, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        att = (Q @ K.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)

        out = (att @ V).transpose(1, 2).flatten(2)
        return self.output_proj(out)


# =========================
# Bi-directional Cross Attention (BCA)
# =========================
class CrossAttentionModel(nn.Module):
    def __init__(self, embed_dim, his_dim, ge_dim, num_heads=8):
        super().__init__()
        self.ca1 = CrossAttentionLayer(embed_dim, his_dim, ge_dim, num_heads)
        self.ca2 = CrossAttentionLayer(embed_dim, ge_dim, his_dim, num_heads)
        self.ca3 = CrossAttentionLayer(embed_dim, embed_dim, embed_dim, num_heads)

    def forward(self, his_fea, gene_fea):
        x1 = self.ca1(his_fea, gene_fea, gene_fea)
        x2 = self.ca2(gene_fea, his_fea, his_fea)
        return self.ca3(x1, x2, x2)


# =========================
# Linear Block
# =========================
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.01, beta=0.01, bias=False, use_act=True):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim)) if bias else None

        self.act = nn.ELU(alpha=alpha, inplace=True) if use_act else nn.Identity()
        self.beta = beta

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, indices=None):
        if indices is None:
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
        else:
            w = self.weight[:, indices]
            out = torch.matmul(x, w)
            if self.bias is not None:
                out = out + self.bias[indices]

        return self.act(out) + self.beta


# =========================
# SpaBCA Model (核心模型)
# =========================
class SpaBCA(pl.LightningModule):
    def __init__(self, lr, num_features, num_genes, num_embeddings, radius, bias=False):
        super().__init__()

        self.lr = lr
        self.radius = radius

        # === Local self-attention (histology) ===
        self.hist_self_att = nn.MultiheadAttention(
            embed_dim=num_features,
            num_heads=8,
            batch_first=True
        )

        # === Bi-directional Cross Attention ===
        self.bca1 = CrossAttentionModel(num_embeddings, num_features, num_genes)
        self.fc1 = Linear(num_embeddings, num_embeddings, bias=bias, use_act=False)

        self.bca2 = CrossAttentionModel(num_embeddings, num_embeddings, num_genes)
        self.fc2 = Linear(num_embeddings, num_embeddings, bias=bias, use_act=False)

        # === Channel Attention ===
        self.channel_att = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_embeddings // 4, num_embeddings),
            nn.Sigmoid()
        )

        # === Output head ===
        self.output_module = nn.Sequential(
            Linear(num_embeddings + num_features, 1024, bias=bias, use_act=False),
            Linear(1024, 512, bias=bias, use_act=False),
            Linear(512, 512, bias=bias, use_act=False),
            Linear(512, 512, bias=bias, use_act=False)
        )

        self.final_layer = Linear(512, num_genes, bias=bias)

        self.save_hyperparameters()

    # =========================
    # Feature Fusion
    # =========================
    def get_multi_feature(self, his_fea, gene_fea):
        # Local self-attention
        att_out, _ = self.hist_self_att(his_fea, his_fea, his_fea)
        his_fea = his_fea + att_out

        # BCA block 1
        x = self.bca1(his_fea, gene_fea)
        x = self.fc1(x) + x

        # BCA block 2
        x = self.bca2(x, gene_fea)
        x = self.fc2(x) + x

        # Channel attention
        weight = x.mean(dim=1, keepdim=True)
        weight = self.channel_att(weight)
        x = x * weight

        return x

    # =========================
    # Gene Prediction
    # =========================
    def get_gene(self, x, indices=None):
        x = self.output_module(x)
        return self.final_layer(x, indices)

    def forward(self, x, indices=None):
        his_fea = x['his'].to(torch.float32)
        gene_fea = x['gene'].to(torch.float32)

        x = self.get_multi_feature(his_fea, gene_fea)
        x = torch.cat((x, his_fea), dim=2)

        return self.get_gene(x, indices)

    # =========================
    # Training Step
    # =========================
    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        y_pred = self.forward(x)

        mask = get_disk_mask(55 / 16)
        mask = torch.BoolTensor(mask).to(y_pred.device)

        y_pred = y_pred.reshape(
            y_pred.size(0),
            mask.shape[0],
            mask.shape[1],
            y_pred.size(-1)
        )

        y_pred = torch.masked_select(
            y_pred,
            mask.unsqueeze(0).unsqueeze(-1)
        ).view(y_pred.size(0), -1, y_pred.size(-1))

        y_mean_pred = y_pred.mean(-2)

        loss = ((y_mean_pred - y_mean) ** 2).mean()
        self.log('loss', loss.sqrt(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)