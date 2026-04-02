import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import Mlp, DropPath

def bipartite_soft_matching(metric, size, r, class_token=True, distill_token=False):
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean", merge_embedding = None) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        if merge_embedding is not None:
            src = src + merge_embedding.unsqueeze(0).unsqueeze(0).expand(n, r, c)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    return merge


def score_split_matching(
    metric,
    size,
    kept_num,
    class_token = True
):
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)

        unimportant_metric, important_metric = metric[:, kept_num:],  metric[:, :kept_num]
        r = unimportant_metric.shape[1]
        scores = torch.einsum('bnd, bmd -> bmn', important_metric, unimportant_metric)

        if class_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        node_max = node_max[..., None]
        dst_idx = node_idx[..., None]

    def merge(x: torch.Tensor, mode="mean", merge_embedding = None) -> torch.Tensor:
        n, _, c = x.shape
        src = x[:, kept_num:]
        dst = x[:, :kept_num]
        if merge_embedding is not None:
            src = src + merge_embedding.unsqueeze(0).unsqueeze(0).expand(n, r, c)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return dst

    return merge


def merge_wavg(
    merge, x, size = None, merge_embedding = None
):
    size_ = torch.ones_like(size)
    if merge_embedding is not None:
        x = merge(x, mode="sum", merge_embedding=merge_embedding[:-2])
        size = merge(size, mode="sum", merge_embedding=merge_embedding[-2])
        size_ = merge(size_, mode="sum", merge_embedding=merge_embedding[-1])
    else:
        x = merge(x, mode="sum")
        size = merge(size, mode="sum")
        size_ = merge(size_, mode="sum")

    x = x / size_
    return x, size


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size=None, sort_flag=False):
        B, N, C = x.shape

        qkv_raw = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv_raw.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_bias = None
        if size is not None:
            attn_bias = size.log().transpose(1, 2).view(B, 1, 1, N)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False
        )

        if sort_flag:
            with torch.no_grad():
                q_cls_raw = qkv_raw[:, :1, :C]    # [B, 1, C]
                k_raw     = qkv_raw[:, :, C:2*C]  # [B, N, C]

                cls_attn = torch.bmm(q_cls_raw, k_raw.transpose(1, 2)) * self.scale
                cls_attn = cls_attn.squeeze(1)

                if attn_bias is not None:
                    cls_attn = cls_attn + self.num_heads * attn_bias.squeeze(1).squeeze(1)

                cls_attn[:, 0] = math.inf
                sort_idx = torch.argsort(cls_attn, descending=True)
        else:
            sort_idx = None

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, sort_idx


class CustomBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 info=None, layer=0, skip_lam=1., depth=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.info = info
        self.layer = layer
        self.kept_num = self.info["merge_schedule"][self.layer]
        self.skip_lam = skip_lam
        self.layer_thresh = 2

    def set_layer_thresh(self, thresh):
        self.layer_thresh = thresh

    def forward(self, x, prefix):
        sort_flag = (self.layer >= self.layer_thresh)

        size = self.info[prefix + "size"]

        x_attn, sort_idx = self.attn(self.norm1(x), size, sort_flag=sort_flag)
        x = x + self.drop_path1(x_attn) / self.skip_lam

        if sort_flag:
            x = torch.gather(x, dim=1, index=sort_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            size = torch.gather(size, dim=1, index=sort_idx.unsqueeze(-1))
        
        r = x.shape[1] - self.kept_num
        if r > 0:
            if self.layer < self.layer_thresh:
                merge = bipartite_soft_matching(x.detach(), size, r)
            else:
                merge = score_split_matching(x.detach(), size, self.kept_num)
            x, size = merge_wavg(merge, x, size)
            
        self.info[prefix + "size"] = size

        x = x + self.drop_path2(self.mlp(self.norm2(x))) / self.skip_lam
        return x
