import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, trunc_normal_
import math


def batch_index_select(x, idx):
    if x is not None:
        if len(x.size()) == 3:
            B, N, C = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
            return out
        elif len(x.size()) == 2:
            B, N = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
            return out
        else:
            raise NotImplementedError
    else:
        return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
):
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

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

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def merge_wavg(
    merge, x: torch.Tensor, size: torch.Tensor = None
):
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


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

    def _compute_cls_scores(
        self,
        q: torch.Tensor,   # [B, H, N, head_dim]
        k: torch.Tensor,   # [B, H, N, head_dim]
        v: torch.Tensor,   # [B, H, N, head_dim]
        attn_bias=None,    # [B, 1, 1, N] or None
    ) -> torch.Tensor:     # [B, N]
        B, H, N, D = q.shape
        C = H * D

        cls_attn_logits = (q[:, :, :1, :] @ k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            cls_attn_logits = cls_attn_logits + attn_bias  # [B, H, 1, N]

        cls_attn_weights = cls_attn_logits.softmax(dim=-1).reshape(B, H, N)  # [B, H, N]

        cls_attn_sum = cls_attn_weights.sum(dim=1)  # [B, N]

        v_norm = torch.linalg.norm(
            v.permute(0, 2, 1, 3).reshape(B, N, C), ord=2, dim=-1
        )  # [B, N]

        scores = cls_attn_sum * v_norm           # [B, N]
        scores = scores[:, 1:]
        scores = scores / scores.sum(dim=-1, keepdim=True)

        cls_score = torch.full((B, 1), 2.0, device=scores.device, dtype=scores.dtype)
        scores = torch.cat([cls_score, scores], dim=1)  # [B, N]

        return scores

    def forward(self, x: torch.Tensor, size: torch.Tensor = None):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, H, N, head_dim]

        attn_bias = None
        if size is not None:
            attn_bias = size.log().permute(0, 2, 1).contiguous().view(B, 1, 1, N)

        x_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )

        with torch.no_grad():
            scores = self._compute_cls_scores(q, k, v, attn_bias)  # [B, N]

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        # (x, k.mean(1), scores)
        return x_out, k.mean(1), scores


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
        self.skip_lam = skip_lam
        self.kept_num = self.info["merge_schedule"][self.layer]

    def forward(self, x, prefix):
        size = self.info.get(prefix + "size", None)
        if size is None:
            size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        x_attn, metric, scores = self.attn(self.norm1(x), size)
        x = x + self.drop_path1(x_attn)

        S_op = torch.var(scores[:, 1:], dim=1)

        x_pooling = None
        x_pruning = None
        attn_size_pooling = None
        attn_size_pruning = None

        r = x.shape[1] - self.kept_num

        if r > 0:
            # Apply pooling
            pooling_imgs_indices = torch.nonzero(S_op < 7e-5).squeeze(-1)
            if pooling_imgs_indices.shape[0] > 0:
                x_pooling = x[pooling_imgs_indices]
                metric_pooling = metric[pooling_imgs_indices]
                attn_size_pooling = size[pooling_imgs_indices]
                merge, _ = bipartite_soft_matching(metric_pooling, r, True, False)
                x_pooling, attn_size_pooling = merge_wavg(merge, x_pooling, attn_size_pooling)

            # Apply pruning
            pruning_imgs_indices = torch.nonzero(S_op >= 7e-5).squeeze(-1)
            if pruning_imgs_indices.shape[0] > 0:
                x_pruning = x[pruning_imgs_indices]
                scores_pruning = scores[pruning_imgs_indices]
                attn_size_pruning = size[pruning_imgs_indices]
                _, sorted_indices = torch.sort(scores_pruning, descending=True, dim=-1)
                pruning_indices = sorted_indices[:, :self.kept_num]  # 上位kept_num個を保持
                x_pruning = batch_index_select(x_pruning, pruning_indices)
                attn_size_pruning = batch_index_select(attn_size_pruning, pruning_indices)

            # Merge two subbatch
            if x_pooling is not None and x_pruning is not None:
                x = torch.zeros(
                    x_pooling.shape[0] + x_pruning.shape[0],
                    x_pooling.shape[1],
                    x_pooling.shape[2],
                ).to(x.device)
                x[pooling_imgs_indices] = x_pooling
                x[pruning_imgs_indices] = x_pruning
                self.info[prefix + "size"] = torch.ones(x.shape[0], x.shape[1], 1).to(x.device)
                self.info[prefix + "size"][pooling_imgs_indices] = attn_size_pooling
                self.info[prefix + "size"][pruning_imgs_indices] = attn_size_pruning
            elif x_pooling is not None:
                x = x_pooling
                self.info[prefix + "size"] = attn_size_pooling
            elif x_pruning is not None:
                x = x_pruning
                self.info[prefix + "size"] = attn_size_pruning
        else:
            self.info[prefix + "size"] = size

        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x