import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, DropPath, trunc_normal_


def get_merge_func(metric, kept_number, class_token = True):
    with torch.no_grad():
        metric = metric/metric.norm(dim=-1, keepdim=True)
        unimportant_tokens_metric = metric[:, kept_number:]
        compress_number = unimportant_tokens_metric.shape[1]
        important_tokens_metric = metric[:,:kept_number]
        similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
        if class_token:
            similarity[..., :, 0] = -math.inf
        node_max, node_idx = similarity.max(dim=-1)
        dst_idx = node_idx[..., None]

    def merge(x: torch.Tensor, mode="mean"):
        src = x[:,kept_number:]
        dst = x[:,:kept_number]
        n, t1, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
        return dst

    return merge, node_max


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
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # --- SDPA ---
        attn_bias = None
        if size is not None:
            # size: (B, N, 1) -> (B, 1, 1, N)
            attn_bias = size.log().transpose(1, 2).view(B, 1, 1, N)

        x_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False
        )

        with torch.no_grad():
            q_cls = q[:, :, :1, :] # (B, H, 1, head_dim)
            cls_attn = (q_cls @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                cls_attn = cls_attn + attn_bias
            cls_attn = cls_attn.softmax(dim=-1) # (B, H, 1, N)

        x = x_out.transpose(1, 2).reshape(B, N, C)
        # ----------------
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, cls_attn


class CustomBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 info = None, layer = 0, skip_lam=1., depth = 0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.info = info
        self.layer = layer
        self.prune_kept_num = self.info["prune_schedule"][self.layer]
        self.merge_kept_num = self.info["merge_schedule"][self.layer]
        self.skip_lam = skip_lam
        self.depth = depth

    def forward(self, x, prefix):
        size = self.info[prefix + "size"]

        x_attn, attn = self.attn(self.norm1(x), size)
        x = x + self.drop_path1(x_attn)/self.skip_lam

        # importance metric
        cls_attn = attn[:, :, 0, 1:]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((x.shape[0], 1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)

        # sorting
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        size = torch.gather(size, dim=1, index=idx.unsqueeze(-1))

        # pruning
        x = x[:, :self.prune_kept_num]
        size = size[:, :self.prune_kept_num]

        # merging
        if self.merge_kept_num < self.prune_kept_num:
            merge, node_max = get_merge_func(x.detach(), kept_number=self.merge_kept_num)
            divider = torch.ones_like(size)
            divider = merge(divider, mode='sum')
            x = merge(x, mode='sum')
            x = x / divider
            # optimize proportional attention in ToMe by considering similarity, this is benefit to the accuracy of off-the-shelf model.
            size = torch.cat((size[:, :self.merge_kept_num], size[:, self.merge_kept_num:]*node_max[..., None]), dim=1)
            size = merge(size, mode='sum')
        
        self.info[prefix + "size"] = size

        x = x + self.drop_path2(self.mlp(self.norm2(x)))/self.skip_lam
        return x