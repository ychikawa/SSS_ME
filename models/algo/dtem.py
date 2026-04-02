import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., feat_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Calculate feat_dim if not provided
        if feat_dim is None:
            feat_dim = self.head_dim if dim < 1024 else 2 * self.head_dim
        self.feat_dim = feat_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.metric_layer = nn.Linear(dim, feat_dim)
        
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size=None, prop_attn=True):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        metric = self.metric_layer(x.detach())

        if self.training:
            q, k, v = q.float(), k.float(), v.float()
            with torch.amp.autocast('cuda', dtype=torch.float32, enabled=True):
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                
                if size is None or not prop_attn:
                    attn = attn.softmax(dim=-1)
                else:
                    _attn = attn - attn.max(dim=-1, keepdim=True)[0]
                    _attn = _attn.exp_() * size[:, None, None, :].float()
                    attn = _attn / _attn.sum(dim=-1, keepdim=True)
                
                if self.attn_drop_p > 0:
                    attn = F.dropout(attn, p=self.attn_drop_p, training=True)
                
                x = attn @ v
        else:
            attn_bias = None
            if size is not None and prop_attn:
                attn_bias = size.log().view(B, 1, 1, N)
            
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=False
            )

        x = x.to(v.dtype)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, metric


class CustomBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 info=None, layer=0, skip_lam=1., depth=0, 
                 k2=3, tau1=0.1, tau2=0.1, feat_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                        attn_drop=attn_drop, proj_drop=drop, feat_dim=feat_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                      act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.info = info
        self.layer = layer
        self.skip_lam = skip_lam
        
        # DTEM specific parameters
        self.k2 = k2
        self.tau1 = tau1
        self.tau2 = tau2
        
        self.kept_num = self.info["merge_schedule"][self.layer] if self.info else None

    def update_kept_num(self, new_kept_num):
        self.kept_num = new_kept_num

    def _select(self, x, k):
        """Numerically stable k-hot selection"""
        EPSILON = 1e-8
        
        # select
        x = x.type(torch.float32)
        
        # mask - topk2 indices
        _idx = x.argsort(dim=-1, descending=True)[..., :self.k2]
        _x = x.gather(dim=-1, index=_idx)
        
        # scale
        _x = _x / (self.tau1 + EPSILON)
        
        # group - differentiable k-hot selection
        B, N, M = _x.shape
        khot = torch.zeros_like(_x)
        for _ in range(k):
            _x_clipped = torch.clamp(_x, min=-10, max=10)
            onehot_approx = F.softmax(_x_clipped.view(B, -1) / (self.tau2 + EPSILON), dim=-1).view(B, N, M)
            khot += onehot_approx
            khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
            _x = _x + torch.log(khot_mask + EPSILON)
        
        # normalize
        tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
        nkhot = khot / (tmp + EPSILON)
        
        # scatter
        assign = torch.zeros_like(x).scatter_reduce(-1, _idx, nkhot, reduce='sum')
        
        return assign

    def _merge_train(self, x, size, r, n):
        """Training time merge with numerical stability"""
        EPSILON = 1e-8
        
        # Get metric from attention output (stored in info)
        metric = self.info.get("dtem_metric", None)
        if metric is None:
            return x, size, n
        
        # Normalize metric
        metric = metric / (metric.norm(dim=-1, keepdim=True) + EPSILON)
        
        # merge profile
        n = n if self.training else x.size()[1]
        r = min(r, (n - 1) // 2)
        
        if r <= 0:
            return x, size, n
        
        # split - only n tokens participates
        xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        wa, wb = size[..., 1:n:2], size[..., 2:n:2]
        
        # scores
        scores = a @ b.transpose(-1, -2)
        
        # select
        assign = self._select(scores, k=r)
        
        # merge operation with numerical stability
        xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)
        wb_new = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]
        
        # Ensure wb_new is not too small
        wb_new = torch.clamp(wb_new, min=EPSILON)
        
        tmp = 1 - assign.sum(dim=-1)
        wa = wa * torch.clamp(tmp, min=0., max=1.)
        
        xb = xb / (wb_new[..., None] + EPSILON)
        
        # concat first
        w = torch.cat([wa, wb_new], dim=-1)
        nx = torch.cat([xa, xb], dim=1)
        
        # sorted idxs
        nidxs = w.argsort(dim=-1, descending=True)
        
        # sort nx and w
        w = w.gather(dim=-1, index=nidxs)
        nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))

        # output
        x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
        size_output = torch.cat([size[:, :1], w, size[:, n:]], dim=-1)
        return x_output, size_output, n - r

    def _merge_eval(self, x, size, r):
        """Evaluation time merge with numerical stability"""
        EPSILON = 1e-8
        
        # Simple bipartite matching for eval
        metric = self.info.get("dtem_metric", None)
        if metric is None:
            return x, size, x.size(1)
        
        # Normalize metric
        metric = metric / (metric.norm(dim=-1, keepdim=True) + EPSILON)
        
        # Bipartite soft matching (ToMe style)
        merge = self._bipartite_soft_matching(metric, r=r)
        x, size_new = self._merge_wavg(merge, x, size[..., None])
        return x, size_new[..., 0], x.size(1)
    
    def _bipartite_soft_matching(self, metric, r):
        """Bipartite soft matching from ToMe"""
        import math
        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            # Protect class token
            scores[..., 0, :] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            unm_idx = unm_idx.sort(dim=1)[0]

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            return torch.cat([unm, dst], dim=1)

        return merge
    
    def _merge_wavg(self, merge, x, size):
        """Weighted average merge with numerical stability"""
        EPSILON = 1e-8
        x = merge(x * size, mode="sum")
        size = merge(size, mode="sum")
        # Prevent division by zero
        size_clamped = torch.clamp(size, min=EPSILON)
        x = x / size_clamped
        return x, size

    def forward(self, x, prefix=""):
        size = self.info.get(prefix + "size", None)
        if size is None:
            size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        # Attention with metric extraction
        x_attn, metric = self.attn(self.norm1(x), size[..., 0], prop_attn=True)
        
        # Store metric in info for merge operation
        self.info["dtem_metric"] = metric
        
        x = x + self.drop_path1(x_attn) / self.skip_lam

        # Token merging
        r = x.shape[1] - self.kept_num
        if r > 0:
            if self.training:
                n = x.shape[1]  # All tokens can participate in training
                x, size, n = self._merge_train(x, size[..., 0], r, n)
            else:
                x, size, n = self._merge_eval(x, size[..., 0], r)
            size = size[..., None]  # Add dimension back

        self.info[prefix + "size"] = size

        # FFN
        x = x + self.drop_path2(self.mlp(self.norm2(x))) / self.skip_lam
        return x