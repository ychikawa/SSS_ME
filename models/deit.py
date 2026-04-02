import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import named_apply, adapt_input_conv, register_model
from timm.layers import PatchEmbed, trunc_normal_, lecun_normal_

from models.algo import default, diffrate, dtem, mctf, ppt, sss_me, sss, tofu, tome

_logger = logging.getLogger(__name__)


ALGO = {
    "default": default,
    "diffrate": diffrate,
    "dtem": dtem,
    "mctf": mctf,
    "ppt": ppt,
    "sss_me": sss_me,
    "sss": sss,
    "tofu": tofu,
    "tome": tome
}


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, input_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.cls_token_num  = 1
        self.dist_token_num = 0

        self.num_tokens = self.cls_token_num + self.dist_token_num

        act_layer = act_layer or nn.GELU
        self.depth       = depth
        self.student     = kwargs["task_type"][0]
        self.teacher     = kwargs["task_type"][1]
        self.consistency = kwargs["task_type"][2]

        self.info = {"num_protected"  : self.cls_token_num + self.dist_token_num,
                     "prune_schedule" : kwargs["prune_schedule"],
                     "merge_schedule" : kwargs["merge_schedule"],
                     }

        img_size = input_size
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.cls_token_num:  self.cls_token = nn.Parameter(torch.zeros(1, self.cls_token_num, embed_dim))
        if self.dist_token_num: self.dist_token = nn.Parameter(torch.zeros(1, self.dist_token_num, embed_dim))

        pos_embed_token_num = num_patches + self.num_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_token_num, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            ALGO[kwargs["algo"]].CustomBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                info = self.info, layer = i, depth = depth)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if self.dist_token_num:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head_dist = None

        self.init_weights(weight_init)

    def update_merge_schedule(self, new_merge_schedule):
        if not isinstance(new_merge_schedule, (list, tuple)):
            raise TypeError("merge_schedule must be a list or tuple.")
        if len(new_merge_schedule) != self.depth:
            raise ValueError(f"Length of merge_schedule ({len(new_merge_schedule)}) must match the depth of the model ({self.depth}).")

        self.info["merge_schedule"] = list(new_merge_schedule)
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'update_kept_num'):
                block.update_kept_num(new_merge_schedule[i])
        print("Model's merge_schedule and each block's kept_num have been updated.")

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.

        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token_num:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.dist_token_num:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def resetInfo(self, x):
        n, t, _ = x.shape
        self.info["teacher_source"]    = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        self.info["student_source"]    = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
        self.info["teacher_size"]      = torch.ones_like(x[..., 0, None])
        self.info["student_size"]      = torch.ones_like(x[..., 0, None])

    def forward(self, x):
        result = {}

        x = self.patch_embed(x)
        B, T, D = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        self.resetInfo(x)

        x_teacher = x_student = x
        for i, blk in enumerate(self.blocks):
            x_student = blk(x_student, prefix="student_")
            if i == len(self.blocks) - 1:
                x_student = self.norm(x_student)
                x_student_cls = self.head(x_student[:, 0])
                result["x_student"] = x_student_cls

        return result["x_student"]

    def parameters(self, recurse=True):
        # original network parameter
        params = []
        for n, m in self.named_parameters():
            if n.find('merge_embedding') > -1:
                continue
            if n.find('metric_layer') > -1:
                continue
            params.append(m)
        return iter(params)   

    def arch_parameters(self):
        params = []
        for block in self.blocks:
            if hasattr(block, 'merge_embedding'):
                params.append(block.merge_embedding)
            if hasattr(block.attn, 'metric_layer'):
                params += list(block.attn.metric_layer.parameters())
        total_params = sum(p.numel() for p in params)
        print(f"Total number of architecture parameters: {total_params}")
        return iter(params)  


class VisionTransformerCLIP(VisionTransformer):
    """ Vision Transformer for OpenAI CLIP weights
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # CLIP版には Transformer Block の直前に LayerNorm が必要
        embed_dim = kwargs.get('embed_dim', 768)
        norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
        self.norm_pre = norm_layer(embed_dim)

    def forward(self, x):
        # result 辞書の初期化（既存クラスの互換性維持）
        result = {}

        # 1. Patch Embedding
        x = self.patch_embed(x)
        B, T, D = x.shape
        
        # 2. Add CLS Token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 3. Positional Embedding & Dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # --- CLIP特有の処理: Pre-LayerNorm ---
        x = self.norm_pre(x)

        # 4. ALGO用の情報初期化 (ToMe, DiffRateなどのため)
        self.resetInfo(x)

        # 5. Transformer Blocks (Studentパス)
        x_student = x
        for i, blk in enumerate(self.blocks):
            x_student = blk(x_student, prefix="student_")
            
            if i == len(self.blocks) - 1:
                # 最終層の処理
                x_student = self.norm(x_student)
                # CLSトークンを取り出してヘッドへ
                x_student_cls = self.head(x_student[:, 0])
                result["x_student"] = x_student_cls

        return result["x_student"]


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

@register_model
def deit_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def deit_small(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def deit_base(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

import timm

@register_model
def augreg_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        timm_model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        timm_state_dict = timm_model.state_dict()
        model.load_state_dict(timm_state_dict, strict=False)
    return model

@register_model
def augreg_small(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        timm_model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        timm_state_dict = timm_model.state_dict()
        model.load_state_dict(timm_state_dict, strict=False)
    return model

@register_model
def augreg_base(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        timm_model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
        timm_state_dict = timm_model.state_dict()
        model.load_state_dict(timm_state_dict, strict=False)
    return model

@register_model
def sam_base(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        timm_model = timm.create_model('vit_base_patch16_224.sam_in1k', pretrained=True)
        timm_state_dict = timm_model.state_dict()
        model.load_state_dict(timm_state_dict, strict=False)
    return model

@register_model
def clip_base(pretrained=False, **kwargs):
    # 設定値の定義
    model_kwargs = dict(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )
    
    # 新しく定義したクラスを使用
    model = VisionTransformerCLIP(**model_kwargs)
    
    if pretrained:
        # timmから学習済みモデルをロード
        timm_model = timm.create_model('vit_base_patch16_clip_224.openai_ft_in1k', pretrained=True)
        state_dict = timm_model.state_dict()
        
        # 重みのロード
        # VisionTransformerCLIP には norm_pre があるため、
        # 今度は "Unexpected key" エラーは出なくなります
        msg = model.load_state_dict(state_dict, strict=False)
        _logger.info(f"Loaded pretrained weights with message: {msg}")
        
    return model
