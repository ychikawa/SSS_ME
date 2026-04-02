import torch
import torch.nn as nn

from timm.models import load_pretrained, register_model
from timm.layers import trunc_normal_, to_2tuple
import numpy as np

from models.algo import default, diffrate, dtem, mctf, ppt, sss_me, sss, tofu, tome

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

class PatchEmbedNaive(nn.Module):
    """ 
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
        proj=img_size*img_size*3*self.embed_dim,
        )
        return sum(block_flops.values())


class PatchEmbed4_2(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
        conv1=img_size/2*img_size/2*3*64*7*7,
        conv2=img_size/2*img_size/2*64*64*3*3,
        conv3=img_size/2*img_size/2*64*64*3*3,
        proj=img_size/2*img_size/2*64*self.embed_dim,
        )
        return sum(block_flops.values())

    
class PatchEmbed4_2_128(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 128, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x
    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
        conv1=img_size/2*img_size/2*3*128*7*7,
        conv2=img_size/2*img_size/2*128*128*3*3,
        conv3=img_size/2*img_size/2*128*128*3*3,
        proj=img_size/2*img_size/2*128*self.embed_dim,
        )
        return sum(block_flops.values())


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate,depth,drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr


class LV_ViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None,
                 skip_lam = 1.0, mix_token=False, return_dense=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes
        if p_emb=='4_2':
            patch_embed_fn = PatchEmbed4_2
        elif p_emb=='4_2_128':
            patch_embed_fn = PatchEmbed4_2_128
        else:
            patch_embed_fn = PatchEmbedNaive

        self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
        self.info = {"num_protected"  : 1,
                     "prune_schedule" : kwargs["prune_schedule"],
                     "merge_schedule" : kwargs["merge_schedule"],
                     }
        self.blocks = nn.ModuleList([
            ALGO[kwargs["algo"]].CustomBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam,
                info = self.info, layer = i, depth = depth)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.return_dense=return_dense
        self.mix_token=mix_token

        if return_dense:
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_embeddings(self,x):
        x = self.patch_embed(x)
        return x

    def resetInfo(self, x):
        self.info["size"]      = torch.ones_like(x[..., 0, None])

    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        self.resetInfo(x)

        for blk in self.blocks:
            x = blk(x, prefix="")
        x = self.norm(x)
        return x

    def forward_features(self,x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)

        # token level mixtoken augmentation 
        # if self.mix_token and self.training:
        #     lam = np.random.beta(self.beta, self.beta)
        #     patch_h, patch_w = x.shape[2],x.shape[3]
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        #     temp_x = x.clone()
        #     temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
        #     x = temp_x
        # else:
        #     bbx1, bby1, bbx2, bby2 = 0,0,0,0

        bbx1, bby1, bbx2, bby2 = 0,0,0,0

        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        x_cls = self.head(x[:,0])


        # if self.return_dense:
        #     x_aux = self.aux_head(x[:,1:])
        #     if not self.training:
        #         return x_cls+0.5*x_aux.max(1)[0]

        #     # recover the mixed part
        #     if self.mix_token and self.training:
        #         x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1])
        #         temp_x = x_aux.clone()
        #         temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
        #         x_aux = temp_x
        #         x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])

        #     return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

        # if self.training:
        #     outputs = {}
        #     outputs["x_student"] = x_cls
        #     return outputs
        return x_cls

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
            if hasattr(block.attn, 'diag_scale_net'):
                params += list(block.attn.diag_scale_net.parameters())
            if hasattr(block, 'merge_embedding'):
                params.append(block.merge_embedding)
            if hasattr(block.attn, 'metric_layer'):
                params += list(block.attn.metric_layer.parameters())
        total_params = sum(p.numel() for p in params)
        print(f"Total number of architecture parameters: {total_params}")
        return iter(params)  

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

@register_model
def vit(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb=1, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model


@register_model
def lvvit(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model

@register_model
def lvvit_t(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=240, depth=12, num_heads=4, mlp_ratio=3.,
        p_emb='4_2',skip_lam=1., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Tiny']
    if pretrained:
        model.load_state_dict(torch.load('./weight/lvvit_t.pth')['state_dict'], strict=False)
    return model


@register_model
def lvvit_s(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    if pretrained:
        model.load_state_dict(torch.load('./weight/lvvit_s-26M-224-83.3.pth.tar'), strict=False)
    return model

@register_model
def lvvit_m(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Medium']
    return model


@register_model
def lvvit_l(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=768,depth=24, num_heads=12, mlp_ratio=3.,
        p_emb='4_2_128',skip_lam=3., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Large']
    return model
