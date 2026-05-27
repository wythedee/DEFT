import torch.nn as nn

from .SSSM import SSSM
from .load_utils import try_load_backbone_weights


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.backbone = SSSM(
            in_channels=200, res_channels=200,
            skip_channels=200, out_channels=200,
            num_res_layers=param.n_layer,
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=param.dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=param.codebook_size_t,
            codebook_size_f=param.codebook_size_f,
            if_codebook=False)
        if getattr(param, 'use_pretrained_weights', False):
            try_load_backbone_weights(self.backbone, param.foundation_dir, cuda_idx=getattr(param, 'cuda', None))
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(49600, 800),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(800, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * patch_size)
        return self.classifier(feats)[:, 0]
