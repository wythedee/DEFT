import torch.nn as nn
import torch.nn.functional as F

from .SSSM import SSSM
from .load_utils import try_load_backbone_weights


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.backbone = SSSM(
            in_channels=param.patch_size, res_channels=param.d_model,
            skip_channels=param.d_model, out_channels=param.patch_size,
            num_res_layers=param.n_layer,
            diffusion_step_embed_dim_in=param.d_model,
            diffusion_step_embed_dim_mid=param.d_model,
            diffusion_step_embed_dim_out=param.d_model,
            s4_lmax=param.seq_len,
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
        self.head = nn.Sequential(
            nn.Linear(6 * 30 * 200, 512),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, param.num_of_classes)

    def forward(self, x):
        bz, seq_len, ch_num, epoch_size = x.shape
        x = x.contiguous().view(bz * seq_len, ch_num, 30, 200)
        epoch_features = self.backbone(x)
        epoch_features = epoch_features.contiguous().view(bz, seq_len, ch_num * 30 * 200)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        return self.classifier(seq_features)
