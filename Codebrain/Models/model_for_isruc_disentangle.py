import torch.nn as nn
import torch.nn.functional as F

from .SSSM import SSSM
from .load_utils import try_load_backbone_weights


class SleepBackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, channels: int = 6, steps: int = 30):
        super().__init__()
        self.backbone = backbone
        self.channels = int(channels)
        self.steps = int(steps)

    def forward(self, x):
        bsz, seq_len, merged_steps, patch_size = x.shape
        if merged_steps != self.channels * self.steps:
            raise ValueError(
                f'Expected merged_steps={self.channels * self.steps}, got {merged_steps} for ISRUC disentangle input'
            )
        x = x.contiguous().view(bsz * seq_len, self.channels, self.steps, patch_size)
        feats = self.backbone(x)
        _, ch_num, steps, d_model = feats.shape
        return feats.contiguous().view(bsz, seq_len, ch_num * steps, d_model)


class SleepClassifierWrapper(nn.Module):
    def __init__(self, head: nn.Module, sequence_encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.head = head
        self.sequence_encoder = sequence_encoder
        self.classifier = classifier

    def forward(self, tokens):
        bsz, seq_len, merged_steps, d_model = tokens.shape
        epoch_features = tokens.contiguous().view(bsz, seq_len, merged_steps * d_model)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        return self.classifier(seq_features)


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        base_backbone = SSSM(
            in_channels=param.patch_size,
            res_channels=param.d_model,
            skip_channels=param.d_model,
            out_channels=param.patch_size,
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
            if_codebook=False,
        )
        if getattr(param, 'use_pretrained_weights', False):
            try_load_backbone_weights(base_backbone, param.foundation_dir, cuda_idx=getattr(param, 'cuda', None))
        base_backbone.proj_out = nn.Identity()

        head = nn.Sequential(
            nn.Linear(6 * 30 * 200, 512),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=2048,
            batch_first=True,
            activation=F.gelu,
            norm_first=True,
        )
        sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        classifier = nn.Linear(512, param.num_of_classes)

        self.backbone = SleepBackboneWrapper(base_backbone, channels=6, steps=30)
        self.classifier = SleepClassifierWrapper(head, sequence_encoder, classifier)
