import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod

CH_NUM = 60
SEQ_LEN = 4
PATCH_DIM = 200
FLAT_DIM = CH_NUM * SEQ_LEN * PATCH_DIM


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=PATCH_DIM, out_dim=PATCH_DIM, d_model=PATCH_DIM,
            dim_feedforward=PATCH_DIM * 4, seq_len=30,
            n_layer=12, nhead=8,
        )
        if param.use_pretrained_weights:
            device = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=device))
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(PATCH_DIM, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(FLAT_DIM, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(FLAT_DIM, PATCH_DIM),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(PATCH_DIM, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        else:
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(FLAT_DIM, 4 * PATCH_DIM),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * PATCH_DIM, PATCH_DIM),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(PATCH_DIM, 1),
                Rearrange('b 1 -> (b 1)'),
            )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)
