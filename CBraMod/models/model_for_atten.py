import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=4,  # 4 temporal patches
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        # Update the linear layers to match the actual data shape (30 channels instead of 75)
        # For binary classification with BCEWithLogitsLoss, we need only 1 output neuron
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, 1),  # 1 output neuron for binary classification
                Rearrange('b 1 -> (b 1)'),  # Reshape to (batch_size,)
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(30 * 4 * 200, 1),  # 1 output neuron for binary classification
                Rearrange('b 1 -> (b 1)'),  # Reshape to (batch_size,)
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(30 * 4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),  # 1 output neuron for binary classification
                Rearrange('b 1 -> (b 1)'),  # Reshape to (batch_size,)
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(30 * 4 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),  # 1 output neuron for binary classification
                Rearrange('b 1 -> (b 1)'),  # Reshape to (batch_size,)
            )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len, patch_size)
        # For ATTEN dataset: (batch_size, 30, 4, 200) - 30 channels instead of 75
        # No need to reshape as data is already in correct format from preprocessing
        
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)  # Output shape: (batch_size, 30, 4, 200)
        out = self.classifier(feats)
        return out