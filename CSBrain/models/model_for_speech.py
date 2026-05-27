import torch
import torch.nn as nn
from functools import partial
from .CSBrain import *


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # Brain region encoding: Frontal (0) | Parietal (1) | Temporal (2) | Occipital (3) | Central (4)
        # The standardized LMDB version of BCIC2020-3 in /path/to/standardized_eeg uses a 60-channel montage.
        brain_regions = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2,
            4, 4, 4, 4,
            2,
            4, 4, 4, 4,
            1, 1, 1, 1, 1,
            3, 3, 3,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
            4, 4, 4,
            2,
            4, 4, 4,
            2,
            1, 1, 1, 1,
            3, 3,
            3, 3, 3, 3
        ]

        electrode_labels = [
            'FP1', 'AF7', 'AF3', 'AFZ', 'F7', 'F5', 'F3', 'F1', 'FZ', 'FT7', 'FC5', 'FC3', 'FC1',
            'T7',
            'C5', 'C3', 'C1', 'CZ',
            'TP7',
            'CP5', 'CP3', 'CP1', 'CPZ',
            'P7', 'P5', 'P3', 'P1', 'PZ',
            'PO7', 'PO3', 'POZ',
            'FP2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8',
            'FC2', 'FC4', 'FC6', 'FT8',
            'C2', 'C4', 'C6',
            'T8',
            'CP2', 'CP4', 'CP6',
            'TP8',
            'P2', 'P4', 'P6', 'P8',
            'PO4', 'PO8',
            'O1', 'OZ', 'O2', 'IZ'
        ]

        topology = {
            0: ["AF7", "AF3", "FP1", "F7", "F5", "F3", "FT7", "FC5", "FC3", "FC1", "F1",
                "AFZ", "FZ",
                "F2", "FC2", "FC4", "FC6", "FT8", "F4", "F6", "F8", "FP2", "AF4", "AF8"],
            4: ["C5", "C3", "C1", "CP5", "CP3", "CP1",
                "CZ", "CPZ",
                "CP2", "CP4", "CP6", "C2", "C4", "C6"],
            1: ["P7", "P5", "P3", "P1",
                "PZ",
                "P2", "P4", "P6", "P8"],
            2: ["TP7", "T7", "T8", "TP8"],
            3: ["PO7", "PO3",
                "O1", "POZ", "OZ",
                "O2", "PO4", "PO8", "IZ"]
        }

        # Group electrode indices by brain region
        region_groups = {}
        for i, region in enumerate(brain_regions):
            if region not in region_groups:
                region_groups[region] = []
            region_groups[region].append((i, electrode_labels[i]))

        # Sort based on topology
        sorted_indices = []
        for region in sorted(region_groups.keys()):
            region_electrodes = region_groups[region]
            sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
            sorted_indices.extend([e[0] for e in sorted_electrodes])

        print("Sorted Indices:", sorted_indices)

        if param.model == 'CSBrain':
            self.backbone = CSBrain(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30,
                n_layer=param.n_layer, nhead=8,
                brain_regions=brain_regions,
                sorted_indices=sorted_indices
            )
        else:
            return 0

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            # Remove "module." and "backbone." prefixes
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "").replace("backbone.", "")
                new_state_dict[new_key] = value

            model_state_dict = self.backbone.state_dict()

            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            model_state_dict.update(matching_dict)
            self.backbone.load_state_dict(model_state_dict)

        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(60 * 3 * 200, 3 * 200),
            nn.ELU(),
            nn.Linear(3 * 200, 200),
            nn.ELU(),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)
        out = self.classifier(feats)
        return out
