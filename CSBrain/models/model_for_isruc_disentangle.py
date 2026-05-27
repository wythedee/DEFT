import torch.nn as nn

from .model_for_isruc import Model as BaseModel


class SleepBackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, channels: int = 6, steps: int = 30):
        super().__init__()
        self.backbone = backbone
        self.channels = int(channels)
        self.steps = int(steps)

    def forward(self, x):
        bsz, seq_len, merged_steps, patch_size = x.shape
        expected = self.channels * self.steps
        if merged_steps != expected:
            raise ValueError(
                f"Expected merged_steps={expected}, got {merged_steps} for ISRUC disentangle input"
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
        base = BaseModel(param)
        self.backbone = SleepBackboneWrapper(base.backbone, channels=6, steps=30)
        self.classifier = SleepClassifierWrapper(base.head, base.sequence_encoder, base.classifier)
