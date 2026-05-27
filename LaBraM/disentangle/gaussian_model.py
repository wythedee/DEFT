from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BranchImpl = Literal["mlp", "gaussian"]


@dataclass(frozen=True)
class LossToggles:
    use_style_contrastive: bool = True
    use_style_invariance: bool = True
    use_var_ratio: bool = True
    use_film_reconstruction: bool = False


def _mpd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected (B,T,D), got {tuple(x.shape)}")
    bsz, steps, _ = x.shape
    if steps <= 1:
        return torch.zeros((bsz,), device=x.device, dtype=torch.float32)
    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        dist = torch.cdist(x.float(), x.float())
    mask = ~torch.eye(steps, device=x.device, dtype=torch.bool).unsqueeze(0)
    mpd = dist.masked_select(mask).view(bsz, steps * (steps - 1)).mean(dim=1).clamp_min(eps)
    return mpd.to(dtype=torch.float32)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        logits = embeddings @ embeddings.t() / self.temperature
        bsz = embeddings.size(0)
        device = embeddings.device
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        labels = labels.view(-1).to(device)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        positives = same & ~eye
        negatives = ~same
        logits = logits.masked_fill(eye, float("-inf"))
        logits = logits - torch.nan_to_num(logits.max(dim=1, keepdim=True).values, nan=0.0)
        exp_logits = torch.exp(logits)
        denom = (exp_logits * (positives | negatives)).sum(dim=1).clamp_min(1e-8)
        num = (exp_logits * positives).sum(dim=1)
        valid = positives.any(dim=1)
        if not valid.any():
            return torch.zeros((), device=device, dtype=embeddings.dtype)
        return -torch.log((num[valid] / denom[valid]).clamp_min(1e-8)).mean()


class TokenMLP(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden or d_model * 4)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualTokenProj(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.norm(x))


class GaussianInferenceLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        transition: Literal["identity", "content_aware"] = "identity",
        num_transition_components: int = 16,
        transition_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.transition = str(transition)
        hidden = int(transition_hidden or d_model * 2)
        self.obs_log_precision = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        last = self.obs_log_precision[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 4.0)
        self.prior_mean = nn.Parameter(torch.zeros(self.d_model))
        self.prior_log_precision = nn.Parameter(torch.zeros(self.d_model))
        if self.transition == "identity":
            self.transition_net = None
        else:
            self.transition_net = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_transition_components),
            )
            self.transition_bank = nn.Parameter(torch.zeros(num_transition_components, self.d_model))
            self.transition_log_precision = nn.Parameter(torch.zeros(num_transition_components, self.d_model))

    def _predict(self, mean_prev: torch.Tensor, log_precision_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transition == "identity" or self.transition_net is None:
            return mean_prev, log_precision_prev
        logits = self.transition_net(mean_prev)
        weights = torch.softmax(logits, dim=-1)
        mean_delta = weights @ self.transition_bank
        log_precision_delta = weights @ self.transition_log_precision
        return mean_prev + mean_delta, log_precision_prev + log_precision_delta

    def forward(
        self,
        z_seq: torch.Tensor,
        remove_mean: Optional[torch.Tensor] = None,
        remove_log_precision: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_seq.dim() != 3:
            raise ValueError(f"Expected z_seq (B,S,D), got {tuple(z_seq.shape)}")
        z_obs = z_seq
        obs_log_precision = self.obs_log_precision(z_obs)
        if remove_mean is not None and remove_log_precision is not None:
            z_obs = z_obs - remove_mean
            obs_log_precision = obs_log_precision + remove_log_precision
        post_means = []
        post_logs = []
        pred_means = []
        pred_logs = []
        mean_prev = self.prior_mean.view(1, 1, -1).expand(z_seq.size(0), 1, -1).squeeze(1)
        log_precision_prev = self.prior_log_precision.view(1, 1, -1).expand(z_seq.size(0), 1, -1).squeeze(1)
        for t in range(z_seq.size(1)):
            mean_pred, log_precision_pred = self._predict(mean_prev, log_precision_prev)
            pred_means.append(mean_pred.unsqueeze(1))
            pred_logs.append(log_precision_pred.unsqueeze(1))
            gate_logits = torch.stack([obs_log_precision[:, t, :], log_precision_pred], dim=-1)
            gains = torch.softmax(gate_logits, dim=-1)
            mean_post = gains[..., 0] * z_obs[:, t, :] + gains[..., 1] * mean_pred
            log_precision_post = torch.logsumexp(gate_logits, dim=-1)
            post_means.append(mean_post.unsqueeze(1))
            post_logs.append(log_precision_post.unsqueeze(1))
            mean_prev = mean_post
            log_precision_prev = log_precision_post
        return {
            "posterior_mean": torch.cat(post_means, dim=1).to(dtype=z_seq.dtype),
            "posterior_log_precision": torch.cat(post_logs, dim=1).to(dtype=z_seq.dtype),
            "predict_mean": torch.cat(pred_means, dim=1).to(dtype=z_seq.dtype),
            "predict_log_precision": torch.cat(pred_logs, dim=1).to(dtype=z_seq.dtype),
        }


class GaussianBranchDisentangler(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        num_transition_components: int = 16,
        transition_hidden: Optional[int] = None,
        remove_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.layer1_static = GaussianInferenceLayer(d_model, transition="identity")
        self.layer2_dynamic = GaussianInferenceLayer(
            d_model,
            transition="content_aware",
            num_transition_components=num_transition_components,
            transition_hidden=transition_hidden,
        )
        self.layer3_static = GaussianInferenceLayer(d_model, transition="identity")
        self.layer4_dynamic = GaussianInferenceLayer(
            d_model,
            transition="content_aware",
            num_transition_components=num_transition_components,
            transition_hidden=transition_hidden,
        )
        self.remove_gate_12_logit = nn.Parameter(torch.tensor(float(remove_gate_init)))
        self.remove_gate_23_logit = nn.Parameter(torch.tensor(float(remove_gate_init)))
        self.remove_gate_34_logit = nn.Parameter(torch.tensor(float(remove_gate_init)))

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        mixed_windows = feats.mean(dim=1)
        layer1 = self.layer1_static(mixed_windows)
        gate12 = torch.sigmoid(self.remove_gate_12_logit)
        layer2 = self.layer2_dynamic(
            mixed_windows,
            remove_mean=gate12 * layer1["predict_mean"],
            remove_log_precision=layer1["predict_log_precision"] + torch.log(gate12.clamp_min(1e-6)),
        )
        gate23 = torch.sigmoid(self.remove_gate_23_logit)
        layer3 = self.layer3_static(
            mixed_windows,
            remove_mean=gate23 * layer2["predict_mean"],
            remove_log_precision=layer2["predict_log_precision"] + torch.log(gate23.clamp_min(1e-6)),
        )
        gate34 = torch.sigmoid(self.remove_gate_34_logit)
        layer4 = self.layer4_dynamic(
            mixed_windows,
            remove_mean=gate34 * layer3["predict_mean"],
            remove_log_precision=layer3["predict_log_precision"] + torch.log(gate34.clamp_min(1e-6)),
        )
        return {
            "mixed_windows": mixed_windows,
            "dynamic_windows": layer4["posterior_mean"],
            "static_windows": layer3["posterior_mean"],
            "remove_gate_12": gate12.to(dtype=feats.dtype),
            "remove_gate_23": gate23.to(dtype=feats.dtype),
            "remove_gate_34": gate34.to(dtype=feats.dtype),
        }


class FiLMDecoder(nn.Module):
    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, content_tokens: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        h = self.norm(content_tokens)
        gamma = self.gamma(style_vec).view(-1, 1, 1, h.size(-1))
        beta = self.beta(style_vec).view(-1, 1, 1, h.size(-1))
        h = h * (1.0 + gamma) + beta
        return self.proj(h)


class DisentangleV7LiteModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        d_model: int,
        patch_size: int,
        dropout: float = 0.0,
        branch_impl: BranchImpl = "gaussian",
        gaussian_num_transition_components: int = 16,
        gaussian_transition_hidden: Optional[int] = None,
        gaussian_remove_gate_init: float = -2.0,
        bypass_mid_mlp: bool = False,
        bypass_content_proj: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "backbone") or not hasattr(base_model, "classifier"):
            raise ValueError("base_model must expose .backbone and .classifier")
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.mid = TokenMLP(self.d_model, dropout=dropout)
        self.bypass_mid_mlp = bool(bypass_mid_mlp)
        self.bypass_content_proj = bool(bypass_content_proj)
        self.branch_impl = str(branch_impl)
        if self.branch_impl == "mlp":
            self.content_proj = ResidualTokenProj(self.d_model)
            self.style_proj = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
            self.gaussian_disentangler = None
        elif self.branch_impl == "gaussian":
            self.content_proj = nn.Identity()
            self.style_proj = nn.Identity()
            self.gaussian_disentangler = GaussianBranchDisentangler(
                self.d_model,
                num_transition_components=gaussian_num_transition_components,
                transition_hidden=gaussian_transition_hidden,
                remove_gate_init=gaussian_remove_gate_init,
            )
        else:
            raise ValueError(f"Unsupported branch_impl: {branch_impl}")
        self.decoder = FiLMDecoder(self.d_model, self.patch_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        if not self.bypass_mid_mlp:
            feats = self.mid(feats)
        gaussian_remove_gate_12 = torch.empty((0,), device=x.device, dtype=feats.dtype)
        gaussian_remove_gate_23 = torch.empty((0,), device=x.device, dtype=feats.dtype)
        gaussian_remove_gate_34 = torch.empty((0,), device=x.device, dtype=feats.dtype)
        if self.branch_impl == "mlp":
            content_tokens = feats if self.bypass_content_proj else self.content_proj(feats)
            style_tokens = self.style_proj(feats)
            style_windows = style_tokens.mean(dim=1)
            style_vec = style_windows.mean(dim=1)
        else:
            if self.bypass_content_proj:
                content_tokens = feats
                style_tokens = feats
                style_windows = style_tokens.mean(dim=1)
                style_vec = style_windows.mean(dim=1)
            else:
                g_out = self.gaussian_disentangler(feats)
                mixed_windows = g_out["mixed_windows"]
                dynamic_windows = g_out["dynamic_windows"]
                static_windows = g_out["static_windows"]
                content_tokens = feats + (dynamic_windows - mixed_windows).unsqueeze(1)
                style_tokens = feats + (static_windows - mixed_windows).unsqueeze(1)
                style_windows = static_windows
                style_vec = style_windows.mean(dim=1)
                gaussian_remove_gate_12 = g_out["remove_gate_12"].reshape(1)
                gaussian_remove_gate_23 = g_out["remove_gate_23"].reshape(1)
                gaussian_remove_gate_34 = g_out["remove_gate_34"].reshape(1)
        task_logits = self.classifier(content_tokens)
        return {
            "mixed_tokens": feats,
            "content_tokens": content_tokens,
            "style_tokens": style_tokens,
            "style_windows": style_windows,
            "style_vec": style_vec,
            "task_logits": task_logits,
            "gaussian_remove_gate_12": gaussian_remove_gate_12,
            "gaussian_remove_gate_23": gaussian_remove_gate_23,
            "gaussian_remove_gate_34": gaussian_remove_gate_34,
        }

    def compute_losses(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        *,
        task_kind: str,
        label_smoothing: float,
        toggles: LossToggles,
        temperature: float,
        var_ratio_r: float,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        logits = outputs["task_logits"]
        if task_kind == "multiclass":
            losses["task"] = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
        elif task_kind == "binary":
            losses["task"] = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))
        else:
            raise ValueError(f"Unsupported task_kind: {task_kind}")
        if toggles.use_style_contrastive:
            info_nce = InfoNCELoss(temperature=temperature).to(logits.device)
            losses["style_cl"] = info_nce(outputs["style_vec"], subject_ids.view(-1))
        if toggles.use_style_invariance:
            style_windows = outputs["style_windows"]
            style_vec = outputs["style_vec"].unsqueeze(1)
            losses["style_invar"] = (style_windows - style_vec).pow(2).mean()
        if toggles.use_var_ratio:
            content_windows = outputs["content_tokens"].mean(dim=1)
            style_windows = outputs["style_windows"]
            v_c = _mpd(content_windows)
            v_s = _mpd(style_windows)
            losses["var_ratio"] = torch.clamp(1.0 - (v_c / (float(var_ratio_r) * v_s + 1e-8)), min=0.0).mean()
        if toggles.use_film_reconstruction:
            recon = self.decoder(outputs["content_tokens"], outputs["style_vec"])
            losses["recon"] = F.mse_loss(recon, x)
        return losses

    def get_num_layers(self) -> int:
        return self.backbone.get_num_layers()

    def no_weight_decay(self) -> set[str]:
        return self.backbone.no_weight_decay()
