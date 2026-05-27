from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CodebookMode = Literal["window", "token"]
ContentCodebookKind = Literal["vq", "retrieval_topk"]
ContentCodebookFuse = Literal["delta", "concat_proj", "concat_replace"]
BranchImpl = Literal["mlp", "gaussian"]


@dataclass(frozen=True)
class LossToggles:
    use_style_contrastive: bool = True
    use_style_invariance: bool = True
    use_var_ratio: bool = True
    use_film_reconstruction: bool = False
    use_content_codebook: bool = False


def _mpd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean pairwise distance over the sequence axis.

    Args:
        x: (B, T, D)
    Returns:
        (B,) MPD per batch item.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x with shape (B,T,D), got {tuple(x.shape)}")
    bsz, steps, _ = x.shape
    if steps <= 1:
        # Use float32 to avoid AMP/float16 underflow in downstream ratios.
        return torch.zeros((bsz,), device=x.device, dtype=torch.float32)

    # NOTE:
    # - Under AMP, torch.cdist may run in float16 and can (a) underflow distances to 0 or (b) produce NaNs.
    # - The var-ratio loss divides by v_s, so v_s==0 combined with tiny eps in float16 can become 0/0 -> NaN.
    # We therefore disable autocast for cdist and always return float32.
    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        x32 = x.float()
        dist = torch.cdist(x32, x32)  # (B, T, T)
    mask = ~torch.eye(steps, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1,T,T)
    mpd = dist.masked_select(mask).view(bsz, steps * (steps - 1)).mean(dim=1).clamp_min(eps)
    return mpd.to(dtype=torch.float32)


class InfoNCELoss(nn.Module):
    """Supervised InfoNCE where positives share the same label (e.g., subject id)."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(f"Expected embeddings (B,D), got {tuple(embeddings.shape)}")
        if labels.dim() != 1:
            labels = labels.view(-1)
        embeddings = F.normalize(embeddings, dim=1)
        logits = embeddings @ embeddings.t() / self.temperature  # (B,B)

        bsz = embeddings.size(0)
        device = embeddings.device
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        labels = labels.to(device)

        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B,B)
        positives = same & ~eye
        negatives = ~same

        logits = logits.masked_fill(eye, float("-inf"))
        # For numerical stability, subtract max per row.
        logits = logits - torch.nan_to_num(logits.max(dim=1, keepdim=True).values, nan=0.0)

        exp_logits = torch.exp(logits)
        denom = (exp_logits * (positives | negatives)).sum(dim=1).clamp_min(1e-8)
        num = (exp_logits * positives).sum(dim=1)
        valid = positives.any(dim=1)
        if not valid.any():
            return torch.zeros((), device=device, dtype=embeddings.dtype)
        loss = -torch.log((num[valid] / denom[valid]).clamp_min(1e-8)).mean()
        return loss


class TokenMLP(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden or d_model * 4)
        self._residual = True
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        # Identity-safe init: start as an exact residual identity (net(x)=0),
        # so enabling the MLP does not destroy pretrained token features.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return x + out if self._residual else out


class ResidualTokenProj(nn.Module):
    """LayerNorm + Linear with residual identity initialization."""

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
    """Diagonal Gaussian inference with either static or content-aware transition."""

    def __init__(
        self,
        d_model: int,
        *,
        transition: Literal["identity", "content_aware"] = "identity",
        num_transition_components: int = 16,
        transition_hidden: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.transition = str(transition)
        self.num_transition_components = int(num_transition_components)
        self.eps = float(eps)

        self.obs_log_precision = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        # Start with high observation precision so inference behaves close to identity initially.
        obs_last = self.obs_log_precision[-1]
        if isinstance(obs_last, nn.Linear):
            nn.init.zeros_(obs_last.weight)
            nn.init.constant_(obs_last.bias, 4.0)

        self.prior_mean = nn.Parameter(torch.zeros(self.d_model))
        self.prior_log_precision = nn.Parameter(torch.full((self.d_model,), -4.0))

        self.transition_bank: Optional[nn.Parameter] = None
        self.transition_diag_logits: Optional[nn.Parameter] = None
        self.filter_generator: Optional[nn.Module] = None
        if self.transition == "content_aware":
            if self.num_transition_components <= 0:
                raise ValueError("num_transition_components must be > 0 for content_aware transition.")
            bank = torch.eye(self.d_model).unsqueeze(0).repeat(self.num_transition_components, 1, 1)
            self.transition_bank = nn.Parameter(bank)
            self.transition_diag_logits = nn.Parameter(torch.zeros(self.num_transition_components, self.d_model))

            hidden_dim = int(transition_hidden or self.d_model * 2)
            self.filter_generator = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_transition_components),
            )
            fg_last = self.filter_generator[-1]
            if isinstance(fg_last, nn.Linear):
                nn.init.zeros_(fg_last.weight)
                nn.init.zeros_(fg_last.bias)
        elif self.transition != "identity":
            raise ValueError(f"Unsupported transition={self.transition!r}")

    def _predict(self, mean_prev: torch.Tensor, log_precision_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transition == "identity":
            return mean_prev, log_precision_prev
        if self.transition_bank is None or self.transition_diag_logits is None or self.filter_generator is None:
            raise RuntimeError("content_aware transition is not initialized correctly.")

        weights = torch.softmax(self.filter_generator(mean_prev), dim=-1)  # (B,N)
        transition = torch.einsum("bn,nij->bij", weights, self.transition_bank)  # (B,D,D)
        mean_pred = torch.einsum("bij,bj->bi", transition, mean_prev)  # (B,D)

        diag_scale_bank = F.softplus(self.transition_diag_logits) + 1e-3  # (N,D), positive
        diag_scale = torch.einsum("bn,nd->bd", weights, diag_scale_bank).clamp_min(self.eps)  # (B,D)
        log_precision_pred = log_precision_prev + torch.log(diag_scale)
        return mean_pred, log_precision_pred

    def forward(
        self,
        z_seq: torch.Tensor,
        *,
        remove_mean: Optional[torch.Tensor] = None,
        remove_log_precision: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_seq.dim() != 3:
            raise ValueError(f"GaussianInferenceLayer expects z_seq (B,S,D), got {tuple(z_seq.shape)}")
        if z_seq.size(-1) != self.d_model:
            raise ValueError(f"Expected z_seq last dim {self.d_model}, got {z_seq.size(-1)}")
        if remove_mean is not None and remove_mean.shape != z_seq.shape:
            raise ValueError(f"remove_mean shape mismatch: expected {tuple(z_seq.shape)}, got {tuple(remove_mean.shape)}")
        if remove_log_precision is not None and remove_log_precision.shape != z_seq.shape:
            raise ValueError(
                "remove_log_precision shape mismatch: "
                f"expected {tuple(z_seq.shape)}, got {tuple(remove_log_precision.shape)}"
            )

        bsz, steps, _ = z_seq.shape
        device_type = "cuda" if z_seq.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z = z_seq.float()
            obs_log_precision = self.obs_log_precision(z).float()  # (B,S,D)
            z_obs = z if remove_mean is None else z - remove_mean.float()

            if remove_log_precision is not None:
                obs_precision = torch.exp(obs_log_precision)
                remove_precision = torch.exp(remove_log_precision.float()).clamp_min(self.eps)
                # Harmonic-mean fusion from the paper's simplified Gaussian update.
                obs_precision = (obs_precision * remove_precision) / (obs_precision + remove_precision + self.eps)
                obs_log_precision = torch.log(obs_precision + self.eps)

            mean_prev = self.prior_mean.float().unsqueeze(0).expand(bsz, -1)
            log_precision_prev = self.prior_log_precision.float().unsqueeze(0).expand(bsz, -1)

            post_means = []
            post_logs = []
            pred_means = []
            pred_logs = []

            for t in range(steps):
                mean_pred, log_precision_pred = self._predict(mean_prev, log_precision_prev)  # (B,D), (B,D)
                pred_means.append(mean_pred.unsqueeze(1))
                pred_logs.append(log_precision_pred.unsqueeze(1))

                gate_logits = torch.stack([obs_log_precision[:, t, :], log_precision_pred], dim=-1)  # (B,D,2)
                gains = torch.softmax(gate_logits, dim=-1)
                gain_obs = gains[..., 0]
                gain_prev = gains[..., 1]

                mean_post = gain_obs * z_obs[:, t, :] + gain_prev * mean_pred
                log_precision_post = torch.logsumexp(gate_logits, dim=-1)

                post_means.append(mean_post.unsqueeze(1))
                post_logs.append(log_precision_post.unsqueeze(1))
                mean_prev = mean_post
                log_precision_prev = log_precision_post

            posterior_mean = torch.cat(post_means, dim=1)
            posterior_log_precision = torch.cat(post_logs, dim=1)
            predict_mean = torch.cat(pred_means, dim=1)
            predict_log_precision = torch.cat(pred_logs, dim=1)

        return {
            "posterior_mean": posterior_mean.to(dtype=z_seq.dtype),
            "posterior_log_precision": posterior_log_precision.to(dtype=z_seq.dtype),
            "predict_mean": predict_mean.to(dtype=z_seq.dtype),
            "predict_log_precision": predict_log_precision.to(dtype=z_seq.dtype),
            "obs_log_precision": obs_log_precision.to(dtype=z_seq.dtype),
        }


class GaussianBranchDisentangler(nn.Module):
    """Four-layer Gaussian disentangler: static -> dynamic -> static -> dynamic."""

    def __init__(
        self,
        d_model: int,
        *,
        num_transition_components: int = 16,
        transition_hidden: Optional[int] = None,
        remove_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.window_norm = nn.Identity()
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
        if feats.dim() != 4:
            raise ValueError(f"GaussianBranchDisentangler expects feats (B,C,S,D), got {tuple(feats.shape)}")
        mixed_windows = feats.mean(dim=1)  # (B,S,D)
        z_seq = self.window_norm(mixed_windows)

        layer1 = self.layer1_static(z_seq)

        gate12 = torch.sigmoid(self.remove_gate_12_logit)
        log_gate12 = torch.log(gate12.clamp_min(1e-6))
        layer2 = self.layer2_dynamic(
            z_seq,
            remove_mean=gate12 * layer1["predict_mean"],
            remove_log_precision=layer1["predict_log_precision"] + log_gate12,
        )

        gate23 = torch.sigmoid(self.remove_gate_23_logit)
        log_gate23 = torch.log(gate23.clamp_min(1e-6))
        layer3 = self.layer3_static(
            z_seq,
            remove_mean=gate23 * layer2["predict_mean"],
            remove_log_precision=layer2["predict_log_precision"] + log_gate23,
        )

        gate34 = torch.sigmoid(self.remove_gate_34_logit)
        log_gate34 = torch.log(gate34.clamp_min(1e-6))
        layer4 = self.layer4_dynamic(
            z_seq,
            remove_mean=gate34 * layer3["predict_mean"],
            remove_log_precision=layer3["predict_log_precision"] + log_gate34,
        )

        dynamic_windows = layer4["posterior_mean"]
        static_windows = layer3["posterior_mean"]

        return {
            "mixed_windows": mixed_windows,
            "dynamic_windows": dynamic_windows,
            "static_windows": static_windows,
            "layer1_mean": layer1["posterior_mean"],
            "layer2_mean": layer2["posterior_mean"],
            "layer3_mean": layer3["posterior_mean"],
            "layer4_mean": layer4["posterior_mean"],
            "remove_gate_12": gate12.to(dtype=feats.dtype),
            "remove_gate_23": gate23.to(dtype=feats.dtype),
            "remove_gate_34": gate34.to(dtype=feats.dtype),
        }


class ConcatResidualFusion(nn.Module):
    """Fuse (x, r) via concat + residual projection.

    This is designed to be "safe" to enable mid-training:
    - projection is zero-initialized so the module starts as identity
    - uses LayerNorm on the concatenated features
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model * 2)
        self.proj = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if x.shape != r.shape:
            raise ValueError(f"ConcatResidualFusion expects x and r same shape, got {tuple(x.shape)} vs {tuple(r.shape)}")
        h = torch.cat([x, r], dim=-1)
        return x + self.proj(self.norm(h))


class GlobalLocalConcatReplace(nn.Module):
    """Project concatenated (local, global) retrieval features to token features.

    This module is used by retrieval_topk/concat_replace to build content tokens
    purely from retrieved features (without adding back the original content tokens).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model * 2)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        if local_feat.shape != global_feat.shape:
            raise ValueError(
                "GlobalLocalConcatReplace expects local/global same shape, "
                f"got {tuple(local_feat.shape)} vs {tuple(global_feat.shape)}"
            )
        h = torch.cat([local_feat, global_feat], dim=-1)
        return self.proj(self.norm(h))


class FiLMDecoder(nn.Module):
    """FiLM-conditioned decoder: uses style vector to modulate per-token content before reconstruction."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, content_tokens: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        # content_tokens: (B,C,S,D), style_vec: (B,D)
        h = self.norm(content_tokens)
        gamma = self.gamma(style_vec).view(-1, 1, 1, h.size(-1))
        beta = self.beta(style_vec).view(-1, 1, 1, h.size(-1))
        h = h * (1.0 + gamma) + beta
        return self.proj(h)


class VectorQuantizer(nn.Module):
    """VQ-VAE style vector quantizer with straight-through estimator."""

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        beta: float = 0.25,
        *,
        compute_usage_loss: bool = False,
        usage_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.beta = float(beta)
        self.compute_usage_loss = bool(compute_usage_loss)
        self.usage_temperature = float(usage_temperature)
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize inputs.

        Args:
            z_e: (..., D)
        Returns:
            z_q: (..., D) quantized vectors (with straight-through)
            indices: (...) long
            vq_loss: scalar tensor
            usage_loss: scalar tensor (KL to uniform over codes; 0 is ideal)
        """
        if z_e.size(-1) != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {z_e.size(-1)}")

        flat = z_e.reshape(-1, self.dim)  # (N,D)
        codebook = self.embedding.weight  # (K,D)
        # Compute squared L2 distance: ||x||^2 + ||e||^2 - 2 x·e
        x2 = (flat ** 2).sum(dim=1, keepdim=True)  # (N,1)
        e2 = (codebook ** 2).sum(dim=1).unsqueeze(0)  # (1,K)
        distances = x2 + e2 - 2.0 * (flat @ codebook.t())  # (N,K)
        indices = torch.argmin(distances, dim=1)  # (N,)

        z_q = self.embedding(indices).view_as(z_e)
        # Losses: codebook + commitment
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        usage_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
        if self.compute_usage_loss:
            # Differentiable usage regularizer: KL( p(code) || Uniform )
            # p is approximated by soft assignments over the codebook.
            tau = max(self.usage_temperature, 1e-6)
            assign = torch.softmax(-distances / tau, dim=1)  # (N,K)
            p = assign.mean(dim=0).clamp_min(1e-12)  # (K,)
            usage_loss = (p * (p.log() + float(torch.log(torch.tensor(self.codebook_size, device=p.device, dtype=p.dtype))))).sum()

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.view(z_e.shape[:-1]), vq_loss, usage_loss


class TopKRetrievalCodebook(nn.Module):
    """Soft top-k retrieval from a learnable codebook.

    This is an attention-like read:
      - compute similarities between queries and K code vectors
      - take top-k per query
      - softmax over the top-k with temperature tau
      - return weighted sum of the selected code vectors

    Optionally computes a differentiable usage regularizer (KL to uniform) using
    the sparse top-k assignment weights.
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        *,
        topk: int = 4,
        tau: float = 0.1,
        compute_usage_loss: bool = False,
        usage_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.topk = int(topk)
        self.tau = float(tau)
        self.compute_usage_loss = bool(compute_usage_loss)
        self.usage_temperature = float(usage_temperature)

        self.embedding = nn.Parameter(torch.empty(self.codebook_size, self.dim))
        nn.init.uniform_(self.embedding, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve codebook vectors.

        Args:
            q: (..., D)
        Returns:
            r: (..., D) retrieved vectors
            topk_idx: (..., K') long, where K'=min(self.topk, codebook_size)
            topk_w: (..., K') float, soft weights that sum to 1
            top1_idx: (...) long, argmax code index (for logging)
            usage_loss: scalar tensor (KL(p(code)||Uniform); 0 is ideal)
        """
        if q.size(-1) != self.dim:
            raise ValueError(f"Expected q last dim {self.dim}, got {q.size(-1)}")

        flat_q = q.reshape(-1, self.dim)  # (N,D)

        # Under AMP, similarity computations can be too noisy in float16. Do the matching
        # in float32 (like we do for other numerically sensitive ops in this repo).
        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            q32 = flat_q.float()
            e32 = self.embedding.float()
            qn = F.normalize(q32, dim=1)
            en = F.normalize(e32, dim=1)  # (K,D)
            scores = qn @ en.t()  # (N,K) float32

        k = min(self.topk, self.codebook_size)
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)  # (N,k)
        tau = max(self.tau, 1e-6)
        topk_w = torch.softmax(topk_vals / tau, dim=1)  # (N,k) float32

        # Gather embeddings and weighted-sum.
        emb = self.embedding.index_select(0, topk_idx.reshape(-1)).view(-1, k, self.dim)  # (N,k,D)
        r = (topk_w.unsqueeze(-1) * emb.float()).sum(dim=1)  # (N,D) float32

        top1_idx = topk_idx[:, 0]  # top-k is returned sorted by score desc

        usage_loss = torch.zeros((), device=q.device, dtype=torch.float32)
        if self.compute_usage_loss:
            # Sparse approximation of p(code) using top-k assignment weights.
            # p_k = mean_n w_{n,k} over N queries.
            counts = torch.zeros((self.codebook_size,), device=q.device, dtype=torch.float32)
            counts.scatter_add_(0, topk_idx.reshape(-1), topk_w.reshape(-1).to(dtype=torch.float32))
            p = (counts / float(flat_q.size(0))).clamp_min(1e-12)
            usage_loss = (p * (p.log() + float(torch.log(torch.tensor(self.codebook_size, device=p.device, dtype=p.dtype))))).sum()

        r = r.to(dtype=q.dtype).view_as(q)
        return (
            r,
            topk_idx.view(*q.shape[:-1], k),
            topk_w.to(dtype=q.dtype).view(*q.shape[:-1], k),
            top1_idx.view(*q.shape[:-1]),
            usage_loss.to(dtype=q.dtype),
        )


class DisentangleV7Model(nn.Module):
    """Wrap an existing (backbone + classifier) model and expose content/style features.

    - content_tokens: token-wise dynamic features used for task prediction
    - style_tokens: token-wise style features (encouraged invariant within trial)
    - style_vec: pooled style embedding used for subject contrastive and FiLM reconstruction
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        patch_size: int,
        downstream_dataset: str,
        dropout: float = 0.0,
        style_dim: Optional[int] = None,
        enable_mid_mlp: bool = True,
        bypass_mid_mlp: bool = False,
        bypass_content_proj: bool = False,
        branch_impl: BranchImpl = "gaussian",
        gaussian_num_transition_components: int = 16,
        gaussian_transition_hidden: Optional[int] = None,
        gaussian_remove_gate_init: float = -2.0,
        use_content_codebook: bool = False,
        content_codebook_kind: ContentCodebookKind = "vq",
        content_codebook_fuse: ContentCodebookFuse = "delta",
        content_codebook_use_global: bool = True,
        content_codebook_use_local: bool = True,
        content_codebook_shared: bool = True,
        content_codebook_topk: int = 4,
        content_codebook_tau: float = 0.1,
        content_codebook_alpha_global: float = 0.1,
        content_codebook_alpha_local: float = 0.1,
        codebook_size: int = 32,
        vq_beta: float = 0.25,
        codebook_mode: CodebookMode = "window",
        task_use_quantized_content: bool = False,
        recon_use_quantized_content: bool = False,
        compute_vq_usage_loss: bool = False,
        vq_usage_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "backbone"):
            raise ValueError("base_model must expose attribute 'backbone'.")
        self.backbone = base_model.backbone
        self.downstream_dataset = str(downstream_dataset)
        self.task_head = getattr(base_model, "classifier", None)
        if self.task_head is None:
            self.task_head = getattr(base_model, "feed_forward", None)
        if self.task_head is None:
            raise ValueError("base_model must expose either 'classifier' or 'feed_forward'.")

        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.style_dim = int(style_dim or d_model)

        self.mid = TokenMLP(self.d_model, dropout=dropout) if enable_mid_mlp else nn.Identity()
        self.bypass_mid_mlp = bool(bypass_mid_mlp)
        self.bypass_content_proj = bool(bypass_content_proj)
        self.branch_impl: BranchImpl = str(branch_impl)  # type: ignore[assignment]
        self.content_proj: nn.Module
        self.style_proj: nn.Module
        self.gaussian_disentangler: Optional[GaussianBranchDisentangler]
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
            raise ValueError(f"Unsupported branch_impl: {self.branch_impl}")

        self.decoder = FiLMDecoder(d_model=self.d_model, out_dim=self.patch_size)

        self.use_content_codebook = bool(use_content_codebook)
        self.content_codebook_kind: ContentCodebookKind = str(content_codebook_kind)  # type: ignore[assignment]
        self.content_codebook_fuse: ContentCodebookFuse = str(content_codebook_fuse)  # type: ignore[assignment]
        self.content_codebook_use_global = bool(content_codebook_use_global)
        self.content_codebook_use_local = bool(content_codebook_use_local)
        self.content_codebook_shared = bool(content_codebook_shared)
        self.content_codebook_topk = int(content_codebook_topk)
        self.content_codebook_tau = float(content_codebook_tau)
        self.content_codebook_alpha_global = float(content_codebook_alpha_global)
        self.content_codebook_alpha_local = float(content_codebook_alpha_local)
        self.content_codebook_size = int(codebook_size)
        self.codebook_mode: CodebookMode = codebook_mode
        self.task_use_quantized_content = bool(task_use_quantized_content)
        self.recon_use_quantized_content = bool(recon_use_quantized_content)
        self.vq = None
        self.retrieval_cb_global: Optional[TopKRetrievalCodebook] = None
        self.retrieval_cb_local: Optional[TopKRetrievalCodebook] = None

        # Query projections for shared codebooks (identity init so enabling retrieval is non-destructive).
        self.retrieval_q_global = nn.Linear(self.d_model, self.d_model)
        self.retrieval_q_local = nn.Linear(self.d_model, self.d_model)
        nn.init.eye_(self.retrieval_q_global.weight)
        nn.init.eye_(self.retrieval_q_local.weight)
        if self.retrieval_q_global.bias is not None:
            nn.init.zeros_(self.retrieval_q_global.bias)
        if self.retrieval_q_local.bias is not None:
            nn.init.zeros_(self.retrieval_q_local.bias)

        # Optional fusion module (used for retrieval_topk when fuse_mode=concat_proj).
        self.cb_concat_fuse = ConcatResidualFusion(self.d_model)
        self.cb_gl_concat_replace = GlobalLocalConcatReplace(self.d_model)

        if self.use_content_codebook:
            if self.content_codebook_kind == "vq":
                self.vq = VectorQuantizer(
                    codebook_size=codebook_size,
                    dim=self.d_model,
                    beta=vq_beta,
                    compute_usage_loss=compute_vq_usage_loss,
                    usage_temperature=vq_usage_temperature,
                )
            elif self.content_codebook_kind == "retrieval_topk":
                # We optionally share a single codebook between global/local.
                cb = TopKRetrievalCodebook(
                    codebook_size=codebook_size,
                    dim=self.d_model,
                    topk=content_codebook_topk,
                    tau=content_codebook_tau,
                    compute_usage_loss=compute_vq_usage_loss,
                    usage_temperature=vq_usage_temperature,
                )
                if self.content_codebook_shared:
                    self.retrieval_cb_global = cb
                    self.retrieval_cb_local = cb
                else:
                    self.retrieval_cb_global = cb
                    self.retrieval_cb_local = TopKRetrievalCodebook(
                        codebook_size=codebook_size,
                        dim=self.d_model,
                        topk=content_codebook_topk,
                        tau=content_codebook_tau,
                        compute_usage_loss=compute_vq_usage_loss,
                        usage_temperature=vq_usage_temperature,
                    )
            else:
                raise ValueError(f"Unsupported content_codebook_kind: {self.content_codebook_kind}")

    def _prepare_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.downstream_dataset == "FACED":
            return x[:, :30, :, :]
        if self.downstream_dataset == "SHU-MI":
            return torch.cat((x[:, 0:16, :, :], x[:, 18:, :, :]), dim=1)
        if self.downstream_dataset == "MentalArithmetic":
            return x[:, :-1, :, :]
        return x

    def _apply_task_head(self, tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(self.task_head, nn.Sequential) and len(self.task_head) > 0:
            first = self.task_head[0]
            if isinstance(first, nn.Linear):
                logits = self.task_head(tokens.contiguous().view(tokens.size(0), -1))
            else:
                logits = self.task_head(tokens)
        else:
            logits = self.task_head(tokens)
        if logits.dim() == 2 and logits.shape[-1] == 1:
            logits = logits[:, 0]
        return logits

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_input = self._prepare_backbone_input(x)
        feats = self.backbone(backbone_input)  # (B,C,S,D)
        if feats.dim() == 2:
            feats = feats.unsqueeze(1).unsqueeze(1)
        elif feats.dim() == 3:
            feats = feats.unsqueeze(2)
        elif feats.dim() != 4:
            raise ValueError(f"Unsupported backbone output shape: {tuple(feats.shape)}")
        if not self.bypass_mid_mlp:
            feats = self.mid(feats)

        gaussian_remove_gate_12 = torch.empty((0,), device=x.device, dtype=feats.dtype)
        gaussian_remove_gate_23 = torch.empty((0,), device=x.device, dtype=feats.dtype)
        gaussian_remove_gate_34 = torch.empty((0,), device=x.device, dtype=feats.dtype)
        if self.branch_impl == "mlp":
            content_tokens_cont = feats if self.bypass_content_proj else self.content_proj(feats)
            style_tokens = self.style_proj(feats)
            # Window axis: reduce across channels first; treat seq_len as "window/time" axis.
            style_windows = style_tokens.mean(dim=1)  # (B,S,D)
            style_vec = style_windows.mean(dim=1)  # (B,D)
        elif self.branch_impl == "gaussian":
            if self.bypass_content_proj:
                # Warmup/task-only path: keep behavior as close as possible to vanilla backbone features.
                content_tokens_cont = feats
                style_tokens = feats
                style_windows = style_tokens.mean(dim=1)
                style_vec = style_windows.mean(dim=1)
            else:
                if self.gaussian_disentangler is None:
                    raise RuntimeError("branch_impl='gaussian' but gaussian_disentangler is not initialized.")
                g_out = self.gaussian_disentangler(feats)
                mixed_windows = g_out["mixed_windows"]  # (B,S,D)
                dynamic_windows = g_out["dynamic_windows"]  # (B,S,D)
                static_windows = g_out["static_windows"]  # (B,S,D)
                # Keep token-level channel details while replacing branch MLPs with Gaussian-inferred windows.
                content_tokens_cont = feats + (dynamic_windows - mixed_windows).unsqueeze(1)
                style_tokens = feats + (static_windows - mixed_windows).unsqueeze(1)
                style_windows = static_windows
                style_vec = style_windows.mean(dim=1)
                gaussian_remove_gate_12 = g_out["remove_gate_12"].reshape(1)
                gaussian_remove_gate_23 = g_out["remove_gate_23"].reshape(1)
                gaussian_remove_gate_34 = g_out["remove_gate_34"].reshape(1)
        else:
            raise ValueError(f"Unsupported branch_impl: {self.branch_impl}")

        content_tokens_q = content_tokens_cont
        vq_loss = None
        vq_usage_loss = None
        code_indices = None
        cb_global_top1: torch.Tensor = torch.empty((0,), device=x.device, dtype=torch.long)
        cb_local_top1: torch.Tensor = torch.empty((0,), device=x.device, dtype=torch.long)
        if self.use_content_codebook:
            if self.content_codebook_kind == "vq":
                if self.vq is None:
                    raise RuntimeError("use_content_codebook=True but VQ module is not initialised.")
                if self.codebook_mode == "window":
                    content_windows = content_tokens_cont.mean(dim=1)  # (B,S,D)
                    content_windows_q, code_indices, vq_loss, vq_usage_loss = self.vq(content_windows)
                    # Broadcast the window-level quantization back to tokens via an additive correction.
                    delta = (content_windows_q - content_windows).unsqueeze(1)  # (B,1,S,D)
                    content_tokens_q = content_tokens_cont + delta
                elif self.codebook_mode == "token":
                    z_q, code_indices, vq_loss, vq_usage_loss = self.vq(content_tokens_cont)  # (B,C,S,D)
                    content_tokens_q = z_q
                else:
                    raise ValueError(f"Unsupported codebook_mode: {self.codebook_mode}")
            elif self.content_codebook_kind == "retrieval_topk":
                cbg = self.retrieval_cb_global
                cbl = self.retrieval_cb_local
                if cbg is None or cbl is None:
                    raise RuntimeError("use_content_codebook=True but retrieval codebook is not initialised.")

                # Local queries: per-window pooled content (B,S,D).
                content_windows = content_tokens_cont.mean(dim=1)
                # Global query: pooled over windows -> (B,D).
                content_vec = content_windows.mean(dim=1)
                _, _, seq_len, _ = content_tokens_cont.shape

                deltas = []
                r_tokens = torch.zeros_like(content_tokens_cont)
                global_windows = torch.zeros(
                    (content_tokens_cont.size(0), seq_len, self.d_model),
                    device=content_tokens_cont.device,
                    dtype=content_tokens_cont.dtype,
                )
                local_windows = torch.zeros_like(global_windows)
                usage_losses = []

                if self.content_codebook_use_global:
                    qg = self.retrieval_q_global(content_vec)
                    rg, _, _, top1g, usage_g = cbg(qg)
                    cb_global_top1 = top1g.detach()
                    usage_losses.append(usage_g)
                    alpha_g = float(self.content_codebook_alpha_global)
                    global_windows = alpha_g * rg.unsqueeze(1).expand(-1, seq_len, -1)
                    if alpha_g != 0.0:
                        if self.content_codebook_fuse == "delta":
                            deltas.append(alpha_g * (rg - content_vec).view(-1, 1, 1, self.d_model))
                        elif self.content_codebook_fuse == "concat_proj":
                            r_tokens = r_tokens + alpha_g * rg.view(-1, 1, 1, self.d_model)
                        elif self.content_codebook_fuse == "concat_replace":
                            pass
                        else:
                            raise ValueError(f"Unsupported content_codebook_fuse: {self.content_codebook_fuse}")

                if self.content_codebook_use_local:
                    ql = self.retrieval_q_local(content_windows)
                    rl, _, _, top1l, usage_l = cbl(ql)
                    cb_local_top1 = top1l.detach()
                    usage_losses.append(usage_l)
                    alpha_l = float(self.content_codebook_alpha_local)
                    local_windows = alpha_l * rl
                    if alpha_l != 0.0:
                        if self.content_codebook_fuse == "delta":
                            deltas.append(alpha_l * (rl - content_windows).unsqueeze(1))  # (B,1,S,D)
                        elif self.content_codebook_fuse == "concat_proj":
                            r_tokens = r_tokens + alpha_l * rl.unsqueeze(1)  # (B,1,S,D)
                        elif self.content_codebook_fuse == "concat_replace":
                            pass
                        else:
                            raise ValueError(f"Unsupported content_codebook_fuse: {self.content_codebook_fuse}")

                if self.content_codebook_fuse == "delta":
                    content_tokens_q = content_tokens_cont + sum(deltas) if deltas else content_tokens_cont
                elif self.content_codebook_fuse == "concat_proj":
                    content_tokens_q = self.cb_concat_fuse(content_tokens_cont, r_tokens)
                elif self.content_codebook_fuse == "concat_replace":
                    # Build token features only from retrieved local/global features.
                    # local/global are window-level (B,S,D), so broadcast back to tokens via channel repeat.
                    fused_windows = self.cb_gl_concat_replace(local_windows, global_windows)  # (B,S,D)
                    content_tokens_q = fused_windows.unsqueeze(1).expand(-1, content_tokens_cont.size(1), -1, -1)
                else:
                    raise ValueError(f"Unsupported content_codebook_fuse: {self.content_codebook_fuse}")

                vq_loss = torch.zeros((), device=x.device, dtype=feats.dtype)
                if usage_losses:
                    usage_stack = torch.stack(
                        [torch.as_tensor(u, device=feats.device, dtype=feats.dtype) for u in usage_losses], dim=0
                    )
                    vq_usage_loss = usage_stack.sum()
                else:
                    vq_usage_loss = torch.zeros((), device=x.device, dtype=feats.dtype)
                code_indices = None
            else:
                raise ValueError(f"Unsupported content_codebook_kind: {self.content_codebook_kind}")

        task_logits_cont = self._apply_task_head(content_tokens_cont)
        task_logits_q = self._apply_task_head(content_tokens_q)
        task_logits = task_logits_q if (self.use_content_codebook and self.task_use_quantized_content) else task_logits_cont

        return {
            "backbone_input": backbone_input,
            "mixed_tokens": feats,
            "content_tokens": content_tokens_q if (self.use_content_codebook and self.task_use_quantized_content) else content_tokens_cont,
            "content_tokens_cont": content_tokens_cont,
            "content_tokens_q": content_tokens_q,
            "style_tokens": style_tokens,
            "style_windows": style_windows,
            "style_vec": style_vec,
            "task_logits": task_logits,
            "task_logits_cont": task_logits_cont,
            "task_logits_q": task_logits_q,
            "vq_loss": vq_loss if vq_loss is not None else torch.zeros((), device=x.device, dtype=feats.dtype),
            "vq_usage_loss": vq_usage_loss if vq_usage_loss is not None else torch.zeros((), device=x.device, dtype=feats.dtype),
            "vq_indices": code_indices if code_indices is not None else torch.empty((0,), device=x.device, dtype=torch.long),
            "cb_global_top1": cb_global_top1,
            "cb_local_top1": cb_local_top1,
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
        task_kind: str,
        label_smoothing: float,
        toggles: LossToggles,
        temperature: float,
        var_ratio_r: float,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        logits = outputs["task_logits"]
        if task_kind == "multiclass":
            if logits.dim() == 3 and labels.dim() == 2 and tuple(logits.shape[:2]) == tuple(labels.shape):
                losses["task"] = F.cross_entropy(logits.transpose(1, 2), labels, label_smoothing=float(label_smoothing))
            else:
                losses["task"] = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
        elif task_kind == "binary":
            losses["task"] = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))
        else:
            raise ValueError(f"Unsupported task_kind={task_kind!r} (V7 disentangle backend disables regression).")

        if toggles.use_style_contrastive:
            info_nce = InfoNCELoss(temperature=temperature).to(logits.device)
            losses["style_cl"] = info_nce(outputs["style_vec"], subject_ids.view(-1))

        if toggles.use_style_invariance:
            # Encourage style to be invariant across windows in the same trial.
            style_windows = outputs["style_windows"]  # (B,S,D)
            style_vec = outputs["style_vec"].unsqueeze(1)  # (B,1,D)
            losses["style_invar"] = (style_windows - style_vec).pow(2).mean()

        if toggles.use_var_ratio:
            # Encourage content to vary more than style across windows in the same trial (hinge ratio).
            content_windows = outputs["content_tokens"].mean(dim=1)  # (B,S,D)
            style_windows = outputs["style_windows"]  # (B,S,D)
            v_c = _mpd(content_windows)
            v_s = _mpd(style_windows)
            r = float(var_ratio_r)
            losses["var_ratio"] = torch.clamp(1.0 - (v_c / (r * v_s + 1e-8)), min=0.0).mean()

        if toggles.use_film_reconstruction:
            content_for_recon = outputs["content_tokens_q"] if self.recon_use_quantized_content else outputs["content_tokens_cont"]
            recon = self.decoder(content_for_recon, outputs["style_vec"])
            losses["recon"] = F.mse_loss(recon, outputs["backbone_input"])

        if toggles.use_content_codebook:
            losses["vq"] = outputs["vq_loss"]
            losses["vq_usage"] = outputs["vq_usage_loss"]

        return losses
