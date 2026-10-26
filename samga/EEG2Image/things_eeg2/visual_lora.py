"""Live OpenCLIP ViT-B/16 features with optional low-rank visual adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from .extract_vit_b16_features import ZippedImages


@dataclass(frozen=True)
class VisualLoRAConfig:
    """Visual adaptation settings; stage is coarse, fine, or both.

    Block numbers are one-based. Q/V adapters are always enabled; optional
    switches add K, attention-output/MLP adapters, or trainable LayerNorms.
    alpha/rank scales updates; max_update_ratio imposes a separate norm cap."""
    enabled: bool = False
    require_cuda: bool = True
    stage: str = "fine"
    checkpoint: Path = Path("models/ViT-B-16-laion2b/open_clip_model.safetensors")
    rank: int = 8
    alpha: float = 16.0
    start_block: int = 6
    blocks: tuple[int, ...] | None = None
    max_update_ratio: float | None = None
    tune_layer_norm: bool = False
    lora_key: bool = False
    lora_output: bool = False
    lora_mlp: bool = False
    images_per_batch: int = 16
    gradient_cache_microbatch_size: int | None = None
    evaluation_image_batch_size: int = 32
    learning_rate: float = 1e-4
    coarse_learning_rate: float | None = None
    gradient_checkpointing: bool = True
    pooling: str = "cls"

    def selected_blocks(self, layer_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Validate settings and return blocks no deeper than the last extracted feature."""
        if self.stage not in {"coarse", "fine", "both"}:
            raise ValueError("LoRA stage must be 'coarse', 'fine', or 'both'")
        if not layer_ids or not 1 <= min(layer_ids) <= max(layer_ids) <= 12:
            raise ValueError("Selected ViT feature layers must be within blocks 1-12")
        if self.rank < 1 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.max_update_ratio is not None and self.max_update_ratio <= 0:
            raise ValueError("max_update_ratio must be positive or None")
        if self.pooling not in {"cls", "patch_mean"}:
            raise ValueError("LoRA pooling must be cls or patch_mean")
        if self.images_per_batch < 1 or self.evaluation_image_batch_size < 1:
            raise ValueError("LoRA image batch sizes must be positive")
        if self.gradient_cache_microbatch_size is not None and self.gradient_cache_microbatch_size < 1:
            raise ValueError("gradient_cache_microbatch_size must be positive or None")
        if self.learning_rate <= 0:
            raise ValueError("LoRA learning_rate must be positive")
        if self.coarse_learning_rate is not None and self.coarse_learning_rate <= 0:
            raise ValueError("LoRA coarse_learning_rate must be positive or None")
        if self.blocks is None:
            if not 1 <= self.start_block <= max(layer_ids):
                raise ValueError("LoRA start_block must be within selected ViT blocks")
            return tuple(range(self.start_block, max(layer_ids) + 1))
        selected = tuple(self.blocks)
        if (
            not selected
            or selected != tuple(sorted(set(selected)))
            or selected[0] < 1
            or selected[-1] > max(layer_ids)
        ):
            raise ValueError(
                "LoRA blocks must be a nonempty, ascending list of distinct "
                "1-based blocks no deeper than the deepest selected feature layer"
            )
        return selected


class FusedQvLoRA(nn.Module):
    """Parametrize fused Q/K/V weights with low-rank Q/V and optional K updates."""

    def __init__(
        self, width: int, rank: int, alpha: float, max_update_ratio: float | None = None,
        include_key: bool = False,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if max_update_ratio is not None and max_update_ratio <= 0:
            raise ValueError("max_update_ratio must be positive or None")
        self.scale = alpha / rank
        self.max_update_ratio = max_update_ratio
        self.include_key = include_key
        self.q_down = nn.Parameter(torch.empty(rank, width))
        self.q_up = nn.Parameter(torch.zeros(width, rank))
        if include_key:
            self.k_down = nn.Parameter(torch.empty(rank, width))
            self.k_up = nn.Parameter(torch.zeros(width, rank))
        self.v_down = nn.Parameter(torch.empty(rank, width))
        self.v_up = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.q_down, a=5**0.5)
        if include_key:
            nn.init.kaiming_uniform_(self.k_down, a=5**0.5)
        nn.init.kaiming_uniform_(self.v_down, a=5**0.5)

    def forward(self, weight: Tensor) -> Tensor:
        q = (self.q_up @ self.q_down) * self.scale
        v = (self.v_up @ self.v_down) * self.scale
        k = (self.k_up @ self.k_down) * self.scale if self.include_key else torch.zeros_like(q)
        if self.max_update_ratio is not None:


            q_limit = self.max_update_ratio * torch.linalg.vector_norm(weight[: q.shape[0]].float())
            v_limit = self.max_update_ratio * torch.linalg.vector_norm(weight[-v.shape[0] :].float())
            q = q * (q_limit / torch.linalg.vector_norm(q.float()).clamp_min(1e-12)).clamp(max=1)
            if self.include_key:
                k_limit = self.max_update_ratio * torch.linalg.vector_norm(weight[q.shape[0] : 2 * q.shape[0]].float())
                k = k * (k_limit / torch.linalg.vector_norm(k.float()).clamp_min(1e-12)).clamp(max=1)
            v = v * (v_limit / torch.linalg.vector_norm(v.float()).clamp_min(1e-12)).clamp(max=1)
        return weight + torch.cat((q, k, v), dim=0)


class LowRankWeightUpdate(nn.Module):
    """Parametrize a linear weight with an initially zero low-rank update."""

    def __init__(
        self, out_features: int, in_features: int, rank: int, alpha: float,
        max_update_ratio: float | None,
    ) -> None:
        super().__init__()
        self.scale = alpha / rank
        self.max_update_ratio = max_update_ratio
        self.down = nn.Parameter(torch.empty(rank, in_features))
        self.up = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=5**0.5)

    def forward(self, weight: Tensor) -> Tensor:
        delta = (self.up @ self.down) * self.scale
        if self.max_update_ratio is not None:
            limit = self.max_update_ratio * torch.linalg.vector_norm(weight.float())
            delta = delta * (
                limit / torch.linalg.vector_norm(delta.float()).clamp_min(1e-12)
            ).clamp(max=1)
        return weight + delta


def attach_vit_adapters(
    blocks: nn.ModuleList | nn.Sequential, lora_blocks: tuple[int, ...],
    config: VisualLoRAConfig,
) -> None:
    """Attach trainable adapters to selected blocks of an otherwise frozen ViT."""
    for block_number in lora_blocks:
        block = blocks[block_number - 1]
        attention = block.attn
        parametrize.register_parametrization(
            attention,
            "in_proj_weight",
            FusedQvLoRA(
                attention.embed_dim, config.rank, config.alpha,
                config.max_update_ratio, include_key=config.lora_key,
            ),
        )
        if config.lora_output:
            parametrize.register_parametrization(
                attention.out_proj, "weight",
                LowRankWeightUpdate(
                    attention.out_proj.out_features, attention.out_proj.in_features,
                    config.rank, config.alpha, config.max_update_ratio,
                ),
            )
        if config.lora_mlp:
            for linear in (block.mlp.c_fc, block.mlp.c_proj):
                parametrize.register_parametrization(
                    linear, "weight",
                    LowRankWeightUpdate(
                        linear.out_features, linear.in_features,
                        config.rank, config.alpha, config.max_update_ratio,
                    ),
                )
        if config.tune_layer_norm:
            for norm in (block.ln_1, block.ln_2):
                norm.requires_grad_(True)


class LiveViTFeatures:
    """Encode unique stimulus images once and reuse features across EEG subjects."""
    def __init__(
        self,
        config: VisualLoRAConfig,
        dataset_root: Path,
        layer_ids: tuple[int, ...],
        device: torch.device,
    ) -> None:
        if not config.checkpoint.is_file():
            raise FileNotFoundError(f"ViT checkpoint missing: {config.checkpoint}")
        lora_blocks = config.selected_blocks(layer_ids)
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("LoRA requires open_clip_torch in the training environment") from exc
        clip, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=str(config.checkpoint), device="cpu"
        )
        self.visual = clip.visual
        del clip
        self.visual.requires_grad_(False)
        blocks = self.visual.transformer.resblocks
        if len(blocks) != 12 or self.visual.class_embedding.numel() != 768:
            raise ValueError("Checkpoint must contain a 12-block, width-768 ViT-B/16")
        attach_vit_adapters(blocks, lora_blocks, config)
        if config.gradient_checkpointing:
            self.visual.set_grad_checkpointing(True)
        self.visual.to(device)
        self.train_images = ZippedImages(dataset_root, "train", preprocess)
        self.test_images = ZippedImages(dataset_root, "test", preprocess)
        self.config = config
        self.lora_blocks = lora_blocks
        self.layer_ids = layer_ids
        self.device = device

    def close(self) -> None:
        self.train_images.close()
        self.test_images.close()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.visual.parameters() if p.requires_grad]

    def adapter_state_dict(self) -> dict[str, Tensor]:
        """Save only trainable adapters and optional LayerNorm parameters."""
        trainable_names = {
            name for name, parameter in self.visual.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.visual.state_dict().items()
            if name in trainable_names
        }

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = set(self.adapter_state_dict())
        if set(state) != expected:
            raise ValueError(
                f"LoRA checkpoint mismatch: missing={sorted(expected - set(state))}, "
                f"extra={sorted(set(state) - expected)}"
            )
        self.visual.load_state_dict(state, strict=False)

    def encode(self, indices: list[int], split: str) -> Tensor:
        """Return [images, selected_layers, 768] class-token or mean-patch features."""
        images = self.train_images if split == "train" else self.test_images
        pixels = torch.stack([images[index] for index in indices]).to(self.device)
        result = self.visual.forward_intermediates(
            pixels,
            indices=[layer - 1 for layer in self.layer_ids],
            stop_early=True,
            intermediates_only=True,
            output_fmt="NLC",
            output_extra_tokens=True,
        )
        if self.config.pooling == "cls":
            layers = [item[:, 0] for item in result["image_intermediates_prefix"]]
        else:
            layers = [item.mean(dim=1) for item in result["image_intermediates"]]
        return torch.stack(layers, dim=1)

    def encode_gradient_cached(self, indices: list[int], *, amp_enabled: bool) -> Tensor:
        """Encode image chunks without a ViT graph; retain a full-batch feature leaf.

        The loss keeps all contrastive/MMD candidates. After loss backward,
        backward_gradient_cached replays image chunks with the feature gradients."""
        size = self.config.gradient_cache_microbatch_size
        if size is None:
            raise ValueError("Gradient caching requires a microbatch size")
        self.visual.eval()  # Both feature passes must use identical dropout behavior.
        pieces: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, len(indices), size):
                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    piece = self.encode(indices[start : start + size], "train")
                pieces.append(piece.float())
        return torch.cat(pieces, dim=0).detach().requires_grad_(True)

    def backward_gradient_cached(
        self, indices: list[int], feature_grad: Tensor, *, amp_enabled: bool
    ) -> None:
        """Replay image chunks and propagate the already scaled feature gradients."""
        size = self.config.gradient_cache_microbatch_size
        if size is None:
            raise ValueError("Gradient caching requires a microbatch size")
        if feature_grad.shape[0] != len(indices):
            raise ValueError("Feature-gradient count does not match image count")
        for start in range(0, len(indices), size):
            stop = min(start + size, len(indices))
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                features = self.encode(indices[start:stop], "train").float()
            torch.autograd.backward(features, grad_tensors=feature_grad[start:stop])

    @torch.no_grad()
    def test_feature_bank(self, *, amp_enabled: bool) -> Tensor:
        self.visual.eval()
        chunks: list[Tensor] = []
        for begin in range(0, len(self.test_images), self.config.evaluation_image_batch_size):
            end = min(begin + self.config.evaluation_image_batch_size, len(self.test_images))
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                features = self.encode(list(range(begin, end)), "test")
            chunks.append(features.float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def support_feature_bank(self, indices: list[int], *, amp_enabled: bool) -> Tensor:
        """Encode labeled training-concept images with the current visual weights."""
        self.visual.eval()
        chunks: list[Tensor] = []
        size = self.config.evaluation_image_batch_size
        for begin in range(0, len(indices), size):
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                features = self.encode(indices[begin:begin + size], "train")
            chunks.append(features.float().cpu())
        return torch.cat(chunks, dim=0)
