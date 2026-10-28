"""Image-only initial-angle and layer-sequence regression models.

Angles are unrestricted scalar outputs in degrees.  The models never receive MC
positions, MC segment targets, initial truth angles, or MC validity masks.
``layer_mask`` is an observed-signal mask, not a truth-availability mask.
"""

from contextlib import nullcontext
from typing import Mapping, Optional, Sequence

import torch
from torch import nn


MODEL_NAMES = ("peak_cnn", "resnet_initial", "resnet_sequence")


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _initialize_regression_output(layer: nn.Linear) -> None:
    # Small, nonzero output weights keep initial predictions near zero while
    # allowing gradients to reach the encoder on the first optimizer step.
    nn.init.normal_(layer.weight, std=0.001)
    nn.init.zeros_(layer.bias)


def _normalization(channels: int) -> nn.GroupNorm:
    groups = next(group for group in (8, 4, 2, 1) if channels % group == 0)
    return nn.GroupNorm(groups, channels)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over observed layers, returning zero for an all-missing event."""
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _full_precision_recurrence(device: torch.device):
    """Keep recurrent CUDA kernels compatible with older cuDNN AMP support."""
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", enabled=False)
    return torch.cuda.amp.autocast(enabled=False)


class PeakCNN(nn.Module):
    """Historical 1->16->32->64 CNN topology (121,729 parameters).

    The historical image-only baseline intentionally does not use ``layer_mask``.
    For the default 12x74 image its flattened pooled feature has 1,536 values.
    """

    def __init__(self, n_layers: int = 12, n_pads: int = 74) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_pads = int(n_pads)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 3 * 8, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.apply(_initialize_module)
        _initialize_regression_output(self.head[-1])

    def forward(
        self, image: torch.Tensor, layer_mask: Optional[torch.Tensor] = None
    ) -> dict:
        _validate_image(image, self.n_layers, self.n_pads)
        if layer_mask is not None and layer_mask.shape != image.shape[:1] + (self.n_layers,):
            raise ValueError("layer_mask must have shape [batch, n_layers]")
        return {"initial": self.head(self.features(image)).squeeze(-1), "segment": None}


class BasicResidualBlock(nn.Module):
    """Two 3x3 convolutions; spatial reduction is along pad x only."""

    def __init__(self, in_channels: int, out_channels: int, x_stride: int = 1) -> None:
        super().__init__()
        stride = (1, int(x_stride))
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            _normalization(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _normalization(out_channels),
        )
        if in_channels != out_channels or x_stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _normalization(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(value) + self.shortcut(value))


def _validate_image(image: torch.Tensor, n_layers: int, n_pads: int) -> None:
    if image.ndim != 4 or tuple(image.shape[1:]) != (1, n_layers, n_pads):
        raise ValueError(
            "Expected image [batch, 1, %d, %d], got %s"
            % (n_layers, n_pads, tuple(image.shape))
        )
    if not image.is_floating_point():
        raise TypeError("image must be floating point, not raw integer ADC")


class LayerPreservingResNet(nn.Module):
    """Small residual CNN + bidirectional GRU, with optional segment head.

    Default shapes: [B,1,12,74] -> [B,96,12,19] -> [B,12,128]
    -> append normalized geometric y and observed mask -> [B,12,130]
    -> bidirectional GRU -> [B,12,192].

    ``multitask=False`` and ``multitask=True`` have identical shared encoders,
    GRU and initial head.  With the same random seed their shared initial
    parameters are identical; only the latter creates a segment-regression head.
    This is an offline full-event model, not a causal online Kalman filter.
    """

    def __init__(
        self,
        multitask: bool = True,
        n_layers: int = 12,
        n_pads: int = 74,
        resnet_widths: Sequence[int] = (32, 64, 96),
        embedding_dim: int = 128,
        gru_hidden: int = 96,
        dropout: float = 0.1,
        layer_y_mm: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_pads = int(n_pads)
        self.multitask = bool(multitask)
        widths = tuple(int(width) for width in resnet_widths)
        if self.n_layers < 2 or self.n_pads < 1:
            raise ValueError("ResNet sequence models need >=2 layers and >=1 pad")
        if len(widths) != 3 or min(widths) < 1:
            raise ValueError("resnet_widths must contain three positive channel widths")
        if int(embedding_dim) < 1 or int(gru_hidden) < 1:
            raise ValueError("embedding_dim and gru_hidden must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        if layer_y_mm is None:
            y_mm = torch.arange(self.n_layers, dtype=torch.float32) * 12.0
        else:
            y_mm = torch.as_tensor(layer_y_mm, dtype=torch.float32)
        if y_mm.shape != (self.n_layers,) or not torch.isfinite(y_mm).all():
            raise ValueError("layer_y_mm must contain one finite center y per layer")
        if not torch.all(y_mm[1:] > y_mm[:-1]):
            raise ValueError("layer_y_mm must be strictly increasing in sequence order")
        y_normalized = (y_mm - y_mm.mean()) / ((y_mm[-1] - y_mm[0]) * 0.5)
        self.register_buffer("layer_y_normalized", y_normalized)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, padding=1, bias=False),
            _normalization(widths[0]),
            nn.ReLU(inplace=True),
            BasicResidualBlock(widths[0], widths[0]),
            BasicResidualBlock(widths[0], widths[0]),
            BasicResidualBlock(widths[0], widths[1], x_stride=2),
            BasicResidualBlock(widths[1], widths[1]),
            BasicResidualBlock(widths[1], widths[2], x_stride=2),
            BasicResidualBlock(widths[2], widths[2]),
        )
        x_after_stride = ((self.n_pads + 1) // 2 + 1) // 2
        self.layer_embedding = nn.Sequential(
            nn.Linear(widths[-1] * x_after_stride, int(embedding_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.gru = nn.GRU(
            input_size=int(embedding_dim) + 2,
            hidden_size=int(gru_hidden),
            batch_first=True,
            bidirectional=True,
        )
        self.initial_head = nn.Sequential(
            nn.Linear(2 * int(gru_hidden), int(gru_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gru_hidden), 1),
        )
        # Initialize shared modules before constructing the optional head, so
        # initial-only/multitask comparisons start from the same shared weights.
        # GRU keeps PyTorch's recurrent initialization (not ReLU Kaiming).
        self.apply(_initialize_module)
        _initialize_regression_output(self.initial_head[-1])

        self.segment_head = None
        if self.multitask:
            self.segment_head = nn.Sequential(
                nn.Linear(4 * int(gru_hidden), int(gru_hidden)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(gru_hidden), 1),
            )
            self.segment_head.apply(_initialize_module)
            _initialize_regression_output(self.segment_head[-1])

    def encode_layers(
        self, image: torch.Tensor, layer_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        _validate_image(image, self.n_layers, self.n_pads)
        if layer_mask is None:
            layer_mask = image.new_ones((image.shape[0], self.n_layers))
        elif layer_mask.shape != image.shape[:1] + (self.n_layers,):
            raise ValueError("layer_mask must have shape [batch, n_layers]")
        observed = layer_mask.to(device=image.device, dtype=image.dtype)
        features = self.encoder(image)
        # Preserve both the layer axis and the pad-x ordering within each layer.
        features = features.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)
        embedded = self.layer_embedding(features)
        geometric_y = self.layer_y_normalized.to(dtype=image.dtype)
        geometric_y = geometric_y.view(1, self.n_layers, 1).expand(image.shape[0], -1, -1)
        recurrent_input = torch.cat((embedded, geometric_y, observed.unsqueeze(-1)), dim=-1)
        # Some older cuDNN versions reject BF16 GRU even when convolution AMP
        # works.  Keep FP32 recurrent weights and input, while the surrounding
        # residual encoder and output heads remain eligible for CUDA autocast.
        with _full_precision_recurrence(image.device):
            if image.device.type == "cuda":
                recurrent_input = recurrent_input.float()
            encoded, _ = self.gru(recurrent_input)
        return encoded, observed

    def forward(
        self, image: torch.Tensor, layer_mask: Optional[torch.Tensor] = None
    ) -> dict:
        encoded, observed = self.encode_layers(image, layer_mask)
        initial = self.initial_head(masked_mean(encoded, observed)).squeeze(-1)
        segment = None
        if self.segment_head is not None:
            adjacent = torch.cat((encoded[:, :-1], encoded[:, 1:]), dim=-1)
            segment = self.segment_head(adjacent).squeeze(-1)
        return {"initial": initial, "segment": segment}


def build_model(name: str, config: Optional[Mapping] = None) -> nn.Module:
    """Build a named architecture from its JSON-serializable model settings.

    Extra settings are ignored so an enclosing experiment configuration can also
    be supplied.  Save this configuration with the state_dict for reconstruction.
    """
    cfg = dict(config or {})
    common = {key: cfg[key] for key in ("n_layers", "n_pads") if key in cfg}
    if name == "peak_cnn":
        return PeakCNN(**common)
    if name not in MODEL_NAMES:
        raise ValueError("Unknown model %r; choose one of %s" % (name, MODEL_NAMES))
    allowed = ("resnet_widths", "embedding_dim", "gru_hidden", "dropout", "layer_y_mm")
    kwargs = {key: cfg[key] for key in allowed if key in cfg}
    return LayerPreservingResNet(multitask=name == "resnet_sequence", **common, **kwargs)
