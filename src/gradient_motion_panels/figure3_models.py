"""Checkpoint-compatible Figure 3 networks and deterministic probe stimuli.

PyTorch is deliberately optional.  Importing this module succeeds without it;
constructing a network gives a focused dependency error.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Figure 3 checkpoint evaluation requires PyTorch.") from exc
    return torch, nn


def make_minimal_network(*, channels: int = 1, time_steps: int = 30) -> Any:
    """Build the preprint's antisymmetric minimal network."""

    torch, nn = _torch_modules()

    class MinimalNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_layer = nn.Conv2d(time_steps, channels, (1, 2))

        def forward(self, input_data: Any) -> Any:
            plus = torch.relu(self.first_layer(input_data))
            minus = torch.relu(self.first_layer(torch.flip(input_data, [-1])))
            opponent = (plus - minus).view(-1, channels).mean(-1)
            return torch.stack((opponent / 2.0, -opponent / 2.0), dim=1)

    return MinimalNetwork()


def make_dense_network(
    *, time_steps: int = 30, widths: tuple[int, ...] = (20, 20)
) -> Any:
    """Build the checkpoint-compatible 60-20-20-2 dense opponent network."""

    torch, nn = _torch_modules()

    class DenseNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer_widths = (2 * time_steps, *widths, 2)
            self.dense_layers = nn.ModuleList(
                [
                    nn.Linear(layer_widths[i], layer_widths[i + 1])
                    for i in range(len(layer_widths) - 1)
                ]
            )

        def forward_module(self, input_data: Any) -> Any:
            output = input_data.clone()
            # The final ReLU is intentional: it matches the archived model code.
            for layer in self.dense_layers:
                output = torch.relu(layer(output))
            return output

        def forward(self, input_data: Any) -> Any:
            direct = self.forward_module(input_data)
            flipped = torch.cat(
                (input_data[:, time_steps:], input_data[:, :time_steps]), dim=1
            )
            return direct - self.forward_module(flipped)

    return DenseNetwork()


def class_one_probability(logits: Any) -> np.ndarray:
    """Convert legacy two-logit output to class-one probability."""

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits shaped (N, 2); got {logits.shape}.")
    difference = logits[:, 1] - logits[:, 0]
    return 1.0 / (1.0 + np.exp(-difference))


def gradient_probe(
    differences: np.ndarray | None = None, *, time_steps: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Constant bilateral signals used for Figure 3 D/I."""

    if differences is None:
        differences = np.linspace(-0.5, 0.5, 11)
    differences = np.asarray(differences, dtype=np.float32)
    samples = np.zeros((len(differences), time_steps, 1, 2), dtype=np.float32)
    samples[..., 0] = 0.25 + differences[:, None, None] / 2.0
    samples[..., 1] = 0.25 - differences[:, None, None] / 2.0
    return differences, samples


def motion_probe(
    *,
    sample_count: int = 100_000,
    max_shift: int = 6,
    time_steps: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Binary bilateral motion stimuli for shifts ``-max_shift..max_shift``.

    The preprint notebook used values 0 and 2 and did not set a seed.  This
    implementation preserves the signal levels but makes the Monte Carlo probe
    repeatable.
    """

    if sample_count < 1 or max_shift < 0:
        raise ValueError("sample_count must be positive and max_shift non-negative.")
    rng = np.random.default_rng(seed)
    shifts = np.arange(-max_shift, max_shift + 1)
    stimuli: list[np.ndarray] = []
    for signed_shift in shifts:
        delay = abs(int(signed_shift))
        raw = (rng.random((sample_count, time_steps + max_shift + 14)) > 0.5).astype(
            np.float32
        ) * 2.0
        sample = np.zeros((sample_count, time_steps, 1, 2), dtype=np.float32)
        if signed_shift >= 0:
            sample[:, :, 0, 0] = raw[:, delay : delay + time_steps]
            sample[:, :, 0, 1] = raw[:, :time_steps]
        else:
            sample[:, :, 0, 1] = raw[:, delay : delay + time_steps]
            sample[:, :, 0, 0] = raw[:, :time_steps]
        stimuli.append(sample)
    return shifts / 60.0, stimuli


def iter_motion_probes(
    *,
    sample_count: int = 100_000,
    max_shift: int = 6,
    time_steps: int = 30,
    seed: int = 0,
) -> Any:
    """Yield ``(shift_seconds, samples)`` without retaining all probe arrays."""

    if sample_count < 1 or max_shift < 0:
        raise ValueError("sample_count must be positive and max_shift non-negative.")
    rng = np.random.default_rng(seed)
    for signed_shift in range(-max_shift, max_shift + 1):
        delay = abs(signed_shift)
        raw = (rng.random((sample_count, time_steps + max_shift + 14)) > 0.5).astype(
            np.float32
        ) * 2.0
        sample = np.zeros((sample_count, time_steps, 1, 2), dtype=np.float32)
        if signed_shift >= 0:
            sample[:, :, 0, 0] = raw[:, delay : delay + time_steps]
            sample[:, :, 0, 1] = raw[:, :time_steps]
        else:
            sample[:, :, 0, 1] = raw[:, delay : delay + time_steps]
            sample[:, :, 0, 0] = raw[:, :time_steps]
        yield signed_shift / 60.0, sample


def dense_input(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 4 or samples.shape[2:] != (1, 2):
        raise ValueError(f"Expected samples shaped (N, T, 1, 2); got {samples.shape}.")
    return np.concatenate((samples[:, :, 0, 0], samples[:, :, 0, 1]), axis=1)
