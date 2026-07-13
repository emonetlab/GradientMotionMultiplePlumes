"""Input adapters for the public Figure 1 plume movies."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Sequence

import numpy as np


SMOOTH_SOURCE_SHAPE = (406, 216)
SMOOTH_CROPPED_SHAPE = (224, 351)
SMOOTH_TARGET_SHAPE = (1088, 1696)


def crop_and_pad_smooth_frame(frame: np.ndarray) -> np.ndarray:
    """Apply the published notebook's Dryad-to-arena crop and zero padding.

    Dryad frames arrive downwind-by-crosswind, ``(406, 216)``.  After
    transposition, the first 351 downwind columns are retained and four zero
    rows are added to each crosswind edge.
    """

    frame = np.asarray(frame)
    if frame.shape == SMOOTH_SOURCE_SHAPE:
        frame = frame.T
    if frame.shape != (216, 406):
        raise ValueError(
            f"Expected a Dryad smooth-plume frame shaped {SMOOTH_SOURCE_SHAPE} "
            f"or {(216, 406)}; got {frame.shape}."
        )
    cropped = frame[:, :351]
    output = np.pad(cropped, ((4, 4), (0, 0)), mode="constant", constant_values=0)
    if output.shape != SMOOTH_CROPPED_SHAPE:
        raise AssertionError(f"Unexpected crop/pad result: {output.shape}")
    return output


def prepare_smooth_frame(
    frame: np.ndarray,
    *,
    target_shape: tuple[int, int] = SMOOTH_TARGET_SHAPE,
) -> np.ndarray:
    """Resize and quantize a public Dryad frame like the conversion notebook."""

    try:
        from skimage.transform import resize
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Smooth-plume conversion requires scikit-image.") from exc
    prepared = resize(crop_and_pad_smooth_frame(frame), target_shape, order=2)
    prepared = np.nan_to_num(prepared, nan=0.0, posinf=1.0, neginf=0.0)
    return np.floor(np.clip(prepared, 0.0, 1.0) * 255.0).astype(np.uint8)


class SmoothTemporalUpsample(Sequence[np.ndarray]):
    """Expose a 15-Hz Dryad movie as the 60-Hz legacy/corrected sequence."""

    def __init__(
        self,
        source: Sequence[np.ndarray],
        *,
        profile: Literal["notebook_legacy", "corrected"] = "notebook_legacy",
        target_shape: tuple[int, int] = SMOOTH_TARGET_SHAPE,
    ) -> None:
        if len(source) < 2:
            raise ValueError(
                "Smooth-plume upsampling requires at least two source frames."
            )
        if profile not in {"notebook_legacy", "corrected"}:
            raise ValueError(f"Unknown smooth temporal profile: {profile}")
        self.source = source
        self.profile = profile
        self.target_shape = target_shape

    def __len__(self) -> int:
        # Both profiles have 4*(N-1)+1 frames, but align them differently.
        return 4 * (len(self.source) - 1) + 1

    @lru_cache(maxsize=8)
    def _source_frame(self, index: int) -> np.ndarray:
        return prepare_smooth_frame(self.source[index], target_shape=self.target_shape)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        if isinstance(index, slice):
            return np.asarray([self[i] for i in range(*index.indices(len(self)))])
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        if self.profile == "notebook_legacy":
            if index == 0:
                return self._source_frame(0)
            block, phase = divmod(index - 1, 4)
        else:
            block, phase = divmod(index, 4)
            if block == len(self.source) - 1:
                return self._source_frame(block)

        first = self._source_frame(block)
        if phase == 0:
            return first
        following = self._source_frame(block + 1)
        alpha = phase / 4.0
        # cv2.addWeighted on uint8 produces a rounded, saturated uint8 image.
        blended = (1.0 - alpha) * first.astype(np.float32) + alpha * following.astype(
            np.float32
        )
        return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)


def describe_smooth_profile(profile: str) -> dict[str, Any]:
    if profile == "notebook_legacy":
        return {
            "profile": profile,
            "description": "Duplicates source frame 0 and omits the final source frame, matching the committed notebook.",
            "source_fps": 15,
            "target_fps": 60,
        }
    if profile == "corrected":
        return {
            "profile": profile,
            "description": "Linear timestamps from the first through final source frame without endpoint duplication.",
            "source_fps": 15,
            "target_fps": 60,
        }
    raise ValueError(f"Unknown smooth temporal profile: {profile}")
