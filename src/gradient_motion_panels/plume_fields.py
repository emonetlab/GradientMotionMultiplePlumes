"""Figure 1 plume-gradient and motion-cue computations.

The functions in this module are a small, testable replacement for the
notebook loops used for Figure 1 in the preprint-era OdorMotionMLdev code.
Frames are indexed as ``(time, y, x)``.  The crosswind gradient is therefore
the finite difference along image axis 0.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


SUMMARY_ARRAYS = (
    "snapshot",
    "gradient_snapshot",
    "motion_snapshot",
    "gradient_mean",
    "gradient_std",
    "motion_mean",
    "motion_std",
)


@dataclass(frozen=True)
class PlumeSummary:
    """Compact data needed to render Figure 1 B/C and E--L."""

    snapshot: np.ndarray
    gradient_snapshot: np.ndarray
    motion_snapshot: np.ndarray
    gradient_mean: np.ndarray
    gradient_std: np.ndarray
    motion_mean: np.ndarray
    motion_std: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        shapes = {np.asarray(getattr(self, name)).shape for name in SUMMARY_ARRAYS}
        if len(shapes) != 1:
            raise ValueError(
                f"All summary arrays must have one shape; got {sorted(shapes)}"
            )
        if len(next(iter(shapes))) != 2:
            raise ValueError("Plume summary arrays must be two-dimensional.")

    @property
    def gradient_zscore(self) -> np.ndarray:
        return safe_standardized_mean(self.gradient_mean, self.gradient_std)

    @property
    def motion_zscore(self) -> np.ndarray:
        return safe_standardized_mean(self.motion_mean, self.motion_std)


class _RunningMoments:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finish an empty running-moment accumulator.")
        variance = self.m2 / self.count
        return self.mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class FrameAxisView(Sequence[np.ndarray]):
    """Present an arbitrary array axis as a sequence of 2-D frames."""

    def __init__(
        self,
        array: Any,
        *,
        time_axis: int = 0,
        channel: int | None = None,
        transpose: bool = False,
        flip_y: bool = False,
        flip_x: bool = False,
    ) -> None:
        ndim = len(array.shape)
        if not -ndim <= time_axis < ndim:
            raise ValueError(
                f"time_axis={time_axis} is invalid for an array with {ndim} axes."
            )
        self.array = array
        self.time_axis = time_axis if time_axis >= 0 else time_axis + ndim
        self.channel = channel
        self.transpose = transpose
        self.flip_y = flip_y
        self.flip_x = flip_x

    def __len__(self) -> int:
        return int(self.array.shape[self.time_axis])

    def __getitem__(self, index: int | slice) -> np.ndarray:
        if isinstance(index, slice):
            return np.asarray([self[i] for i in range(*index.indices(len(self)))])
        selection: list[Any] = [slice(None)] * len(self.array.shape)
        selection[self.time_axis] = int(index)
        frame = np.asarray(self.array[tuple(selection)])
        frame = np.squeeze(frame)
        if frame.ndim == 3:
            if self.channel is None:
                raise ValueError(
                    f"Frame {index} has shape {frame.shape}; pass a channel index to select a 2-D image."
                )
            frame = frame[..., self.channel]
        if frame.ndim != 2:
            raise ValueError(
                f"Frame {index} must be 2-D after selection; got {frame.shape}."
            )
        if self.transpose:
            frame = frame.T
        if self.flip_y:
            frame = frame[::-1]
        if self.flip_x:
            frame = frame[:, ::-1]
        return np.asarray(frame)


def finite_difference_y(frame: np.ndarray) -> np.ndarray:
    """Legacy-compatible centered spatial difference along crosswind ``y``."""

    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[0] < 2:
        raise ValueError(
            f"Expected a 2-D frame with at least two rows; got {frame.shape}."
        )
    difference = np.empty_like(frame, dtype=np.float32)
    difference[0] = frame[1] - frame[0]
    difference[-1] = frame[-1] - frame[-2]
    difference[1:-1] = (frame[2:] - frame[:-2]) / 2.0
    return difference


def gradient_and_motion(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``dI/dy`` and the legacy motion cue ``-(dI/dy)(dI/dt)``."""

    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    following = np.asarray(following, dtype=np.float32)
    if previous.shape != current.shape or current.shape != following.shape:
        raise ValueError(
            "Previous, current, and following frames must have the same shape."
        )
    gradient = finite_difference_y(current)
    temporal_difference = (following - previous) / 2.0
    return gradient, -gradient * temporal_difference


def safe_standardized_mean(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Compute the notebook's per-pixel temporal mean/std ratio.

    The source notebook replaced zero standard deviations with one before the
    division.  This matters only for pixels with a constant nonzero cue.
    """

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != std.shape:
        raise ValueError("Mean and standard-deviation arrays must have the same shape.")
    denominator = np.where(std == 0, 1.0, std)
    return np.asarray(mean / denominator, dtype=np.float32)


def _prepare_frame(frame: np.ndarray, *, scale: float, sigma: float) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32) * np.float32(scale)
    if sigma > 0:
        frame = gaussian_filter(frame, sigma=sigma)
    return np.asarray(frame, dtype=np.float32)


def compute_plume_summary(
    frames: Sequence[np.ndarray],
    *,
    snapshot_index: int,
    cue_snapshot_center_index: int | None = None,
    start: int = 0,
    stop: int | None = None,
    map_sigma: float = 0.0,
    cue_snapshot_sigma: float = 1.5,
    snapshot_sigma: float = 3.0,
    intensity_scale: float = 1.0,
    output_stride: int = 1,
    frame_index_offset: int = 0,
    metadata: dict[str, Any] | None = None,
) -> PlumeSummary:
    """Compute the Figure 1 snapshot and whole-window cue statistics.

    ``start`` is inclusive and ``stop`` is exclusive in the loaded frame
    coordinates.  Gradient moments use every selected frame; motion moments
    use the centered frames and therefore have two fewer samples.  The plume
    snapshot is taken at ``snapshot_index``.  By default its cue panels use
    frames ``snapshot_index:snapshot_index+3`` and are centered one frame
    later, as in the source notebook.
    """

    n_frames = len(frames)
    stop = n_frames if stop is None else int(stop)
    start = int(start)
    if start < 0 or stop > n_frames or start >= stop:
        raise ValueError(
            f"Requested frame window [{start}, {stop}) is outside a source with "
            f"{n_frames} frames."
        )
    if stop - start < 3:
        raise ValueError("At least three selected frames are required.")
    if not 0 <= snapshot_index < n_frames:
        raise ValueError(
            f"snapshot_index={snapshot_index} is outside a source with {n_frames} frames."
        )
    cue_snapshot_center_index = (
        snapshot_index + 1
        if cue_snapshot_center_index is None
        else int(cue_snapshot_center_index)
    )
    if not 1 <= cue_snapshot_center_index < n_frames - 1:
        raise ValueError(
            "cue_snapshot_center_index must have a previous and following "
            f"source frame; got {cue_snapshot_center_index} for {n_frames} frames."
        )
    if output_stride < 1:
        raise ValueError("output_stride must be >= 1.")

    previous = _prepare_frame(frames[start], scale=intensity_scale, sigma=map_sigma)
    current = _prepare_frame(frames[start + 1], scale=intensity_scale, sigma=map_sigma)
    moments_gradient = _RunningMoments(current.shape)
    moments_motion = _RunningMoments(current.shape)
    moments_gradient.update(finite_difference_y(previous))
    moments_gradient.update(finite_difference_y(current))
    for index in range(start + 2, stop):
        following = _prepare_frame(
            frames[index], scale=intensity_scale, sigma=map_sigma
        )
        moments_gradient.update(finite_difference_y(following))
        _, motion = gradient_and_motion(previous, current, following)
        moments_motion.update(motion)
        previous, current = current, following

    # Snapshot cue panels used sigma=1.5, while the temporal maps used the
    # unsmoothed movie.  Keep those paths separate.
    snapshot_previous = _prepare_frame(
        frames[cue_snapshot_center_index - 1],
        scale=intensity_scale,
        sigma=cue_snapshot_sigma,
    )
    snapshot_current = _prepare_frame(
        frames[cue_snapshot_center_index],
        scale=intensity_scale,
        sigma=cue_snapshot_sigma,
    )
    snapshot_following = _prepare_frame(
        frames[cue_snapshot_center_index + 1],
        scale=intensity_scale,
        sigma=cue_snapshot_sigma,
    )
    gradient_snapshot, motion_snapshot = gradient_and_motion(
        snapshot_previous, snapshot_current, snapshot_following
    )

    gradient_mean, gradient_std = moments_gradient.finish()
    motion_mean, motion_std = moments_motion.finish()
    snapshot = _prepare_frame(
        frames[snapshot_index], scale=intensity_scale, sigma=snapshot_sigma
    )
    sl = (slice(None, None, output_stride), slice(None, None, output_stride))
    details = dict(metadata or {})
    details.update(
        {
            "snapshot_index": int(snapshot_index),
            "cue_snapshot_center_index": int(cue_snapshot_center_index),
            "source_start": int(start),
            "source_stop": int(stop),
            "frame_index_offset": int(frame_index_offset),
            "snapshot_full_video_frame": int(frame_index_offset + snapshot_index),
            "cue_snapshot_center_full_video_frame": int(
                frame_index_offset + cue_snapshot_center_index
            ),
            "window_full_video_start": int(frame_index_offset + start),
            "window_full_video_stop": int(frame_index_offset + stop),
            "gradient_samples": int(moments_gradient.count),
            "motion_samples": int(moments_motion.count),
            "map_sigma_px": float(map_sigma),
            "cue_snapshot_sigma_px": float(cue_snapshot_sigma),
            "snapshot_sigma_px": float(snapshot_sigma),
            "intensity_scale": float(intensity_scale),
            "output_stride": int(output_stride),
        }
    )
    return PlumeSummary(
        snapshot=snapshot[sl],
        gradient_snapshot=gradient_snapshot[sl],
        motion_snapshot=motion_snapshot[sl],
        gradient_mean=gradient_mean[sl],
        gradient_std=gradient_std[sl],
        motion_mean=motion_mean[sl],
        motion_std=motion_std[sl],
        metadata=details,
    )


def save_plume_summary(summary: PlumeSummary, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(getattr(summary, name), dtype=np.float32)
        for name in SUMMARY_ARRAYS
    }
    arrays["metadata_json"] = np.asarray(json.dumps(summary.metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return path


def load_plume_summary(path: str | Path) -> PlumeSummary:
    with np.load(Path(path), allow_pickle=False) as bundle:
        missing = sorted(set(SUMMARY_ARRAYS) - set(bundle.files))
        if missing:
            raise KeyError(f"Plume summary is missing arrays: {missing}")
        metadata_raw = (
            bundle["metadata_json"].item() if "metadata_json" in bundle else "{}"
        )
        return PlumeSummary(
            **{name: np.asarray(bundle[name]) for name in SUMMARY_ARRAYS},
            metadata=json.loads(str(metadata_raw)),
        )


def _candidate_hdf5_datasets(group: Any, prefix: str = "") -> list[str]:
    candidates: list[str] = []
    for name, item in group.items():
        item_path = f"{prefix}/{name}"
        if hasattr(item, "shape") and len(item.shape) >= 3:
            candidates.append(item_path)
        elif hasattr(item, "items"):
            candidates.extend(_candidate_hdf5_datasets(item, item_path))
    return candidates


@contextmanager
def open_frame_source(
    path: str | Path,
    *,
    dataset: str | None = None,
    time_axis: int = 0,
    channel: int | None = None,
    transpose: bool = False,
    flip_y: bool = False,
    flip_x: bool = False,
) -> Iterator[FrameAxisView]:
    """Open a memory-mapped NPY or an HDF5/NWB dataset as a frame sequence."""

    path = Path(path)
    suffix = path.suffix.lower()
    handle: Any = None
    try:
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        elif suffix == ".npz":
            handle = np.load(path, allow_pickle=False)
            key = dataset or ("frames" if "frames" in handle.files else None)
            if key is None:
                raise ValueError(f"Pass --dataset; NPZ keys are {sorted(handle.files)}")
            array = handle[key]
        elif suffix in {".h5", ".hdf5", ".nwb"}:
            try:
                import h5py
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "Reading HDF5/NWB plume movies requires h5py."
                ) from exc
            handle = h5py.File(path, "r")
            if dataset is None:
                candidates = _candidate_hdf5_datasets(handle)
                if len(candidates) != 1:
                    preview = ", ".join(candidates[:12])
                    raise ValueError(
                        "Pass --dataset to select an image array; "
                        f"found {len(candidates)} candidates: {preview}"
                    )
                dataset = candidates[0]
            array = handle[dataset]
        else:
            raise ValueError(
                f"Unsupported plume movie format {suffix!r}; use NPY, NPZ, HDF5, or NWB."
            )
        yield FrameAxisView(
            array,
            time_axis=time_axis,
            channel=channel,
            transpose=transpose,
            flip_y=flip_y,
            flip_x=flip_x,
        )
    finally:
        if handle is not None:
            handle.close()
