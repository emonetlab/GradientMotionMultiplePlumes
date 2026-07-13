#!/usr/bin/env python3
"""Generate a compact Figure 1 plume summary from a public movie dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.plume_fields import (  # noqa: E402
    compute_plume_summary,
    open_frame_source,
    save_plume_summary,
)
from gradient_motion_panels.plume_sources import (  # noqa: E402
    SmoothTemporalUpsample,
    describe_smooth_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="NPY/NPZ/HDF5/NWB plume movie."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output compressed NPZ summary."
    )
    parser.add_argument("--dataset", help="Dataset path for NPZ/HDF5/NWB input.")
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument(
        "--channel",
        type=int,
        help="Channel index for channel-last frames after selecting the time axis.",
    )
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument(
        "--cue-snapshot-center-index",
        type=int,
        help="Cue center; defaults to snapshot-index + 1, matching the notebook.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=0,
        help="Full-video frame represented by loaded row 0 (for cropped assets).",
    )
    parser.add_argument("--map-sigma", type=float, default=0.0)
    parser.add_argument("--cue-snapshot-sigma", type=float, default=1.5)
    parser.add_argument("--snapshot-sigma", type=float, default=3.0)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument("--plume", choices=("smooth", "complex"), required=True)
    parser.add_argument(
        "--smooth-profile",
        choices=("none", "notebook_legacy", "corrected"),
        default="none",
        help="Convert the 15-Hz Dryad dataset to the legacy 60-Hz arena geometry.",
    )
    parser.add_argument("--source-url")
    parser.add_argument(
        "--snapshot-role",
        choices=("published", "substitute"),
        default="published",
        help="Mark whether the selected plume/cue snapshot is the published frame.",
    )
    parser.add_argument("--snapshot-note")
    parser.add_argument(
        "--hash-input",
        action="store_true",
        help="Record SHA-256 (slow for multi-GB movies).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    metadata = {
        "plume": args.plume,
        "input_path": str(input_path),
        "input_sha256": sha256(input_path) if args.hash_input else None,
        "input_dataset": args.dataset,
        "source_url": args.source_url,
        "generator": "scripts/generate_figure1_summary.py",
        "snapshot_role": args.snapshot_role,
        "snapshot_note": args.snapshot_note,
    }
    with open_frame_source(
        input_path,
        dataset=args.dataset,
        time_axis=args.time_axis,
        channel=args.channel,
        transpose=args.transpose,
        flip_y=args.flip_y,
        flip_x=args.flip_x,
    ) as source:
        metadata["frame_adapter"] = {
            "requested_time_axis": args.time_axis,
            "normalized_time_axis": source.time_axis,
            "channel": args.channel,
            "channel_axis_after_time_selection": -1
            if args.channel is not None
            else None,
            "transpose": args.transpose,
            "flip_y": args.flip_y,
            "flip_x": args.flip_x,
        }
        metadata["source_array_shape"] = [int(value) for value in source.array.shape]
        metadata["source_array_dtype"] = str(source.array.dtype)
        metadata["loaded_frame_count"] = len(source)
        metadata["loaded_frame_shape"] = [int(value) for value in source[0].shape]
        frames = source
        if args.smooth_profile != "none":
            if args.plume != "smooth":
                raise ValueError(
                    "--smooth-profile may only be used with --plume smooth."
                )
            frames = SmoothTemporalUpsample(source, profile=args.smooth_profile)
            metadata["smooth_conversion"] = describe_smooth_profile(args.smooth_profile)
        metadata["effective_frame_count"] = len(frames)
        metadata["effective_frame_shape"] = [int(value) for value in frames[0].shape]
        summary = compute_plume_summary(
            frames,
            snapshot_index=args.snapshot_index,
            cue_snapshot_center_index=args.cue_snapshot_center_index,
            start=args.start,
            stop=args.stop,
            map_sigma=args.map_sigma,
            cue_snapshot_sigma=args.cue_snapshot_sigma,
            snapshot_sigma=args.snapshot_sigma,
            intensity_scale=args.intensity_scale,
            output_stride=args.output_stride,
            frame_index_offset=args.frame_index_offset,
            metadata=metadata,
        )
    save_plume_summary(summary, output_path)
    print(
        json.dumps({"output": str(output_path.resolve()), **summary.metadata}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
