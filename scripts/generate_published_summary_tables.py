#!/usr/bin/env python3
"""Generate published behavior-panel summary tables from archived raw behavior data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradient_motion_panels.summary_tables import (  # noqa: E402
    DEFAULT_PLUMES,
    discover_behavior_nwbs,
    download_dandi_behavior_nwbs,
    load_published_params,
    prepare_turn_table_from_timeseries,
    read_nwb_sessions,
    read_turn_table,
    write_summary_tables,
)


DEFAULT_PARAMS = REPO_ROOT / "metadata" / "published_panel_params.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "generated_summary_tables"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path, help="Local DANDI/NWB directory containing *_behavior.nwb files.")
    source.add_argument("--download-dandi", action="store_true", help="Download DANDI behavior NWB files into --dandi-cache first.")
    source.add_argument("--smooth-turn-table", type=Path, help="Prepared smooth-plume turn table for fast local validation.")
    parser.add_argument("--complex-turn-table", type=Path, help="Prepared complex-plume turn table; required with --smooth-turn-table.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dandiset-id", default="001871")
    parser.add_argument("--dandi-version", default="0.260630.1657")
    parser.add_argument("--dandi-cache", type=Path, default=REPO_ROOT / "data" / "dandi_cache" / "001871")
    parser.add_argument("--max-assets-per-plume", type=int, default=0, help="Optional smoke-test limit; 0 means all assets.")
    parser.add_argument("--write-reconstructed-turns", action="store_true", help="Write reconstructed turn tables alongside summary CSVs.")
    return parser.parse_args()


def _turns_from_nwbs(grouped_paths: dict[str, list[Path]], params) -> tuple:
    reconstructed = {}
    for plume_name in ("smooth", "complex"):
        paths = grouped_paths.get(plume_name, [])
        if not paths:
            raise FileNotFoundError(f"No NWB files found for plume={plume_name}")
        timeseries, turns = read_nwb_sessions(paths, plume_name=plume_name)
        reconstructed[plume_name] = prepare_turn_table_from_timeseries(
            timeseries,
            turns,
            plume_spec=DEFAULT_PLUMES[plume_name],
            params=params,
        )
    return reconstructed["smooth"], reconstructed["complex"]


def main() -> int:
    args = _parse_args()
    if args.smooth_turn_table and not args.complex_turn_table:
        raise SystemExit("--complex-turn-table is required with --smooth-turn-table")

    params_path = args.params if args.params.is_absolute() else REPO_ROOT / args.params
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    params = load_published_params(params_path)

    source_manifest = {}
    if args.smooth_turn_table:
        smooth_turns = read_turn_table(args.smooth_turn_table)
        complex_turns = read_turn_table(args.complex_turn_table)
        source_manifest = {
            "mode": "prepared_turn_tables",
            "smooth_turn_table": str(args.smooth_turn_table),
            "complex_turn_table": str(args.complex_turn_table),
        }
    else:
        max_assets = args.max_assets_per_plume or None
        if args.download_dandi:
            grouped_paths = download_dandi_behavior_nwbs(
                cache_dir=args.dandi_cache,
                dandiset_id=args.dandiset_id,
                version=args.dandi_version,
                max_assets_per_plume=max_assets,
            )
            source_manifest = {
                "mode": "download_dandi",
                "dandiset_id": args.dandiset_id,
                "dandi_version": args.dandi_version,
                "dandi_cache": str(args.dandi_cache),
            }
        else:
            grouped_paths = discover_behavior_nwbs(args.input_root)
            if max_assets is not None:
                grouped_paths = {k: v[:max_assets] for k, v in grouped_paths.items()}
            source_manifest = {"mode": "input_root", "input_root": str(args.input_root)}
        smooth_turns, complex_turns = _turns_from_nwbs(grouped_paths, params)
        source_manifest["n_nwb_files"] = {k: len(v) for k, v in grouped_paths.items()}

    outputs = write_summary_tables(
        smooth_turns=smooth_turns,
        complex_turns=complex_turns,
        output_dir=output_dir,
        params=params,
    )

    if args.write_reconstructed_turns:
        smooth_turns.to_parquet(output_dir / "reconstructed_turns_smooth.parquet", index=False)
        complex_turns.to_parquet(output_dir / "reconstructed_turns_complex.parquet", index=False)

    manifest_payload = json.loads(outputs.manifest.read_text())
    manifest_payload["source"] = source_manifest
    outputs.manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")

    for path in outputs.__dict__.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
