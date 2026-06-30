"""Generate published behavior-panel summary tables from archived raw behavior data."""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import butter, filtfilt


DANDI_ASSETS_API = "https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/{version}/assets/?page_size=100"

BASE_PREDICTORS = ("odor_velocity", "spatial_gradient", "signal")
VARIANT_ORDER = ("grad_only", "joint_drop", "mother", "vel_only")
VARIANT_FORMULAS = {
    "mother": "turn_direction ~ odor_velocity_esmooth_{timescale}ms + spatial_gradient_esmooth_{timescale}ms + signal_esmooth_{timescale}ms",
    "vel_only": "turn_direction ~ odor_velocity_esmooth_{timescale}ms + signal_esmooth_{timescale}ms",
    "grad_only": "turn_direction ~ spatial_gradient_esmooth_{timescale}ms + signal_esmooth_{timescale}ms",
    "joint_drop": "turn_direction ~ signal_esmooth_{timescale}ms",
}


@dataclass(frozen=True)
class PlumeSpec:
    name: str
    path_substrings: tuple[str, ...]
    upwind_facing_min: float
    upwind_facing_max: float


DEFAULT_PLUMES = {
    "smooth": PlumeSpec(
        name="smooth",
        path_substrings=("nagel-smoke",),
        upwind_facing_min=160.0,
        upwind_facing_max=200.0,
    ),
    "complex": PlumeSpec(
        name="complex",
        path_substrings=("smoke-2a",),
        upwind_facing_min=150.0,
        upwind_facing_max=210.0,
    ),
}


@dataclass(frozen=True)
class PublishedParams:
    timescale_ms: int = 200
    x_min: float = 30.0
    x_max: float = 220.0
    y_min: float = 67.0
    y_max: float = 97.0
    response_offset_s: float = 0.05
    margin_mm: float = 15.0
    arena_bounds_quantile: float = 0.01
    velocity_filter_cutoff_hz: float = 0.2
    velocity_filter_order: int = 4


@dataclass(frozen=True)
class SummaryOutputs:
    beta_smooth: Path
    beta_complex: Path
    comparison_smooth: Path
    comparison_complex: Path
    manifest: Path


def load_published_params(path: Path) -> PublishedParams:
    payload = json.loads(path.read_text())
    filt = payload["analysis_filter"]
    per_plume = filt.get("per_plume_filter_parameters", {})
    return PublishedParams(
        timescale_ms=int(round(float(filt["timescale_ms"]))),
        x_min=float(filt["x_min"]),
        x_max=float(filt["x_max"]),
        y_min=float(filt["y_min"]),
        y_max=float(filt["y_max"]),
        response_offset_s=float(filt.get("response_offset_s", 0.05)),
        margin_mm=float(filt.get("margin_mm", 15.0)),
        arena_bounds_quantile=float(filt.get("arena_bounds_quantile", 0.01)),
        velocity_filter_cutoff_hz=float(filt.get("velocity_filter_cutoff_hz", 0.2)),
        velocity_filter_order=int(filt.get("velocity_filter_order", 4)),
    )


def list_dandi_assets(dandiset_id: str = "001871", version: str = "0.260630.1657") -> list[dict[str, Any]]:
    url: str | None = DANDI_ASSETS_API.format(dandiset_id=dandiset_id, version=version)
    assets: list[dict[str, Any]] = []
    while url:
        with urllib.request.urlopen(url) as response:
            payload = json.load(response)
        assets.extend(payload["results"])
        url = payload.get("next")
    return assets


def classify_asset_path(path: str, plume_specs: dict[str, PlumeSpec] | None = None) -> str | None:
    specs = plume_specs or DEFAULT_PLUMES
    lower_path = path.lower()
    if not lower_path.endswith("_behavior.nwb"):
        return None
    if "figs1" in lower_path:
        return None
    for plume_name, spec in specs.items():
        if all(token.lower() in lower_path for token in spec.path_substrings):
            return plume_name
    return None


def download_dandi_behavior_nwbs(
    *,
    cache_dir: Path,
    dandiset_id: str = "001871",
    version: str = "0.260630.1657",
    plume_specs: dict[str, PlumeSpec] | None = None,
    max_assets_per_plume: int | None = None,
) -> dict[str, list[Path]]:
    """Download behavior NWB assets needed for the published smooth/complex panels."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Path]] = {name: [] for name in (plume_specs or DEFAULT_PLUMES)}
    counts: dict[str, int] = {name: 0 for name in grouped}

    for asset in list_dandi_assets(dandiset_id=dandiset_id, version=version):
        plume = classify_asset_path(asset["path"], plume_specs=plume_specs)
        if plume is None:
            continue
        if max_assets_per_plume is not None and counts[plume] >= max_assets_per_plume:
            continue

        detail_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/{version}/assets/{asset['asset_id']}/"
        with urllib.request.urlopen(detail_url) as response:
            detail = json.load(response)
        download_url = detail["contentUrl"][0]
        out_path = cache_dir / asset["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists() or out_path.stat().st_size != int(asset["size"]):
            urllib.request.urlretrieve(download_url, out_path)
        grouped[plume].append(out_path)
        counts[plume] += 1

    return grouped


def discover_behavior_nwbs(input_root: Path, plume_specs: dict[str, PlumeSpec] | None = None) -> dict[str, list[Path]]:
    specs = plume_specs or DEFAULT_PLUMES
    grouped: dict[str, list[Path]] = {name: [] for name in specs}
    for path in sorted(input_root.rglob("*_behavior.nwb")):
        plume = classify_asset_path(str(path), plume_specs=specs)
        if plume is not None:
            grouped[plume].append(path)
    return grouped


def _series_timestamps(series: Any, n_rows: int) -> np.ndarray:
    timestamps = getattr(series, "timestamps", None)
    if timestamps is not None:
        return np.asarray(timestamps[:], dtype=float)
    rate = getattr(series, "rate", None)
    if rate is None:
        raise ValueError(f"Time series {getattr(series, 'name', '<unnamed>')} lacks timestamps and rate.")
    start = float(getattr(series, "starting_time", 0.0) or 0.0)
    return start + np.arange(n_rows, dtype=float) / float(rate)


def read_nwb_sessions(nwb_paths: Iterable[Path], plume_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading DANDI NWB files requires pynwb.") from exc

    ts_tables: list[pd.DataFrame] = []
    turn_tables: list[pd.DataFrame] = []
    for nwb_path in nwb_paths:
        with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
            nwb = io.read()
            session_id = str(nwb.session_id or nwb.identifier)
            behavior = nwb.processing["behavior"]
            position = behavior.data_interfaces["position"]
            derived = behavior.data_interfaces["derived_signals"]

            per_traj: dict[str, pd.DataFrame] = {}
            for name, spatial_series in position.spatial_series.items():
                full_trjn = name.rsplit("xy_full_trjn_", 1)[-1]
                xy = np.asarray(spatial_series.data[:], dtype=float)
                timestamps = _series_timestamps(spatial_series, len(xy))
                per_traj[full_trjn] = pd.DataFrame(
                    {
                        "session_id": session_id,
                        "full_trjn": str(full_trjn),
                        "cluster_id": f"{session_id}:{full_trjn}",
                        "navigation_plume": plume_name,
                        "t": timestamps,
                        "x": xy[:, 0],
                        "y": xy[:, 1],
                    }
                )

            for ts_name, time_series in derived.time_series.items():
                if "_full_trjn_" not in ts_name:
                    continue
                signal_name, full_trjn = ts_name.rsplit("_full_trjn_", 1)
                if full_trjn not in per_traj:
                    continue
                values = np.asarray(time_series.data[:], dtype=float)
                if len(values) != len(per_traj[full_trjn]):
                    raise ValueError(f"Length mismatch for {ts_name} in {nwb_path}")
                per_traj[full_trjn][signal_name] = values

            ts_tables.extend(per_traj.values())

            if "turn_events" not in nwb.intervals:
                raise KeyError(f"NWB file lacks turn_events interval table: {nwb_path}")
            turns = nwb.intervals["turn_events"].to_dataframe().reset_index(drop=True)
            turns = turns.rename(columns={"start_time": "turn_start_t", "stop_time": "turn_end_t"})
            turns["session_id"] = session_id
            turns["full_trjn"] = turns["full_trjn"].astype(str)
            turns["cluster_id"] = turns["session_id"] + ":" + turns["full_trjn"]
            turns["navigation_plume"] = plume_name
            turn_tables.append(turns)

    if not ts_tables or not turn_tables:
        raise ValueError(f"No usable NWB behavior data found for plume={plume_name}")
    return pd.concat(ts_tables, ignore_index=True), pd.concat(turn_tables, ignore_index=True)


def exponential_smooth_signal(signal: np.ndarray, fps: float, timescale_ms: float) -> np.ndarray:
    if signal.size < 2:
        return signal.astype(float)
    dt = 1.0 / fps
    tau = timescale_ms / 1000.0
    alpha = 1.0 - math.exp(-dt / tau)
    smoothed = np.zeros_like(signal, dtype=float)
    smoothed[0] = 0.0 if np.isnan(signal[0]) else signal[0]
    for idx in range(1, len(signal)):
        value = signal[idx]
        if np.isnan(value):
            value = smoothed[idx - 1]
        smoothed[idx] = alpha * value + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def _infer_fps(timestamps: pd.Series) -> float:
    diffs = np.diff(np.asarray(timestamps, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 60.0
    return float(1.0 / np.median(diffs))


def _butterworth_or_raw(values: pd.Series, fps: float, cutoff_hz: float, order: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    if len(arr) == 0:
        return arr
    nyq = 0.5 * fps
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return arr
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(arr) <= padlen:
        return arr
    return filtfilt(b, a, arr)


def prepare_turn_table_from_timeseries(
    timeseries: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    plume_spec: PlumeSpec,
    params: PublishedParams,
) -> pd.DataFrame:
    required_ts = {"cluster_id", "t", "x", "y", "theta", "vx", *BASE_PREDICTORS}
    missing_ts = sorted(required_ts - set(timeseries.columns))
    if missing_ts:
        raise KeyError(f"Timeseries table missing required columns: {missing_ts}")

    required_turns = {"cluster_id", "turn_start_t", "turn_end_t", "turn_direction", "turn_x", "turn_y", "turn_theta"}
    missing_turns = sorted(required_turns - set(turns.columns))
    if missing_turns:
        raise KeyError(f"Turn table missing required columns: {missing_turns}")

    ts = timeseries.copy()
    x_vals = pd.to_numeric(ts["x"], errors="coerce")
    y_vals = pd.to_numeric(ts["y"], errors="coerce")
    q = params.arena_bounds_quantile
    x_min = float(np.nanquantile(x_vals, q))
    x_max = float(np.nanquantile(x_vals, 1.0 - q))
    y_min = float(np.nanquantile(y_vals, q))
    y_max = float(np.nanquantile(y_vals, 1.0 - q))

    prepared_groups: list[pd.DataFrame] = []
    for _, group in ts.sort_values(["cluster_id", "t"]).groupby("cluster_id", sort=False):
        group = group.copy()
        fps = _infer_fps(group["t"])
        group["smoothed_vx"] = _butterworth_or_raw(
            group["vx"],
            fps=fps,
            cutoff_hz=params.velocity_filter_cutoff_hz,
            order=params.velocity_filter_order,
        )
        group["walking_upwind"] = group["smoothed_vx"] < 0
        group["facing_upwind"] = (
            (pd.to_numeric(group["theta"], errors="coerce") >= plume_spec.upwind_facing_min)
            & (pd.to_numeric(group["theta"], errors="coerce") <= plume_spec.upwind_facing_max)
        )
        group["near_margin"] = (
            (group["x"] < x_min + params.margin_mm)
            | (group["x"] > x_max - params.margin_mm)
            | (group["y"] < y_min + params.margin_mm)
            | (group["y"] > y_max - params.margin_mm)
        )
        for base in BASE_PREDICTORS:
            out_col = f"{base}_esmooth_{params.timescale_ms}ms"
            group[out_col] = exponential_smooth_signal(
                pd.to_numeric(group[base], errors="coerce").to_numpy(dtype=float),
                fps=fps,
                timescale_ms=params.timescale_ms,
            )
        prepared_groups.append(group)

    ts = pd.concat(prepared_groups, ignore_index=True)
    predictor_cols = [f"{base}_esmooth_{params.timescale_ms}ms" for base in BASE_PREDICTORS]
    attach_cols = ["cluster_id", "t", "facing_upwind", "walking_upwind", "near_margin", *predictor_cols]

    merged_groups: list[pd.DataFrame] = []
    for cluster_id, turn_group in turns.sort_values(["cluster_id", "turn_start_t"]).groupby("cluster_id", sort=False):
        ts_group = ts.loc[ts["cluster_id"] == cluster_id, attach_cols].sort_values("t")
        if ts_group.empty:
            continue
        turn_group = turn_group.copy().sort_values("turn_start_t")
        target_time = (turn_group["turn_start_t"] - params.response_offset_s).round(10)
        turn_group["target_time"] = target_time
        fps = _infer_fps(ts_group["t"])
        merged = pd.merge_asof(
            turn_group,
            ts_group,
            left_on="target_time",
            right_on="t",
            by="cluster_id",
            direction="backward",
            tolerance=1.0 / fps,
        )
        merged_groups.append(merged.drop(columns=["t", "target_time"], errors="ignore"))

    if not merged_groups:
        raise ValueError("No turns could be matched to timeseries rows.")
    out = pd.concat(merged_groups, ignore_index=True)
    for col in ("near_margin", "facing_upwind", "walking_upwind"):
        out[col] = out[col].fillna(False).astype(bool)
    return out


def _map_binary_outcome(y: pd.Series) -> np.ndarray:
    values = set(pd.Series(y).dropna().unique())
    arr = y.to_numpy()
    if values.issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
        return (arr > 0).astype(int)
    return arr


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fit_logit_formula(df: pd.DataFrame, formula: str, cluster_col: str | None) -> Any:
    fit_df = df.copy()
    fit_df["turn_direction"] = _map_binary_outcome(fit_df["turn_direction"])
    required_cols = _required_columns(formula)
    fit_df = fit_df.dropna(subset=required_cols)
    if len(fit_df) < 10:
        raise ValueError(f"Not enough valid rows for formula {formula}: {len(fit_df)}")
    if fit_df["turn_direction"].nunique() < 2:
        raise ValueError(f"Outcome has fewer than two classes for formula {formula}")
    model = smf.logit(formula=formula, data=fit_df, missing="drop")
    if cluster_col:
        result = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": fit_df[cluster_col]})
    else:
        result = model.fit(disp=False, cov_type="HC3")
    return result, fit_df


def _required_columns(formula: str) -> list[str]:
    lhs, rhs = formula.split("~", 1)
    cols = [lhs.strip()]
    for token in rhs.replace("+", " ").replace("-", " ").split():
        token = token.strip()
        if token and token != "1":
            cols.append(token)
    return list(dict.fromkeys(cols))


def _coef_stats(result: Any, term: str) -> dict[str, float]:
    conf = result.conf_int()
    if term in result.params.index:
        return {
            "beta": float(result.params[term]),
            "pvalue": float(result.pvalues[term]),
            "stderr": float(result.bse[term]),
            "ci_lower": float(conf.loc[term].iloc[0]),
            "ci_upper": float(conf.loc[term].iloc[1]),
        }
    return {"beta": float("nan"), "pvalue": float("nan"), "stderr": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}


def filter_published_turns(turn_df: pd.DataFrame, params: PublishedParams) -> pd.DataFrame:
    required = {
        "turn_x",
        "turn_y",
        "near_margin",
        "facing_upwind",
        "walking_upwind",
        "turn_direction",
        "cluster_id",
        f"odor_velocity_esmooth_{params.timescale_ms}ms",
        f"spatial_gradient_esmooth_{params.timescale_ms}ms",
        f"signal_esmooth_{params.timescale_ms}ms",
    }
    missing = sorted(required - set(turn_df.columns))
    if missing:
        raise KeyError(f"Turn table missing required columns: {missing}")
    return turn_df.loc[
        (~turn_df["near_margin"].astype(bool))
        & turn_df["facing_upwind"].astype(bool)
        & turn_df["walking_upwind"].astype(bool)
        & (pd.to_numeric(turn_df["turn_x"], errors="coerce") >= params.x_min)
        & (pd.to_numeric(turn_df["turn_x"], errors="coerce") <= params.x_max)
        & (pd.to_numeric(turn_df["turn_y"], errors="coerce") >= params.y_min)
        & (pd.to_numeric(turn_df["turn_y"], errors="coerce") <= params.y_max)
    ].copy()


def fit_comparison_table(turn_df: pd.DataFrame, params: PublishedParams, cluster_col: str = "cluster_id") -> pd.DataFrame:
    filtered = filter_published_turns(turn_df, params)
    if filtered.empty:
        raise ValueError("Published filter produced an empty turn table.")

    rows: list[dict[str, Any]] = []
    motion_term = f"odor_velocity_esmooth_{params.timescale_ms}ms"
    gradient_term = f"spatial_gradient_esmooth_{params.timescale_ms}ms"
    for variant in VARIANT_ORDER:
        formula = VARIANT_FORMULAS[variant].format(timescale=params.timescale_ms)
        result, fit_df = _fit_logit_formula(filtered, formula, cluster_col=cluster_col)
        y_true = _map_binary_outcome(fit_df["turn_direction"])
        y_score = np.asarray(result.predict(fit_df), dtype=float)
        motion = _coef_stats(result, motion_term)
        gradient = _coef_stats(result, gradient_term)
        rows.append(
            {
                "variant": variant,
                "formula": formula,
                "timescale": float(params.timescale_ms),
                "x_min": float(params.x_min),
                "x_max": float(params.x_max),
                "y_min": float(params.y_min),
                "y_max": float(params.y_max),
                "auroc": _roc_auc_score(y_true, y_score),
                "log_likelihood_mean": float(result.llf) / float(result.nobs),
                "log_likelihood": float(result.llf),
                "nobs": int(result.nobs),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "beta_motion": motion["beta"],
                "p_motion": motion["pvalue"],
                "beta_motion_se": motion["stderr"],
                "beta_motion_ci_lower": motion["ci_lower"],
                "beta_motion_ci_upper": motion["ci_upper"],
                "beta_gradient": gradient["beta"],
                "p_gradient": gradient["pvalue"],
                "beta_gradient_se": gradient["stderr"],
                "beta_gradient_ci_lower": gradient["ci_lower"],
                "beta_gradient_ci_upper": gradient["ci_upper"],
            }
        )
    return pd.DataFrame(rows)


def beta_summary_table(comparison: pd.DataFrame) -> pd.DataFrame:
    mother = comparison.loc[comparison["variant"] == "mother"]
    if mother.empty:
        raise ValueError("Comparison table lacks mother row.")
    return mother.loc[
        :,
        [
            "variant",
            "beta_motion",
            "p_motion",
            "beta_motion_se",
            "beta_gradient",
            "p_gradient",
            "beta_gradient_se",
        ],
    ].reset_index(drop=True)


def write_summary_tables(
    *,
    smooth_turns: pd.DataFrame,
    complex_turns: pd.DataFrame,
    output_dir: Path,
    params: PublishedParams,
) -> SummaryOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    smooth_comparison = fit_comparison_table(smooth_turns, params)
    complex_comparison = fit_comparison_table(complex_turns, params)
    smooth_beta = beta_summary_table(smooth_comparison)
    complex_beta = beta_summary_table(complex_comparison)

    paths = SummaryOutputs(
        beta_smooth=output_dir / "fig5e_cue_beta_smooth_plume.csv",
        beta_complex=output_dir / "fig5e_cue_beta_complex_plume.csv",
        comparison_smooth=output_dir / "fig5f_s5_model_comparison_smooth_plume.csv",
        comparison_complex=output_dir / "fig5f_s5_model_comparison_complex_plume.csv",
        manifest=output_dir / "summary_generation_manifest.json",
    )
    smooth_beta.to_csv(paths.beta_smooth, index=False)
    complex_beta.to_csv(paths.beta_complex, index=False)
    smooth_comparison.loc[:, ["variant", "log_likelihood_mean"]].to_csv(paths.comparison_smooth, index=False)
    complex_comparison.loc[:, ["variant", "log_likelihood_mean"]].to_csv(paths.comparison_complex, index=False)
    paths.manifest.write_text(
        json.dumps(
            {
                "timescale_ms": params.timescale_ms,
                "x_min": params.x_min,
                "x_max": params.x_max,
                "y_min": params.y_min,
                "y_max": params.y_max,
                "response_offset_s": params.response_offset_s,
                "smooth": {
                    "n_turns_input": int(len(smooth_turns)),
                    "n_turns_filtered": int(len(filter_published_turns(smooth_turns, params))),
                },
                "complex": {
                    "n_turns_input": int(len(complex_turns)),
                    "n_turns_filtered": int(len(filter_published_turns(complex_turns, params))),
                },
                "outputs": {key: str(value) for key, value in paths.__dict__.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return paths


def read_turn_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported turn table format: {path}")
