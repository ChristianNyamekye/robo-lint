"""
Core analysis engine — dataset loading, per-episode scoring, and report generation.
"""

import glob
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from robo_lint.metrics import (
    metric_smoothness,
    metric_static_periods,
    metric_gripper_chatter,
    metric_timestamp_regularity,
    metric_action_saturation,
    metric_episode_length,
)

warnings.filterwarnings("ignore")


# ── Metric Weights & Impact Descriptions ──────────────────────

METRIC_WEIGHTS = {
    "smoothness": 0.20,
    "static_periods": 0.20,
    "gripper_chatter": 0.15,
    "timestamp_regularity": 0.15,
    "action_saturation": 0.15,
    "episode_length": 0.15,
}

TRAINING_IMPACT = {
    "high_jerk": "Policy learns jerky motions → ~15-25% worse success rate on smooth tasks",
    "mostly_static_remove_this": "Dead time corrupts action distribution → policy stalls mid-task",
    "severe_gripper_chatter": "Policy learns to oscillate gripper → grasp tasks degrade ~30-40%",
    "severe_irregularity_dropped": "Timing mismatch at inference → policy runs out of sync",
    "heavy_saturation_hw_limits_hit": "Policy over-saturates actions → jerky, dangerous behavior",
}


# ── Data Loading ──────────────────────────────────────────────

def load_dataset_local(path: Path) -> dict:
    """Load a LeRobot dataset from local Parquet files."""
    data_dir = path / "data"
    if not data_dir.exists():
        data_dir = path

    parquet_files = sorted(glob.glob(str(data_dir / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {path}")

    episodes = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "episode_index" in df.columns:
            for ep_idx, ep_df in df.groupby("episode_index"):
                episodes[int(ep_idx)] = ep_df.reset_index(drop=True)
        else:
            ep_name = Path(pf).stem
            episodes[ep_name] = df.reset_index(drop=True)

    meta = {}
    meta_path = path / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {"episodes": episodes, "meta": meta, "source": str(path)}


def load_dataset_hf(repo_id: str) -> dict:
    """Load a LeRobot dataset from HuggingFace Hub."""
    if repo_id.startswith("hf://"):
        repo_id = repo_id[5:]

    from huggingface_hub import snapshot_download, hf_hub_download
    import tempfile

    print(f"  Downloading metadata from {repo_id}...", file=sys.stderr)
    try:
        info_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
    except Exception:
        pass

    print(f"  Downloading data (first chunk only for speed)...", file=sys.stderr)
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["data/chunk-000/*.parquet", "meta/*.json", "meta/*.jsonl"],
            ignore_patterns=["videos/*", "*.mp4"],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download from HF Hub: {e}")

    return load_dataset_local(Path(local_dir))


def load_dataset(source: str) -> dict:
    """Auto-detect source type and load dataset."""
    if source.startswith("hf://") or (
        "/" in source
        and not source.startswith(".")
        and not source.startswith("/")
        and not Path(source).exists()
    ):
        return load_dataset_hf(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {source}")
        return load_dataset_local(path)


# ── Column Expansion ──────────────────────────────────────────

def _expand_array_column(df, col: str):
    """If a column contains arrays/lists, expand it into scalar columns."""
    sample = df[col].iloc[0]
    if hasattr(sample, "__len__") and not isinstance(sample, str):
        try:
            arr = np.stack(df[col].values)
            new_cols = []
            for i in range(arr.shape[1]):
                new_name = f"{col}_{i}"
                df[new_name] = arr[:, i].astype(float)
                new_cols.append(new_name)
            return new_cols
        except Exception:
            return []
    return [col]


# ── Per-Episode Scoring ──────────────────────────────────────

def score_episode(ep_idx, df, meta: dict) -> dict:
    """Run all metrics on one episode and return a structured result."""
    # Detect action columns
    action_cols = [c for c in df.columns if c.startswith("action") or c == "action"]
    if not action_cols:
        action_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["joint", "motor", "pos", "vel", "torque"])
        ]

    # Expand array-valued columns
    expanded = []
    for col in action_cols:
        if col in df.columns:
            expanded.extend(_expand_array_column(df, col))
        else:
            expanded.append(col)
    action_cols = expanded

    metrics = {
        "smoothness": metric_smoothness(df, action_cols),
        "static_periods": metric_static_periods(df, action_cols),
        "gripper_chatter": metric_gripper_chatter(df),
        "timestamp_regularity": metric_timestamp_regularity(df),
        "action_saturation": metric_action_saturation(df, action_cols),
        "episode_length": metric_episode_length(df, meta),
    }

    composite = sum(
        metrics[name][0] * weight for name, weight in METRIC_WEIGHTS.items()
    )
    composite = round(composite, 2)

    flags = []
    training_impacts = []
    for name, (score, detail) in metrics.items():
        if score < 4.0:
            flags.append(detail)
            if detail in TRAINING_IMPACT:
                training_impacts.append(TRAINING_IMPACT[detail])

    if composite >= 7.5:
        recommendation = "KEEP"
        reason = "High quality — include in training set"
    elif composite >= 5.0:
        if flags:
            recommendation = "TRIM"
            reason = f"Usable but has issues: {', '.join(flags[:2])}"
        else:
            recommendation = "KEEP"
            reason = "Acceptable quality — include with caution"
    else:
        recommendation = "DELETE"
        reason = f"Too many issues: {', '.join(flags[:3])}"

    return {
        "episode_index": ep_idx,
        "composite_score": composite,
        "recommendation": recommendation,
        "reason": reason,
        "flags": flags,
        "training_impact": training_impacts,
        "metrics": {
            name: {"score": score, "detail": detail}
            for name, (score, detail) in metrics.items()
        },
        "frame_count": len(df),
    }


# ── Dataset-Level Analysis ────────────────────────────────────

def analyze_dataset(
    dataset: dict,
    min_quality: float = 0.0,
    max_episodes: int = 200,
    progress_callback=None,
) -> dict:
    """Run per-episode analysis and build dataset-level report."""
    episodes = dataset["episodes"]
    meta = dataset["meta"]

    results = []
    ep_items = list(episodes.items())[:max_episodes]

    for i, (ep_idx, df) in enumerate(ep_items):
        if progress_callback:
            progress_callback(i, len(ep_items))
        result = score_episode(ep_idx, df, meta)
        if result["composite_score"] >= min_quality:
            results.append(result)

    scores = [r["composite_score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    keep = [r for r in results if r["recommendation"] == "KEEP"]
    trim = [r for r in results if r["recommendation"] == "TRIM"]
    delete = [r for r in results if r["recommendation"] == "DELETE"]

    all_flags = []
    for r in results:
        all_flags.extend(r["flags"])
    flag_counts = {}
    for f in all_flags:
        flag_counts[f] = flag_counts.get(f, 0) + 1
    top_issues = sorted(flag_counts.items(), key=lambda x: -x[1])[:5]

    usable_pct = len(keep) / max(len(results), 1) * 100
    deleted_pct = len(delete) / max(len(results), 1) * 100

    summary_recommendation = []
    if deleted_pct > 30:
        summary_recommendation.append(
            f"⚠️  {deleted_pct:.0f}% of episodes should be deleted — dataset quality is critical"
        )
    if len(delete) > 0:
        summary_recommendation.append(
            f"Delete {len(delete)} episodes → estimated +{min(deleted_pct * 0.4, 25):.0f}% training success rate improvement"
        )
    if len(trim) > 0:
        summary_recommendation.append(
            f"Inspect {len(trim)} borderline episodes before including in training"
        )
    if top_issues:
        top_issue = top_issues[0][0]
        if top_issue in TRAINING_IMPACT:
            summary_recommendation.append(f"Primary issue: {TRAINING_IMPACT[top_issue]}")

    return {
        "source": dataset["source"],
        "total_episodes_analyzed": len(results),
        "average_quality_score": round(avg_score, 2),
        "keep_count": len(keep),
        "trim_count": len(trim),
        "delete_count": len(delete),
        "usable_percentage": round(usable_pct, 1),
        "top_issues": [{"flag": f, "count": c} for f, c in top_issues],
        "summary_recommendations": summary_recommendation,
        "episodes": sorted(results, key=lambda x: x["composite_score"]),
    }
