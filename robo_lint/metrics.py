"""
Per-episode quality metrics for robot demonstration datasets.

Each metric returns (score: float, detail: str) where:
  - score: 0-10 (higher = better quality)
  - detail: human-readable diagnosis string
"""

import numpy as np


def _safe_col_to_float(df, col: str) -> np.ndarray | None:
    """Safely extract a column as a 1D float array, handling array-valued columns."""
    try:
        raw = df[col].values
        # If it's already numeric scalars, just cast
        if raw.dtype in (np.float32, np.float64, np.int32, np.int64, float, int):
            return raw.astype(np.float64)
        # Try direct cast (works for scalar object columns)
        result = raw.astype(np.float64)
        if result.ndim == 1:
            return result
        return None
    except (ValueError, TypeError):
        # Array-valued column — can't use as 1D, skip
        return None


def metric_smoothness(df, action_cols: list) -> tuple:
    """
    LDLJ (Log Dimensionless Jerk) proxy — lower jerk = smoother.
    Measures jerk (3rd derivative of position) to detect jerky, twitchy demos.
    High jerk → policy learns noise, not intent.
    Reference: Hogan & Sternad, 2009.
    """
    if not action_cols:
        return 5.0, "no_action_data"

    scores = []
    for col in action_cols:
        if col not in df.columns:
            continue
        vals = _safe_col_to_float(df, col)
        if vals is None or len(vals) < 4:
            continue
        # Guard against NaN
        if np.any(np.isnan(vals)):
            vals = np.nan_to_num(vals, nan=0.0)
        vel = np.diff(vals)
        acc = np.diff(vel)
        jerk = np.diff(acc)
        if len(jerk) == 0 or np.all(jerk == 0):
            scores.append(10.0)
            continue
        rms_jerk = np.sqrt(np.mean(jerk ** 2))
        rms_vel = np.sqrt(np.mean(vel ** 2)) + 1e-9
        jerk_ratio = rms_jerk / rms_vel
        score = max(0.0, min(10.0, 10.0 * (1.0 - (jerk_ratio - 0.3) / 4.7)))
        scores.append(score)

    if not scores:
        return 5.0, "insufficient_data"
    avg = float(np.mean(scores))
    detail = "smooth" if avg >= 7 else ("moderate_jerk" if avg >= 4 else "high_jerk")
    return round(avg, 2), detail


def metric_static_periods(df, action_cols: list) -> tuple:
    """
    Detect long idle periods where the robot isn't moving.
    Dead time → policy learns to do nothing, reducing action entropy.
    Reference: Liu et al. SCIZOR 2025.
    """
    if not action_cols:
        return 8.0, "no_action_data"

    threshold = 1e-4
    all_actions = []
    for col in action_cols:
        if col in df.columns:
            vals = _safe_col_to_float(df, col)
            if vals is not None:
                if np.any(np.isnan(vals)):
                    vals = np.nan_to_num(vals, nan=0.0)
                all_actions.append(vals)

    if not all_actions:
        return 8.0, "no_action_data"

    action_matrix = np.stack(all_actions, axis=1)
    movement = np.max(np.abs(np.diff(action_matrix, axis=0)), axis=1)
    static_ratio = float(np.mean(movement < threshold))

    if static_ratio < 0.1:
        return 9.0, "active_throughout"
    elif static_ratio < 0.25:
        return 7.0, "minor_idle_periods"
    elif static_ratio < 0.50:
        return 5.0, "significant_idle_time"
    else:
        return 2.0, "mostly_static_remove_this"


def metric_gripper_chatter(df) -> tuple:
    """
    Detect rapid open/close gripper transitions.
    Chatter → policy can't learn consistent grasp intent.
    Reference: Sakr et al. 2024.
    """
    gripper_cols = [c for c in df.columns if "gripper" in c.lower() or "finger" in c.lower()]
    if not gripper_cols:
        return 8.0, "no_gripper_data"

    col = gripper_cols[0]
    vals = _safe_col_to_float(df, col)
    if vals is None or len(vals) < 4:
        return 8.0, "insufficient_data"

    val_range = vals.max() - vals.min()
    if val_range < 1e-6:
        return 7.0, "gripper_constant"

    binary = (vals > (vals.min() + val_range * 0.5)).astype(int)
    transitions = int(np.sum(np.abs(np.diff(binary))))
    chatter_rate = transitions / len(vals)

    if chatter_rate < 0.02:
        return 9.0, "clean_gripper_transitions"
    elif chatter_rate < 0.05:
        return 7.0, "minor_chatter"
    elif chatter_rate < 0.15:
        return 4.0, "moderate_chatter"
    else:
        return 1.5, "severe_gripper_chatter"


def metric_timestamp_regularity(df) -> tuple:
    """
    Check for dropped frames, duplicate timestamps, and frequency jitter.
    Irregular timestamps → policy timing assumptions break during inference.
    """
    if "timestamp" not in df.columns:
        return 7.0, "no_timestamp_column"

    ts = df["timestamp"].values.astype(float)
    if len(ts) < 3:
        return 7.0, "insufficient_data"

    diffs = np.diff(ts)
    if len(diffs) == 0:
        return 7.0, "insufficient_data"

    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return 3.0, "non_monotonic_timestamps"

    cv = float(np.std(diffs) / (median_dt + 1e-9))
    dropped = int(np.sum(diffs > 3 * median_dt))

    if cv < 0.05 and dropped == 0:
        return 10.0, "perfectly_regular"
    elif cv < 0.15 and dropped == 0:
        return 8.0, "minor_jitter"
    elif cv < 0.30 or dropped < 5:
        return 5.5, f"jitter_cv={cv:.2f}_dropped={dropped}"
    else:
        return 2.0, f"severe_irregularity_dropped={dropped}"


def metric_action_saturation(df, action_cols: list) -> tuple:
    """
    Time spent at hardware limits (saturated actions).
    Saturation → policy can't distinguish "trying hard" from "at limit".
    """
    if not action_cols:
        return 8.0, "no_action_data"

    sat_ratios = []
    for col in action_cols:
        if col not in df.columns:
            continue
        vals = _safe_col_to_float(df, col)
        if vals is None or len(vals) < 2:
            continue
        if np.any(np.isnan(vals)):
            vals = np.nan_to_num(vals, nan=0.0)
        vmin, vmax = vals.min(), vals.max()
        val_range = vmax - vmin
        if val_range < 1e-6:
            continue
        margin = val_range * 0.05
        at_limit = (vals < vmin + margin) | (vals > vmax - margin)
        sat_ratios.append(float(np.mean(at_limit)))

    if not sat_ratios:
        return 8.0, "no_action_data"

    avg_sat = float(np.mean(sat_ratios))
    if avg_sat < 0.05:
        return 10.0, "no_saturation"
    elif avg_sat < 0.15:
        return 7.5, "minor_saturation"
    elif avg_sat < 0.35:
        return 5.0, "moderate_saturation"
    else:
        return 2.0, "heavy_saturation_hw_limits_hit"


def metric_episode_length(df, meta: dict) -> tuple:
    """
    Penalize extremely short episodes (< 10 frames) or suspiciously long ones.
    Short episodes: barely any demonstration content.
    Very long: likely includes setup/teardown that confuses the policy.
    """
    n = len(df)
    if n < 5:
        return 1.0, f"only_{n}_frames_too_short"
    elif n < 15:
        return 4.0, f"{n}_frames_borderline_short"
    elif n < 500:
        return 9.0, f"{n}_frames_good_length"
    else:
        return 6.5, f"{n}_frames_check_for_idle"
