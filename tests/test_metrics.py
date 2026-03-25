"""Unit tests for robo-lint metrics."""

import numpy as np
import pandas as pd
import pytest

from robo_lint.metrics import (
    metric_smoothness,
    metric_static_periods,
    metric_gripper_chatter,
    metric_timestamp_regularity,
    metric_action_saturation,
    metric_episode_length,
)


def _make_df(n=100, seed=42):
    """Generate a synthetic smooth episode."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n)
    return pd.DataFrame({
        "timestamp": np.linspace(0, n / 30, n),
        "action_0": np.sin(t) + rng.normal(0, 0.01, n),
        "action_1": np.cos(t) + rng.normal(0, 0.01, n),
        "gripper_open": np.where(t < np.pi, 1.0, 0.0),
    })


def _make_jerky_df(n=100, seed=42):
    """Generate a jerky/noisy episode."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "timestamp": np.linspace(0, n / 30, n),
        "action_0": rng.normal(0, 5, n),
        "action_1": rng.normal(0, 5, n),
        "gripper_open": rng.choice([0.0, 1.0], n),
    })


class TestSmoothness:
    def test_smooth_episode_scores_high(self):
        df = _make_df()
        score, detail = metric_smoothness(df, ["action_0", "action_1"])
        assert score >= 6.0, f"Smooth episode scored too low: {score}"
        assert detail == "smooth" or detail == "moderate_jerk"

    def test_noisy_episode_scores_low(self):
        df = _make_jerky_df()
        score, detail = metric_smoothness(df, ["action_0", "action_1"])
        assert score < 5.0, f"Jerky episode scored too high: {score}"

    def test_no_action_cols(self):
        df = _make_df()
        score, detail = metric_smoothness(df, [])
        assert score == 5.0


class TestStaticPeriods:
    def test_active_episode(self):
        df = _make_df()
        score, _ = metric_static_periods(df, ["action_0", "action_1"])
        assert score >= 7.0

    def test_static_episode(self):
        df = pd.DataFrame({
            "action_0": np.zeros(100),
            "action_1": np.zeros(100),
        })
        score, detail = metric_static_periods(df, ["action_0", "action_1"])
        assert score <= 3.0
        assert "static" in detail


class TestGripperChatter:
    def test_clean_transitions(self):
        df = _make_df()
        score, _ = metric_gripper_chatter(df)
        assert score >= 7.0

    def test_severe_chatter(self):
        df = _make_jerky_df()
        score, _ = metric_gripper_chatter(df)
        assert score < 5.0


class TestTimestampRegularity:
    def test_regular(self):
        df = _make_df()
        score, _ = metric_timestamp_regularity(df)
        assert score >= 8.0

    def test_no_timestamp(self):
        df = pd.DataFrame({"action_0": [1, 2, 3]})
        score, detail = metric_timestamp_regularity(df)
        assert detail == "no_timestamp_column"


class TestActionSaturation:
    def test_no_saturation(self):
        df = _make_df(n=200)
        score, _ = metric_action_saturation(df, ["action_0", "action_1"])
        assert score >= 5.0


class TestEpisodeLength:
    def test_good_length(self):
        df = _make_df(n=100)
        score, _ = metric_episode_length(df, {})
        assert score >= 8.0

    def test_too_short(self):
        df = _make_df(n=3)
        score, _ = metric_episode_length(df, {})
        assert score <= 2.0
