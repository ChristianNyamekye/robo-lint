"""robo-lint — Robot Dataset Quality Auditor for LeRobot training pipelines."""

__version__ = "0.1.0"

from robo_lint.core import analyze_dataset, score_episode, load_dataset
from robo_lint.metrics import (
    metric_smoothness,
    metric_static_periods,
    metric_gripper_chatter,
    metric_timestamp_regularity,
    metric_action_saturation,
    metric_episode_length,
)

__all__ = [
    "analyze_dataset",
    "score_episode",
    "load_dataset",
    "metric_smoothness",
    "metric_static_periods",
    "metric_gripper_chatter",
    "metric_timestamp_regularity",
    "metric_action_saturation",
    "metric_episode_length",
]
