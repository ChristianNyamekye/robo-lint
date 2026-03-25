"""
robo-lint HuggingFace Space — Upload a LeRobot dataset → get quality report.
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

# Add parent to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent))
from robo_lint.core import load_dataset_local, analyze_dataset, score_episode
from robo_lint.metrics import (
    metric_smoothness,
    metric_static_periods,
    metric_gripper_chatter,
    metric_timestamp_regularity,
    metric_action_saturation,
    metric_episode_length,
)


def analyze_hf_dataset(repo_id: str, progress=gr.Progress()):
    """Analyze a HuggingFace Hub dataset by repo ID."""
    if not repo_id or not repo_id.strip():
        return "⚠️ Please enter a HuggingFace dataset ID (e.g., lerobot/aloha_static_coffee)", "", ""

    repo_id = repo_id.strip()
    if repo_id.startswith("hf://"):
        repo_id = repo_id[5:]

    progress(0.1, desc="Downloading from HuggingFace Hub...")

    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["data/chunk-000/*.parquet", "meta/*.json", "meta/*.jsonl"],
            ignore_patterns=["videos/*", "*.mp4"],
        )
        dataset = load_dataset_local(Path(local_dir))
    except Exception as e:
        return f"❌ Error loading dataset: {e}", "", ""

    return _run_analysis(dataset, progress)


def analyze_uploaded_files(files, progress=gr.Progress()):
    """Analyze uploaded Parquet files or a ZIP of a LeRobot dataset."""
    if not files:
        return "⚠️ Please upload Parquet files or a ZIP archive of your dataset", "", ""

    progress(0.1, desc="Processing uploaded files...")

    tmpdir = tempfile.mkdtemp(prefix="robo_lint_")
    data_dir = Path(tmpdir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        fpath = Path(f.name) if hasattr(f, 'name') else Path(f)
        if fpath.suffix == ".zip":
            with zipfile.ZipFile(fpath, "r") as zf:
                zf.extractall(tmpdir)
        elif fpath.suffix == ".parquet":
            import shutil
            shutil.copy2(fpath, data_dir / fpath.name)
        else:
            return f"⚠️ Unsupported file type: {fpath.suffix}. Upload .parquet or .zip files.", "", ""

    try:
        dataset = load_dataset_local(Path(tmpdir))
    except Exception as e:
        return f"❌ Error loading dataset: {e}", "", ""

    return _run_analysis(dataset, progress)


def _run_analysis(dataset, progress):
    """Run analysis and format results."""
    n_episodes = len(dataset["episodes"])
    if n_episodes == 0:
        return "❌ No episodes found in dataset", "", ""

    def progress_cb(i, total):
        progress((i + 1) / total, desc=f"Analyzing episode {i+1}/{total}...")

    report = analyze_dataset(dataset, max_episodes=200, progress_callback=progress_cb)

    # Format summary
    avg = report["average_quality_score"]
    grade = "🟢 Excellent" if avg >= 8 else ("🟡 Good" if avg >= 6.5 else ("🟠 Fair" if avg >= 5 else "🔴 Poor"))

    summary = f"""## 📊 Dataset Quality Report

| Metric | Value |
|--------|-------|
| **Episodes Analyzed** | {report['total_episodes_analyzed']} |
| **Average Quality** | {avg}/10 {grade} |
| **KEEP** | {report['keep_count']} episodes |
| **TRIM** | {report['trim_count']} episodes |
| **DELETE** | {report['delete_count']} episodes |
| **Usable** | {report['usable_percentage']}% |

### Top Issues
"""
    for issue in report["top_issues"][:5]:
        flag = issue["flag"].replace("_", " ")
        summary += f"- **{flag}** — {issue['count']} episodes\n"

    if report["summary_recommendations"]:
        summary += "\n### Recommendations\n"
        for rec in report["summary_recommendations"]:
            summary += f"- {rec}\n"

    # Per-episode table
    rows = []
    for ep in report["episodes"]:
        emoji = {"KEEP": "🟢", "TRIM": "🟡", "DELETE": "🔴"}.get(ep["recommendation"], "⚪")
        rows.append([
            ep["episode_index"],
            f"{ep['composite_score']:.1f}",
            f"{emoji} {ep['recommendation']}",
            ep["frame_count"],
            ep["reason"][:60],
        ])

    episode_table = pd.DataFrame(rows, columns=["Episode", "Score", "Action", "Frames", "Reason"])

    # JSON report
    json_report = json.dumps(report, indent=2)

    return summary, episode_table, json_report


# ── Gradio UI ─────────────────────────────────────────────────

with gr.Blocks(
    title="robo-lint — Robot Dataset Quality Auditor",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
# 🤖 robo-lint
### Robot Dataset Quality Auditor for LeRobot Training Pipelines

Stop training on garbage. Upload your dataset and get a per-episode quality report with surgical fix recommendations.

**6 physics-informed metrics** · **91.7% precision/recall** · **Validated on ALOHA, PushT, and community datasets**

---
    """)

    with gr.Tabs():
        with gr.Tab("🌐 HuggingFace Dataset"):
            hf_input = gr.Textbox(
                label="Dataset ID",
                placeholder="lerobot/aloha_static_coffee",
                info="Enter a HuggingFace dataset ID (e.g., lerobot/aloha_static_coffee)",
            )
            hf_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Tab("📁 Upload Files"):
            file_input = gr.File(
                label="Upload Parquet files or ZIP archive",
                file_count="multiple",
                file_types=[".parquet", ".zip"],
            )
            upload_btn = gr.Button("🔍 Analyze", variant="primary")

    with gr.Row():
        summary_output = gr.Markdown(label="Summary")

    with gr.Accordion("📋 Per-Episode Details", open=False):
        table_output = gr.Dataframe(
            label="Episode Quality Scores",
            headers=["Episode", "Score", "Action", "Frames", "Reason"],
        )

    with gr.Accordion("📄 Raw JSON Report", open=False):
        json_output = gr.Code(label="JSON", language="json")

    hf_btn.click(
        fn=analyze_hf_dataset,
        inputs=[hf_input],
        outputs=[summary_output, table_output, json_output],
    )
    upload_btn.click(
        fn=analyze_uploaded_files,
        inputs=[file_input],
        outputs=[summary_output, table_output, json_output],
    )

    gr.Markdown("""
---
**[GitHub](https://github.com/ChristianNyamekye/robo-lint)** · **Install:** `pip install robo-lint` · **Built for the LeRobot ecosystem**
    """)


if __name__ == "__main__":
    demo.launch()
