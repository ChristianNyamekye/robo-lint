"""CLI entry point for robo-lint."""

import argparse
import json
import os
import sys

from robo_lint import __version__
from robo_lint.core import load_dataset, analyze_dataset


# ── Formatting ────────────────────────────────────────────────

COLORS = {
    "KEEP": "\033[92m",
    "TRIM": "\033[93m",
    "DELETE": "\033[91m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "CYAN": "\033[96m",
}


def colorize(text: str, color: str) -> str:
    if sys.platform == "win32" and "WT_SESSION" not in os.environ:
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"


def print_report(report: dict, verbose: bool = False):
    """Pretty-print the analysis report."""
    print()
    print(colorize("=" * 62, "BOLD"))
    print(colorize("  robo-lint — Robot Dataset Quality Audit", "BOLD"))
    print(colorize("=" * 62, "BOLD"))
    print(f"\n  Source:   {report['source']}")
    print(f"  Episodes: {report['total_episodes_analyzed']}")
    avg = report["average_quality_score"]
    avg_color = "KEEP" if avg >= 7 else ("TRIM" if avg >= 5 else "DELETE")
    print(f"  Avg Quality: {colorize(f'{avg}/10', avg_color)}")
    print()

    k = report["keep_count"]
    t = report["trim_count"]
    d = report["delete_count"]
    total = max(k + t + d, 1)
    bar_width = 40
    keep_w = int(k / total * bar_width)
    trim_w = int(t / total * bar_width)
    del_w = bar_width - keep_w - trim_w

    bar = (
        colorize("█" * keep_w, "KEEP")
        + colorize("█" * trim_w, "TRIM")
        + colorize("█" * del_w, "DELETE")
    )
    print(f"  Quality Distribution: [{bar}]")
    print(
        f"    {colorize(f'KEEP {k}', 'KEEP')}  "
        f"{colorize(f'TRIM {t}', 'TRIM')}  "
        f"{colorize(f'DELETE {d}', 'DELETE')}"
    )
    print(f"    Usable: {report['usable_percentage']}%")
    print()

    if report["top_issues"]:
        print(colorize("  Top Issues:", "BOLD"))
        for issue in report["top_issues"][:4]:
            flag = issue["flag"].replace("_", " ")
            print(f"    • {flag} (×{issue['count']} episodes)")
        print()

    if report["summary_recommendations"]:
        print(colorize("  Recommendations:", "BOLD"))
        for rec in report["summary_recommendations"]:
            print(f"    → {rec}")
        print()

    print(colorize("  Per-Episode Fix Plan:", "BOLD"))
    print(colorize("  " + "-" * 60, "DIM"))
    print(colorize(f"  {'Ep':>4}  {'Score':>5}  {'Action':<8}  Reason", "BOLD"))
    print(colorize("  " + "-" * 60, "DIM"))

    ep_list = report["episodes"]
    if not verbose:
        deletes = [e for e in ep_list if e["recommendation"] == "DELETE"]
        trims = [e for e in ep_list if e["recommendation"] == "TRIM"]
        keeps = [e for e in ep_list if e["recommendation"] == "KEEP"]
        ep_display = deletes + trims[:5] + keeps[-3:]
    else:
        ep_display = ep_list

    for ep in ep_display:
        score = ep["composite_score"]
        rec = ep["recommendation"]
        idx = ep["episode_index"]
        reason = ep["reason"][:48]
        score_str = f"{score:4.1f}"
        print(
            f"  {str(idx):>4}  {colorize(score_str, rec):>5}  "
            f"{colorize(rec, rec):<8}  {reason}"
        )

    if not verbose and len(ep_list) > len(ep_display):
        hidden = len(ep_list) - len(ep_display)
        print(colorize(f"\n  ... {hidden} more episodes (use --verbose to see all)", "DIM"))

    print()
    print(colorize("  " + "=" * 60, "DIM"))
    print(colorize(f"  robo-lint v{__version__} | github.com/ChristianNyamekye/robo-lint", "DIM"))
    print()


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="robo-lint — Robot dataset quality auditor for LeRobot training pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  robo-lint ./my_dataset
  robo-lint hf://lerobot/aloha_static_coffee
  robo-lint ./my_dataset --min-quality 5.0
  robo-lint ./my_dataset --json > report.json
  robo-lint ./my_dataset --export report.json --verbose
        """,
    )
    parser.add_argument("dataset", help="Path to LeRobot dataset directory or hf://user/repo")
    parser.add_argument(
        "--min-quality", type=float, default=0.0,
        help="Only show episodes above this quality threshold",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--export", metavar="FILE", help="Save JSON report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all episodes")
    parser.add_argument(
        "--max-episodes", type=int, default=200, help="Max episodes to analyze (default: 200)"
    )
    parser.add_argument(
        "--delete-list", action="store_true",
        help="Output only episode indices to delete (for piping)",
    )
    parser.add_argument("--version", action="version", version=f"robo-lint {__version__}")
    args = parser.parse_args()

    _out = sys.stderr if (args.json or args.delete_list) else sys.stdout
    print(f"\nrobo-lint — loading dataset: {args.dataset}", file=_out)

    try:
        dataset = load_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(dataset['episodes'])} episodes", file=_out)

    report = analyze_dataset(
        dataset, min_quality=args.min_quality, max_episodes=args.max_episodes
    )

    if args.delete_list:
        to_delete = [
            str(ep["episode_index"])
            for ep in report["episodes"]
            if ep["recommendation"] == "DELETE"
        ]
        print("\n".join(to_delete))
        return

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print_report(report, verbose=args.verbose)

    if args.export:
        with open(args.export, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {args.export}")


if __name__ == "__main__":
    main()
