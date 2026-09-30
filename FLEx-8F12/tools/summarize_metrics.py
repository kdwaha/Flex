#!/usr/bin/env python3
"""Print held-out loss or task-wise generation metrics from federated runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing run output folders")
    parser.add_argument(
        "--generation-phase",
        choices=("final", "post-local"),
        default=None,
        help="Show task-wise ROUGE-L/BLEU final scores or latest post-local personalized scores.",
    )
    args = parser.parse_args()

    if args.generation_phase:
        _summarize_generation(args.root, args.generation_phase)
        return

    rows = []
    for metrics_path in sorted(args.root.rglob("metrics.json")):
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        rows.append(
            (
                metrics.get("algorithm", "unknown"),
                metrics.get("evaluation", {}).get("weighted_eval_loss"),
                metrics_path.parent,
            )
        )
    if not rows:
        raise SystemExit(f"No metrics.json found below {args.root}")
    print("algorithm\tweighted_eval_loss\trun")
    for algorithm, loss, run in rows:
        rendered_loss = "n/a" if loss is None else f"{loss:.6f}"
        print(f"{algorithm}\t{rendered_loss}\t{run}")


def _render_metric(value):
    return "n/a" if value is None else f"{float(value):.6f}"


def _summarize_generation(root: Path, phase: str) -> None:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        if phase == "final":
            records = metrics.get("evaluation", {}).get("per_client", [])
            round_value = "final"
        else:
            records = metrics.get("post_local_personalized_evaluation", {}).get(
                "latest_per_client", []
            )
            round_value = "latest-local"
        for record in records:
            if "rougeL_f1" not in record:
                continue
            rows.append(
                (
                    metrics.get("algorithm", "unknown"),
                    record.get("round", round_value),
                    record.get("client"),
                    record.get("task", "iid"),
                    record.get("examples", 0),
                    record.get("rougeL_f1"),
                    record.get("bleu4_pct"),
                    metrics_path.parent,
                )
            )
    if not rows:
        raise SystemExit(
            f"No {phase} generation metrics found below {root}. "
            "Run with --generation_eval and/or --post_local_generation_eval."
        )
    print("algorithm\tphase\tclient\ttask\texamples\trougeL_f1\tbleu4_pct\trun")
    for algorithm, round_value, client, task, examples, rouge_l, bleu, run in rows:
        print(
            f"{algorithm}\t{round_value}\t{client}\t{task}\t{examples}\t"
            f"{_render_metric(rouge_l)}\t{_render_metric(bleu)}\t{run}"
        )


if __name__ == "__main__":
    main()
