from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS: Dict[str, Dict[str, str]] = {
    "hard": {
        "targets": "dataset/pistachio_hard_targets.txt",
        "output_dir": "output_Pistachio Hard",
        "prefix_name": "pistachio_hard_V3_W5_B100",
        "label": "Pistachio Hard",
    },
    "reachable": {
        "targets": "dataset/pistachio_reachable_targets.txt",
        "output_dir": "output_Pistachio Reachable",
        "prefix_name": "pistachio_reachable_V3_W5_B100",
        "label": "Pistachio Reachable",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pistachio benchmark experiments with checkpointed workers."
    )
    parser.add_argument(
        "dataset",
        choices=sorted(DATASETS),
        help="Pistachio split to run.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--route-proposal-workers",
        type=int,
        default=1,
        help="Concurrent LLM calls inside one molecule expansion.",
    )
    parser.add_argument("--progress-interval", type=int, default=15)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--stream-worker-logs", action="store_true")
    parser.add_argument("--target-indices", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-prefix", default=None)

    parser.add_argument("--model", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--route-width", type=int, default=5)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--template-top-k", type=int, default=100)
    parser.add_argument("--candidate-width", type=int, default=5)
    parser.add_argument("--candidate-template-top-k", type=int, default=1000)
    parser.add_argument("--ablation-no-macro", action="store_true")
    parser.add_argument("--ablation-no-micro", action="store_true")
    parser.add_argument("--ablation-no-candidate-cuts", action="store_true")
    parser.add_argument("--ablation-no-initial-macro-debate", action="store_true")
    return parser.parse_args()


def default_output_prefix(dataset: str) -> Path:
    info = DATASETS[dataset]
    return Path(info["output_dir"]) / info["prefix_name"]


def build_command(args: argparse.Namespace) -> list[str]:
    info = DATASETS[args.dataset]
    output_prefix = Path(args.output_prefix) if args.output_prefix else default_output_prefix(args.dataset)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_output = Path(f"{output_prefix}_merged.json")

    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "run_aot_parallel.py"),
        "--targets",
        info["targets"],
        "--output-prefix",
        str(output_prefix),
        "--completed-source",
        str(merged_output),
        "--merged-output",
        str(merged_output),
        "--workers",
        str(args.workers),
        "--budget",
        str(args.budget),
        "--route-width",
        str(args.route_width),
        "--rag-top-k",
        str(args.rag_top_k),
        "--max-depth",
        str(args.max_depth),
        "--temperature",
        str(args.temperature),
        "--template-top-k",
        str(args.template_top_k),
        "--candidate-width",
        str(args.candidate_width),
        "--candidate-template-top-k",
        str(args.candidate_template_top_k),
        "--route-proposal-workers",
        str(args.route_proposal_workers),
        "--progress-interval",
        str(args.progress_interval),
        "--progress-label",
        info["label"],
    ]
    if args.model:
        command += ["--model", args.model]
    if args.critic_model:
        command += ["--critic-model", args.critic_model]
    if args.ablation_no_macro:
        command.append("--ablation-no-macro")
    if args.ablation_no_micro:
        command.append("--ablation-no-micro")
    if args.ablation_no_candidate_cuts:
        command.append("--ablation-no-candidate-cuts")
    if args.ablation_no_initial_macro_debate:
        command.append("--ablation-no-initial-macro-debate")
    if not args.stream_worker_logs:
        command.append("--progress-bar")
    if args.merge_only:
        command.append("--merge-only")
    if args.stream_worker_logs:
        command.append("--stream-worker-logs")
    if args.target_indices:
        command += ["--target-indices", args.target_indices]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
