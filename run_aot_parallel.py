from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run experience-guided retrosynthesis search over a target set. "
            "Each worker writes an independent checkpoint, then outputs are merged."
        )
    )
    parser.add_argument("--targets", default="data/test_sets/USPTO-190.smi")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--target-indices",
        default=None,
        help=(
            "Optional 1-based target indices to run, e.g. 78,117 or 78-80. "
            "Indices refer to the original targets file."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="output/uspto190_V3_W5_B100",
        help=(
            "Prefix for worker and merged outputs. Do not include _worker*. "
            "If a .json suffix is supplied it is removed."
        ),
    )
    parser.add_argument(
        "--completed-source",
        default=None,
        help=(
            "Existing completed result file to seed resume/merge. Defaults to "
            "<output-prefix>.json, which preserves the first molecule already run."
        ),
    )
    parser.add_argument("--merged-output", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Seconds between parent progress reports while workers are running.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-worker logs. Defaults to <output-prefix>_logs.",
    )
    parser.add_argument(
        "--stream-worker-logs",
        action="store_true",
        help="Stream verbose worker logs to the terminal instead of log files.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show a single updating progress bar instead of periodic progress lines.",
    )
    parser.add_argument(
        "--progress-label",
        default=None,
        help="Label used for the progress bar.",
    )

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
    parser.add_argument(
        "--route-proposal-workers",
        type=int,
        default=None,
        help="Concurrent LLM calls inside one expansion. Defaults to config value.",
    )

    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    prefix = Path(args.output_prefix)
    if prefix.suffix.lower() == ".json":
        prefix = prefix.with_suffix("")
    args.output_prefix = str(prefix)

    if args.completed_source is None:
        args.completed_source = str(prefix.with_suffix(".json"))
    if args.merged_output is None:
        args.merged_output = str(Path(f"{args.output_prefix}_merged.json"))
    if args.log_dir is None:
        args.log_dir = str(Path(f"{args.output_prefix}_logs"))
    return args


def load_targets(path: str | Path, limit: Optional[int] = None) -> List[str]:
    targets: List[str] = []
    canonicalize = None
    try:
        from chem_utils.reaction import canonicalize_smiles
    except Exception:
        canonicalize = None

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            raw = stripped.split()[0] if stripped else ""
            if not raw or raw.startswith("#"):
                continue
            if stripped[0] in "([":
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, (tuple, list)) and parsed:
                        raw = str(parsed[0]).strip()
                except Exception:
                    pass
            if ">>" in raw:
                raw = raw.split(">>", 1)[0]
            clean = canonicalize(raw) if canonicalize else raw
            if clean:
                targets.append(clean)
            if limit is not None and len(targets) >= limit:
                break
    return targets


def parse_target_indices(spec: str | None, total: int) -> List[int]:
    if not spec:
        return list(range(1, total + 1))

    spec = spec.replace("，", ",")
    selected: List[int] = []
    for raw_part in spec.replace("，", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))

    unique_sorted = sorted(set(selected))
    invalid = [index for index in unique_sorted if index < 1 or index > total]
    if invalid:
        raise ValueError(
            f"--target-indices contains out-of-range indices {invalid}; "
            f"valid range is 1..{total}."
        )
    if not unique_sorted:
        raise ValueError("--target-indices did not contain any valid indices.")
    return unique_sorted


def selected_target_indices(args: argparse.Namespace, total: int) -> List[int]:
    return parse_target_indices(args.target_indices, total)


def load_records(path: str | Path | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    record_path = Path(path)
    if not record_path.exists():
        return []
    try:
        data = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Checkpoint file is not valid JSON: {record_path}") from exc

    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    if isinstance(data, list):
        return data
    return []


def valid_record_index(record: Dict[str, Any], targets: List[str]) -> Optional[int]:
    try:
        index = int(record.get("target_index"))
    except Exception:
        return None
    if not (1 <= index <= len(targets)):
        return None
    if record.get("target") != targets[index - 1]:
        return None
    return index


def prefer_record(current: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    current_success = bool(current.get("is_successful"))
    candidate_success = bool(candidate.get("is_successful"))
    if candidate_success and not current_success:
        return candidate
    if current_success and not candidate_success:
        return current
    if current.get("error") and not candidate.get("error"):
        return candidate
    return current


def records_by_index(
    targets: List[str],
    record_groups: Iterable[List[Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for records in record_groups:
        for record in records:
            index = valid_record_index(record, targets)
            if index is None:
                continue
            if index in merged:
                merged[index] = prefer_record(merged[index], record)
            else:
                merged[index] = record
    return merged


def save_payload(
    records: List[Dict[str, Any]],
    path: str | Path,
    config: Dict[str, Any],
    **extra: Any,
) -> None:
    payload = {
        "config": config,
        "completed_count": len(records),
        "records": records,
        **extra,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def save_json_atomic(data: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def worker_output_path(args: argparse.Namespace, worker_id: int) -> Path:
    return Path(f"{args.output_prefix}_worker{worker_id}.json")


def summary_path(args: argparse.Namespace) -> Path:
    return Path(args.merged_output).with_name(Path(args.merged_output).stem + "_summary.json")


def configure_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    from config import default_config

    config_snapshot = default_config.configure_experiment(
        model=args.model,
        critic_model=args.critic_model,
        budget=args.budget,
        route_width=args.route_width,
        rag_top_k=args.rag_top_k,
        max_search_depth=args.max_depth,
        temperature=args.temperature,
        template_top_k=args.template_top_k,
        candidate_width=args.candidate_width,
        candidate_template_top_k=args.candidate_template_top_k,
        ablation_policy={
            "disable_macro_experience": args.ablation_no_macro,
            "disable_micro_experience": args.ablation_no_micro,
            "disable_candidate_cuts": args.ablation_no_candidate_cuts,
            "disable_initial_macro_debate": args.ablation_no_initial_macro_debate,
        },
    )
    if args.route_proposal_workers is not None:
        default_config.LLM_RUNTIME["route_proposal_workers"] = max(
            1,
            int(args.route_proposal_workers),
        )
        default_config._sync_runtime_aliases()
        config_snapshot = default_config.experiment_snapshot()

    config_snapshot["parallel"] = {
        "workers": args.workers if args.worker_id is None else args.worker_count,
        "worker_id": args.worker_id,
        "route_proposal_workers": getattr(default_config, "LLM_ROUTE_PROPOSAL_WORKERS", 1),
        "completed_source": args.completed_source,
        "merged_output": args.merged_output,
    }
    return config_snapshot


def structured_experience_snapshot(planner) -> Dict[str, Any]:
    if planner is None:
        return {}
    memory = getattr(planner, "experience_memory", None)
    if memory is None:
        return {}
    try:
        if hasattr(memory, "snapshot_all"):
            return memory.snapshot_all(getattr(planner, "validation_stats", {}))
        if hasattr(memory, "snapshot"):
            return memory.snapshot()
        return {}
    except Exception:
        return {}


def experience_snapshot(planner=None) -> Dict[str, Any]:
    if planner is None:
        return {
            "initial_macro_experience": "",
            "initial_macro_experience_chars": 0,
            "final_macro_experience": "",
            "final_macro_experience_chars": 0,
            "experience_trace": [],
            "structured_experience": {},
        }
    macro_experience = getattr(planner, "macro_experience", "") or ""
    initial_macro_experience = getattr(planner, "initial_macro_experience", "") or ""
    return {
        "initial_macro_experience": initial_macro_experience,
        "initial_macro_experience_chars": len(initial_macro_experience),
        "final_macro_experience": macro_experience,
        "final_macro_experience_chars": len(macro_experience),
        "experience_trace": getattr(planner, "experience_trace", []),
        "structured_experience": structured_experience_snapshot(planner),
    }


def route_record(
    target_index: int,
    target: str,
    route,
    planner,
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
    error: str = "",
) -> Dict[str, Any]:
    steps = [asdict(step) for step in route.steps] if route else []
    report = getattr(route, "validation_report", None) if route else None
    planner_log = getattr(planner, "last_log_record", {}) if planner else {}
    return {
        "target_index": target_index,
        "target": target,
        "is_successful": bool(route and route.is_successful),
        "reward": getattr(route, "reward", 0.0) if route else 0.0,
        "steps": steps,
        "validation": {
            "steps_valid": getattr(report, "steps_valid", False),
            "terminal_molecules": getattr(report, "terminal_molecules", []),
            "non_purchasable": getattr(report, "non_purchasable", []),
            "first_invalid_index": getattr(report, "first_invalid_index", -1),
        },
        "elapsed_seconds": elapsed_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
        "actual_iterations": getattr(planner, "actual_iterations", 0) if planner else 0,
        "root_value": getattr(getattr(planner, "root", None), "value", 0.0) if planner else 0.0,
        "root_solved": getattr(getattr(planner, "root", None), "is_solved", False) if planner else False,
        "validation_stats": dict(getattr(planner, "validation_stats", {})) if planner else {},
        "frontier_snapshot": planner.frontier_snapshot() if planner else {},
        "dead_end_snapshot": planner.dead_end_snapshot() if planner else [],
        "planner_log": planner_log,
        "experience_snapshot": experience_snapshot(planner),
    }


def is_empty_infrastructure_failure(record: Dict[str, Any]) -> bool:
    """Detect failed targets produced by no usable LLM response, not by search."""
    planner_log = record.get("planner_log") or {}
    validation_stats = record.get("validation_stats") or {}
    generated = int(validation_stats.get("generated") or 0)
    llm_generated = int(validation_stats.get("llm_pathway_generated") or 0)
    return (
        not record.get("is_successful")
        and not record.get("error")
        and int(record.get("actual_iterations") or 0) == 0
        and not record.get("steps")
        and not planner_log
        and generated == 0
        and llm_generated == 0
    )


def merge_outputs(args: argparse.Namespace, save: bool = True, quiet: bool = False) -> List[Dict[str, Any]]:
    targets = load_targets(args.targets, args.limit)
    selected_indices = selected_target_indices(args, len(targets))
    config = configure_runtime(args)

    sources = []
    completed_source = Path(args.completed_source)
    if completed_source.exists():
        sources.append(completed_source)
    for worker_id in range(args.workers):
        path = worker_output_path(args, worker_id)
        if path.exists():
            sources.append(path)

    groups = [load_records(path) for path in sources]
    merged_by_index = records_by_index(targets, groups)
    merged_records = [
        merged_by_index[index]
        for index in selected_indices
        if index in merged_by_index
    ]

    solved_count = sum(1 for record in merged_records if record.get("is_successful"))
    error_count = sum(1 for record in merged_records if record.get("error"))
    token_total = sum(
        int((record.get("planner_log") or {}).get("total_tokens") or 0)
        for record in merged_records
    )

    if save:
        save_payload(
            merged_records,
            args.merged_output,
            config,
            merged_at=datetime.now().isoformat(timespec="seconds"),
            total_targets=len(selected_indices),
            source_total_targets=len(targets),
            selected_target_indices=selected_indices if args.target_indices else None,
            solved_count=solved_count,
            error_count=error_count,
            source_files=[str(path) for path in sources],
        )
        summary = {
            "total_targets": len(selected_indices),
            "source_total_targets": len(targets),
            "selected_target_indices": selected_indices if args.target_indices else None,
            "completed_count": len(merged_records),
            "solved_count": solved_count,
            "error_count": error_count,
            "total_tokens": token_total,
            "source_files": [str(path) for path in sources],
            "merged_output": args.merged_output,
        }
        save_json_atomic({"config": config, **summary}, summary_path(args))

    if not quiet:
        print(
            f"[Merge][INFO] completed={len(merged_records)}/{len(selected_indices)}, "
            f"solved={solved_count}, errors={error_count}, tokens={token_total}"
        )
        print(f"[Merge][INFO] output={args.merged_output}")
        print(f"[Merge][INFO] summary={summary_path(args)}")
    return merged_records


def run_worker(args: argparse.Namespace) -> int:
    if args.worker_id is None or args.worker_count is None:
        raise ValueError("Worker mode requires --worker-id and --worker-count.")
    if not (0 <= args.worker_id < args.worker_count):
        raise ValueError("--worker-id must be in [0, --worker-count).")

    targets = load_targets(args.targets, args.limit)
    selected_indices = selected_target_indices(args, len(targets))
    config = configure_runtime(args)

    from config import default_config
    from llm.client import LLMAccountError, LLMServiceError, reset_llm_clients
    from planner.strategic_search import StrategicRetrosynthesisPlanner

    reset_llm_clients()

    output_path = worker_output_path(args, args.worker_id)
    seed_records = load_records(args.completed_source)
    worker_records = load_records(output_path)
    worker_records = list(
        records_by_index(targets, [worker_records]).values()
    )
    completed = set(records_by_index(targets, [seed_records, worker_records]))

    assigned_indices = [
        index
        for index in selected_indices
        if (index - 1) % args.worker_count == args.worker_id
    ]

    batch_started = datetime.now().isoformat(timespec="seconds")
    print(
        f"[Worker][INFO] id={args.worker_id}/{args.worker_count}, "
        f"assigned={len(assigned_indices)}, already_completed={len(completed)}"
    )
    print(f"[Worker][INFO] id={args.worker_id}, output={output_path}")
    print(f"[Worker][INFO] id={args.worker_id}, config={config}")

    try:
        for index in assigned_indices:
            target = targets[index - 1]
            if index in completed:
                print(f"[Worker][INFO] id={args.worker_id}, skip target={index}/{len(targets)}")
                continue

            print(f"[Worker][INFO] id={args.worker_id}, start target={index}/{len(targets)}, smiles={target}")
            started_at = datetime.now().isoformat(timespec="seconds")
            tic = time.perf_counter()
            planner = None
            route = None
            error = ""
            try:
                planner = StrategicRetrosynthesisPlanner(
                    target_molecule=target,
                    config=default_config,
                )
                route = planner.run()
            except LLMAccountError as exc:
                save_payload(
                    worker_records,
                    output_path,
                    config,
                    worker_id=args.worker_id,
                    worker_count=args.worker_count,
                    batch_started=batch_started,
                    batch_finished=datetime.now().isoformat(timespec="seconds"),
                    assigned_indices=assigned_indices,
                    fatal_error=f"LLMAccountError: {exc}",
                )
                print(
                    f"[Worker][FATAL] id={args.worker_id}, LLM account error; "
                    f"checkpoint={output_path}, error={exc}"
                )
                return 2
            except LLMServiceError as exc:
                save_payload(
                    worker_records,
                    output_path,
                    config,
                    worker_id=args.worker_id,
                    worker_count=args.worker_count,
                    batch_started=batch_started,
                    batch_finished=datetime.now().isoformat(timespec="seconds"),
                    assigned_indices=assigned_indices,
                    fatal_error=f"LLMServiceError: {exc}",
                )
                print(
                    f"[Worker][WARN] id={args.worker_id}, transient LLM service error; "
                    f"checkpoint={output_path}, error={exc}"
                )
                return 3
            except Exception as exc:
                traceback.print_exc(limit=3)
                save_payload(
                    worker_records,
                    output_path,
                    config,
                    worker_id=args.worker_id,
                    worker_count=args.worker_count,
                    batch_started=batch_started,
                    batch_finished=datetime.now().isoformat(timespec="seconds"),
                    assigned_indices=assigned_indices,
                    fatal_error=f"Unexpected worker error at target {index}: {exc!r}",
                )
                print(
                    f"[Worker][ERROR] id={args.worker_id}, unexpected error at "
                    f"target={index}; checkpoint={output_path}, error={exc!r}"
                )
                return 4

            elapsed = time.perf_counter() - tic
            finished_at = datetime.now().isoformat(timespec="seconds")
            record = route_record(
                index,
                target,
                route,
                planner,
                elapsed,
                started_at,
                finished_at,
                error,
            )
            if is_empty_infrastructure_failure(record):
                save_payload(
                    worker_records,
                    output_path,
                    config,
                    worker_id=args.worker_id,
                    worker_count=args.worker_count,
                    batch_started=batch_started,
                    batch_finished=datetime.now().isoformat(timespec="seconds"),
                    assigned_indices=assigned_indices,
                    fatal_error=(
                        "Empty infrastructure failure: no iterations, no planner log, "
                        "and no validation stats were produced."
                    ),
                )
                print(
                    f"[Worker][WARN] id={args.worker_id}, empty search result; "
                    f"target={index}, checkpoint={output_path}"
                )
                return 3
            worker_records.append(record)
            worker_records = [
                records_by_index(targets, [worker_records])[idx]
                for idx in sorted(records_by_index(targets, [worker_records]))
            ]
            completed.add(index)
            save_payload(
                worker_records,
                output_path,
                config,
                worker_id=args.worker_id,
                worker_count=args.worker_count,
                batch_started=batch_started,
                batch_finished=None,
                assigned_indices=assigned_indices,
            )
            print(
                f"[Worker][INFO] id={args.worker_id}, target={index}, "
                f"solved={record['is_successful']}, reward={record['reward']:.4f}, "
                f"elapsed={elapsed:.1f}s"
            )
            print(f"[Worker][INFO] id={args.worker_id}, checkpoint={output_path}")
    except KeyboardInterrupt:
        print(f"[Worker][WARN] id={args.worker_id}, interrupted; checkpoint={output_path}")
        return 130

    save_payload(
        worker_records,
        output_path,
        config,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
        batch_started=batch_started,
        batch_finished=datetime.now().isoformat(timespec="seconds"),
        assigned_indices=assigned_indices,
    )
    print(f"[Worker][INFO] id={args.worker_id}, finished.")
    return 0


def worker_command(args: argparse.Namespace, worker_id: int) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker-id",
        str(worker_id),
        "--worker-count",
        str(args.workers),
        "--workers",
        str(args.workers),
        "--targets",
        args.targets,
        "--output-prefix",
        args.output_prefix,
        "--completed-source",
        args.merged_output,
        "--merged-output",
        args.merged_output,
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
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.target_indices is not None:
        command += ["--target-indices", args.target_indices]
    if args.route_proposal_workers is not None:
        command += ["--route-proposal-workers", str(args.route_proposal_workers)]
    return command


def worker_log_path(args: argparse.Namespace, worker_id: int) -> Path:
    return Path(args.log_dir) / f"worker{worker_id}.log"


def launch_workers(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    use_progress_bar = bool(args.progress_bar and not args.stream_worker_logs)
    progress_bar = None
    selected_indices: List[int] = []
    if use_progress_bar:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
            use_progress_bar = False
    if use_progress_bar:
        targets = load_targets(args.targets, args.limit)
        selected_indices = selected_target_indices(args, len(targets))
        initial_records = merge_outputs(args, save=True, quiet=True)
        initial_solved = sum(
            1 for record in initial_records if record.get("is_successful")
        )
        progress_bar = tqdm(
            total=len(selected_indices),
            initial=len(initial_records),
            desc=args.progress_label or Path(args.output_prefix).name,
            unit="mol",
            dynamic_ncols=True,
        )
        progress_bar.set_postfix(solved=initial_solved)
    else:
        print("[Runner][INFO] Preparing merged checkpoint before launching workers.")
        merge_outputs(args, save=True, quiet=False)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    processes: List[subprocess.Popen] = []
    log_handles = []
    try:
        for worker_id in range(args.workers):
            command = worker_command(args, worker_id)
            if not use_progress_bar:
                print(f"[Runner][INFO] Launching worker {worker_id}: {' '.join(command)}")
            if args.stream_worker_logs:
                processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=env))
            else:
                log_path = worker_log_path(args, worker_id)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
                log_handles.append(log_handle)
                log_handle.write(
                    f"\n[Worker][INFO] session_start={datetime.now().isoformat(timespec='seconds')}\n"
                    f"[Worker][INFO] command={' '.join(command)}\n"
                )
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                )
                if not use_progress_bar:
                    print(f"[Runner][INFO] Worker {worker_id} verbose_log={log_path}")

        return_codes: List[Optional[int]] = [None for _ in processes]
        last_progress = 0.0
        while any(code is None for code in return_codes):
            for index, process in enumerate(processes):
                if return_codes[index] is None:
                    code = process.poll()
                    if code is not None:
                        return_codes[index] = code
                        if not use_progress_bar:
                            print(f"[Runner][INFO] Worker {index} exited with code {code}.", flush=True)
                        if code not in (0, 130):
                            message = (
                                "[Runner][WARN] A worker stopped before completing its batch; "
                                "terminating other workers to keep checkpoints clean."
                            )
                            if progress_bar is not None:
                                progress_bar.write(message)
                            else:
                                print(message, flush=True)
                            for other_index, other in enumerate(processes):
                                if other_index != index and other.poll() is None:
                                    other.terminate()

            now = time.time()
            if now - last_progress >= max(10, args.progress_interval):
                last_progress = now
                try:
                    records = merge_outputs(args, save=True, quiet=True)
                    if not selected_indices:
                        targets = load_targets(args.targets, args.limit)
                        selected_indices = selected_target_indices(args, len(targets))
                    solved = sum(1 for record in records if record.get("is_successful"))
                    if progress_bar is not None:
                        progress_bar.n = len(records)
                        progress_bar.set_postfix(solved=solved)
                        progress_bar.refresh()
                    else:
                        print(
                            f"[Progress][INFO] completed={len(records)}/{len(selected_indices)}, "
                            f"solved={solved}, merged={args.merged_output}",
                            flush=True,
                        )
                except Exception as exc:
                    message = f"[Progress][WARN] merge check failed: {exc}"
                    if progress_bar is not None:
                        progress_bar.write(message)
                    else:
                        print(message, flush=True)

            time.sleep(2)
    except KeyboardInterrupt:
        if progress_bar is not None:
            progress_bar.write("[Runner][WARN] Parent interrupted; terminating worker processes.")
            progress_bar.close()
        else:
            print("[Runner][WARN] Parent interrupted; terminating worker processes.")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                process.wait(timeout=30)
        for handle in log_handles:
            try:
                handle.close()
            except Exception:
                pass
        return 130

    final_records = merge_outputs(args, save=True, quiet=True if progress_bar is not None else False)
    if progress_bar is not None:
        solved = sum(1 for record in final_records if record.get("is_successful"))
        progress_bar.n = len(final_records)
        progress_bar.set_postfix(solved=solved)
        progress_bar.refresh()
        progress_bar.close()
        print(
            f"[Runner][INFO] completed={len(final_records)}/{len(selected_indices)}, "
            f"solved={solved}, output={args.merged_output}"
        )
        print(f"[Runner][INFO] worker_return_codes={return_codes}")
    else:
        print(f"[Runner][INFO] worker_return_codes={return_codes}")
    for handle in log_handles:
        try:
            handle.close()
        except Exception:
            pass
    return 0 if all(code == 0 for code in return_codes) else 1


def main() -> int:
    args = normalize_args(parse_args())
    os.chdir(PROJECT_ROOT)

    if args.worker_id is not None:
        return run_worker(args)
    if args.merge_only:
        merge_outputs(args, save=True, quiet=False)
        return 0
    return launch_workers(args)


if __name__ == "__main__":
    raise SystemExit(main())

