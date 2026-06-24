from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceFile:
    name: str
    path: Path
    stage: str
    budget: int
    optional: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    total_targets: int
    output_dir: Path
    consolidated_prefix: str
    sources: List[SourceFile]
    aggregation_note: str


def project_path(path: str) -> Path:
    return ROOT / path


DATASETS: Dict[str, DatasetConfig] = {
    "uspto190": DatasetConfig(
        dataset="USPTO-190",
        total_targets=190,
        output_dir=project_path("output_USPTO"),
        consolidated_prefix="uspto190_hseretro_current_consolidated",
        sources=[
            SourceFile(
                "main_combined_b100",
                project_path("output_USPTO/uspto190_V3_W5_B100_combined_with_initialmacro_retry.json"),
                "main_combined_b100",
                100,
            ),
            SourceFile(
                "failed8_b500",
                project_path("output_USPTO/uspto190_prompt_compact_failed8_B500_merged.json"),
                "failed8_continuation_b500",
                500,
            ),
            SourceFile(
                "root_init_bugfix_b500",
                project_path("output_USPTO/uspto190_root_init_fix_retry_merged.json"),
                "root_init_bugfix_b500",
                500,
                optional=True,
            ),
        ],
        aggregation_note=(
            "USPTO uses the B100 combined run as the baseline. The B500 failed-target "
            "continuation replaces only originally failed targets when computing B300/B500."
        ),
    ),
    "pistachio_hard": DatasetConfig(
        dataset="Pistachio-Hard",
        total_targets=100,
        output_dir=project_path("output_Pistachio Hard"),
        consolidated_prefix="pistachio_hard_hseretro_current_consolidated",
        sources=[
            SourceFile(
                "old88_head_b500",
                project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_oldsuccess88_B500_merged.json"),
                "old_success88_retest_b500",
                500,
            ),
            SourceFile(
                "old88_tail20_b500",
                project_path(
                    "output_Pistachio Hard/pistachio_hard_prompt_compact_oldsuccess88_B500_tail20_merged.json"
                ),
                "old_success88_retest_b500",
                500,
            ),
            SourceFile(
                "failed12_b100",
                project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_failed12_merged.json"),
                "failed12_retest_b100",
                100,
            ),
            SourceFile(
                "failed6_b500",
                project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_failed6_B500_merged.json"),
                "failed6_continuation_b500",
                500,
            ),
            SourceFile(
                "root_init_bugfix_b500",
                project_path("output_Pistachio Hard/pistachio_hard_root_init_fix_retry_merged.json"),
                "root_init_bugfix_b500",
                500,
                optional=True,
            ),
        ],
        aggregation_note=(
            "Hard uses the current-code B500 retest for the original-success 88 targets. "
            "The original-failed 12 use their B100 retest for B100, and the B500 continuation "
            "for B300/B500."
        ),
    ),
    "pistachio_reachable": DatasetConfig(
        dataset="Pistachio-Reachable",
        total_targets=150,
        output_dir=project_path("output_Pistachio Reachable"),
        consolidated_prefix="pistachio_reachable_hseretro_current_consolidated",
        sources=[
            SourceFile(
                "main_b100",
                project_path("output_Pistachio Reachable/pistachio_reachable_V3_W5_B100_merged.json"),
                "main_b100",
                100,
            ),
            SourceFile(
                "failed9_b500",
                project_path(
                    "output_Pistachio Reachable/pistachio_reachable_prompt_compact_failed9_B500_w3_merged.json"
                ),
                "failed9_continuation_b500",
                500,
            ),
            SourceFile(
                "root_init_bugfix_b500",
                project_path(
                    "output_Pistachio Reachable/pistachio_reachable_root_init_fix_retry_merged.json"
                ),
                "root_init_bugfix_b500",
                500,
                optional=True,
            ),
        ],
        aggregation_note=(
            "Reachable uses the B100 full run as baseline. The B500 failed-target continuation "
            "replaces only originally failed targets when computing B300/B500."
        ),
    ),
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = load_json(path)
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    return records


def load_records_optional(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_records(path)


def record_index(record: Dict[str, Any]) -> int:
    return int(record["target_index"])


def is_success(record: Optional[Dict[str, Any]], budget: int) -> bool:
    if not record or not record.get("is_successful"):
        return False
    iterations = record.get("actual_iterations")
    if iterations is None:
        return False
    return int(iterations) <= budget


def slim_record(record: Dict[str, Any], source: SourceFile) -> Dict[str, Any]:
    return {
        "target_index": record_index(record),
        "target": record.get("target"),
        "is_successful": bool(record.get("is_successful")),
        "actual_iterations": record.get("actual_iterations"),
        "root_value": record.get("root_value"),
        "reward": record.get("reward"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "root_solved": record.get("root_solved"),
        "error": record.get("error"),
        "steps": record.get("steps") or [],
        "validation": record.get("validation"),
        "validation_stats": record.get("validation_stats") or {},
        "planner_log": record.get("planner_log") or {},
        "frontier_snapshot": record.get("frontier_snapshot") or [],
        "dead_end_snapshot": record.get("dead_end_snapshot") or [],
        "experience_snapshot": record.get("experience_snapshot") or {},
        "source_file": str(source.path.relative_to(ROOT)),
        "source_stage": source.stage,
        "source_budget": source.budget,
    }


def load_stage_maps(config: DatasetConfig) -> Dict[str, Dict[int, Dict[str, Any]]]:
    stages: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for source in config.sources:
        if source.optional and not source.path.exists():
            continue
        source_records = load_records_optional(source.path) if source.optional else load_records(source.path)
        if source.optional and not source_records:
            continue
        stage_map = stages.setdefault(source.stage, {})
        for record in source_records:
            stage_map[record_index(record)] = slim_record(record, source)
    return stages


def prefer_better_record(
    current: Optional[Dict[str, Any]],
    candidate: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return current
    if current is None:
        return candidate

    current_success = bool(current.get("is_successful"))
    candidate_success = bool(candidate.get("is_successful"))
    if candidate_success and not current_success:
        return candidate
    if current_success and not candidate_success:
        return current

    if candidate.get("error") and not current.get("error"):
        return current
    if current.get("error") and not candidate.get("error"):
        return candidate

    current_root_solved = bool(current.get("root_solved"))
    candidate_root_solved = bool(candidate.get("root_solved"))
    if candidate_root_solved and not current_root_solved:
        return candidate
    if current_root_solved and not candidate_root_solved:
        return current

    current_iterations = current.get("actual_iterations")
    candidate_iterations = candidate.get("actual_iterations")
    if current_success and candidate_success and current_iterations is not None and candidate_iterations is not None:
        return candidate if int(candidate_iterations) < int(current_iterations) else current

    return current


def choose_baseline_and_continuation(
    config: DatasetConfig,
    stages: Dict[str, Dict[int, Dict[str, Any]]],
    target_index: int,
) -> Dict[str, Dict[str, Any]]:
    if config.dataset == "Pistachio-Hard":
        if target_index in stages.get("old_success88_retest_b500", {}):
            return {"baseline": stages["old_success88_retest_b500"][target_index]}

        baseline = stages["failed12_retest_b100"][target_index]
        records = {"baseline": baseline}
        continuation = None
        for stage_name in ["failed6_continuation_b500", "root_init_bugfix_b500"]:
            continuation = prefer_better_record(
                continuation,
                stages.get(stage_name, {}).get(target_index),
            )
        if continuation:
            records["continuation"] = continuation
        return records

    baseline_stage = "main_combined_b100" if config.dataset == "USPTO-190" else "main_b100"
    baseline = stages[baseline_stage][target_index]
    records = {"baseline": baseline}
    continuation_stages = [
        stage
        for stage in stages
        if "continuation_b500" in stage or stage == "root_init_bugfix_b500"
    ]
    best_continuation = None
    for stage in continuation_stages:
        continuation = stages[stage].get(target_index)
        if continuation:
            best_continuation = prefer_better_record(best_continuation, continuation)
    if best_continuation:
        records["continuation"] = best_continuation
    return records


def final_record_for_budget(records: Dict[str, Dict[str, Any]], budget: int) -> Dict[str, Any]:
    baseline = records["baseline"]
    continuation = records.get("continuation")
    if budget <= 100:
        return baseline
    if is_success(baseline, budget):
        return baseline
    return continuation or baseline


def consolidate_dataset(config: DatasetConfig) -> Dict[str, Any]:
    stages = load_stage_maps(config)
    stage_indices = sorted({idx for stage in stages.values() for idx in stage})
    missing = sorted(set(range(1, config.total_targets + 1)) - set(stage_indices))
    if missing:
        raise RuntimeError(f"{config.dataset} is missing target indices: {missing}")

    records: List[Dict[str, Any]] = []
    for target_index in range(1, config.total_targets + 1):
        stage_records = choose_baseline_and_continuation(config, stages, target_index)
        target = stage_records["baseline"].get("target")
        budget_success = {}
        final_by_budget = {}
        for budget in [100, 300, 500]:
            chosen = final_record_for_budget(stage_records, budget)
            success = is_success(chosen, budget)
            budget_success[str(budget)] = success
            final_by_budget[str(budget)] = {
                "source_stage": chosen.get("source_stage"),
                "actual_iterations": chosen.get("actual_iterations"),
                "root_value": chosen.get("root_value"),
                "reward": chosen.get("reward"),
            }
        final_500 = final_record_for_budget(stage_records, 500)
        records.append(
            {
                "target_index": target_index,
                "target": target,
                "success_at_budget_100": budget_success["100"],
                "success_at_budget_300": budget_success["300"],
                "success_at_budget_500": budget_success["500"],
                "final_success": budget_success["500"],
                "final_actual_iterations": final_500.get("actual_iterations"),
                "final_root_value": final_500.get("root_value"),
                "final_reward": final_500.get("reward"),
                "final_source_stage": final_500.get("source_stage"),
                "budget_records": final_by_budget,
                "stage_records": stage_records,
            }
        )

    budget_stats = {}
    for budget in [100, 300, 500]:
        key = f"success_at_budget_{budget}"
        solved = [record for record in records if record[key]]
        failed = [record for record in records if not record[key]]
        budget_stats[str(budget)] = {
            "solved_count": len(solved),
            "failed_count": len(failed),
            "success_rate": len(solved) / config.total_targets,
            "solved_indices": [record["target_index"] for record in solved],
            "failed_indices": [record["target_index"] for record in failed],
        }

    return {
        "summary": {
            "dataset": config.dataset,
            "framework": "HSE-Retro",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "aggregation_rule": config.aggregation_note,
            "total_targets": config.total_targets,
            "budget_stats": budget_stats,
            "final_failed_at_500": budget_stats["500"]["failed_indices"],
            "stage_files": {source.name: str(source.path.relative_to(ROOT)) for source in config.sources},
        },
        "records": records,
    }


def write_dataset_outputs(config: DatasetConfig, payload: Dict[str, Any]) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_dir / config.consolidated_prefix
    json_path = prefix.with_suffix(".json")
    summary_path = Path(str(prefix) + "_summary.json")
    csv_path = prefix.with_suffix(".csv")

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "target_index",
            "success_at_budget_100",
            "success_at_budget_300",
            "success_at_budget_500",
            "final_success",
            "final_actual_iterations",
            "final_root_value",
            "final_reward",
            "final_source_stage",
            "target",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in payload["records"]:
            writer.writerow({field: record.get(field) for field in fields})
    return {"json": json_path, "summary": summary_path, "csv": csv_path}


def legal_prefix_acceptance(record: Dict[str, Any]) -> Optional[float]:
    stats = record.get("validation_stats") or {}
    generated = stats.get("llm_pathways_generated")
    accepted = stats.get("llm_pathways_accepted")
    if not generated:
        return None
    return float(accepted or 0) / float(generated)


def collect_metric_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    metric_records = []
    for record in payload["records"]:
        final_stage = record.get("final_source_stage")
        stage_records = record.get("stage_records") or {}
        final_record = None
        for candidate in stage_records.values():
            if candidate.get("source_stage") == final_stage:
                final_record = candidate
                break
        if final_record is None:
            final_record = next(iter(stage_records.values()))
        metric_records.append(final_record)
    return metric_records


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(concrete) / len(concrete)


def sum_stat(records: List[Dict[str, Any]], key: str) -> int:
    total = 0
    for record in records:
        total += int((record.get("validation_stats") or {}).get(key, 0) or 0)
    return total


def sum_any_stat(records: List[Dict[str, Any]], keys: List[str]) -> int:
    total = 0
    for record in records:
        stats = record.get("validation_stats") or {}
        for key in keys:
            if key in stats:
                total += int(stats.get(key, 0) or 0)
                break
    return total


def efficiency_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload["summary"]
    records = collect_metric_records(payload)
    solved_records = [record for record in payload["records"] if record.get("success_at_budget_500")]
    metric_solved = [record for record in records if record.get("is_successful")]
    total_tokens = 0
    total_calls = 0
    for record in records:
        planner_log = record.get("planner_log") or {}
        usage = planner_log.get("llm_usage") or {}
        total_tokens += int(planner_log.get("total_tokens", usage.get("total_tokens", 0)) or 0)
        total_calls += int(planner_log.get("calls", usage.get("calls", 0)) or 0)
    total_generated = sum_any_stat(records, ["llm_pathway_generated", "llm_pathways_generated"])
    total_accepted = sum_any_stat(records, ["llm_pathway_accepted", "llm_pathways_accepted"])
    return {
        "dataset": summary["dataset"],
        "solved_at_500": summary["budget_stats"]["500"]["solved_count"],
        "total_targets": summary["total_targets"],
        "total_llm_calls_from_records": total_calls,
        "total_tokens_from_records": total_tokens,
        "avg_tokens_per_solved": round(total_tokens / len(solved_records), 2) if solved_records else "",
        "avg_elapsed_seconds_per_molecule": round(average(record.get("elapsed_seconds") for record in records) or 0, 2),
        "avg_iterations_per_solved": round(average(record.get("actual_iterations") for record in metric_solved) or 0, 2),
        "llm_pathways_generated": total_generated,
        "llm_pathways_accepted": total_accepted,
        "legal_prefix_acceptance_rate": round(total_accepted / total_generated, 4) if total_generated else "",
        "template_exact": sum_stat(records, "template_exact"),
        "template_topk": sum_stat(records, "template_topk"),
        "template_unmatched": sum_stat(records, "template_unmatched"),
        "llm_rejected": sum_any_stat(records, ["llm_pathway_rejected", "llm_pathways_rejected"]),
        "root_init_failures": len(
            [record for record in payload["records"] if not record.get("final_success") and record.get("final_actual_iterations") == 0]
        ),
    }


def classify_failure(record: Dict[str, Any]) -> str:
    if record.get("final_actual_iterations") == 0:
        return "Root initialization failure"
    stage_records = record.get("stage_records") or {}
    text_parts = []
    for stage_record in stage_records.values():
        text_parts.append(json.dumps(stage_record.get("frontier_snapshot") or [], ensure_ascii=False))
        text_parts.append(json.dumps(stage_record.get("dead_end_snapshot") or [], ensure_ascii=False))
    text = " ".join(text_parts)
    if any(fragment in text for fragment in ["[Li]", "OBr", "O=O", "[Mg]", "[Na]"]):
        return "Search trapped in chemically implausible intermediates"
    if int(record.get("final_actual_iterations") or 0) >= 500:
        return "Budget exhausted with many invalid prefixes"
    return "Unresolved search failure"


def error_rows(payloads: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for payload in payloads.values():
        dataset = payload["summary"]["dataset"]
        for record in payload["records"]:
            if record.get("success_at_budget_500"):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "target_index": record["target_index"],
                    "failure_type": classify_failure(record),
                    "actual_iterations": record.get("final_actual_iterations"),
                    "root_value": record.get("final_root_value"),
                    "target": record.get("target"),
                }
            )
    return rows


def extract_case_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    stage_records = record.get("stage_records") or {}
    final_stage = record.get("final_source_stage")
    final_record = None
    for candidate in stage_records.values():
        if candidate.get("source_stage") == final_stage:
            final_record = candidate
            break
    final_record = final_record or next(iter(stage_records.values()))
    experience = final_record.get("experience_snapshot") or {}
    steps = final_record.get("steps") or []
    first_step = steps[0] if steps else {}
    return {
        "target_index": record["target_index"],
        "target": record.get("target"),
        "success": record.get("success_at_budget_500"),
        "actual_iterations": record.get("final_actual_iterations"),
        "source_stage": record.get("final_source_stage"),
        "route_steps": len(steps),
        "first_reaction": first_step.get("reaction"),
        "first_reactants": first_step.get("reactants"),
        "macro_preview": str(
            experience.get("initial_macro_experience")
            or experience.get("final_macro_experience")
            or ""
        )[:1000],
        "micro_nodes": len(
            (
                (experience.get("structured_experience") or {}).get("micro_experiences")
                or (experience.get("structured_experience") or {}).get("micro")
                or {}
            )
        ),
        "validation_feedback": first_step.get("feedback"),
    }


def find_record(payload: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    for record in payload["records"]:
        if record["target_index"] == index:
            return record
    return None


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_commands(path: Path) -> None:
    commands = {
        "status": "ready_to_run",
        "reason": (
            "The runner exposes these ablation flags and stores the ablation policy in each "
            "experiment snapshot. These commands launch API-consuming experiments."
        ),
        "variants": [
        {
            "variant": "w/o Macro Experience",
            "command": (
                "python -u run_pistachio_parallel.py hard --budget 100 --workers 2 "
                "--route-proposal-workers 1 --output-prefix \"output_Pistachio Hard/ablation_no_macro_B100\" "
                "--ablation-no-macro"
            ),
            "note": "Requires adding/confirming a planner switch that disables initial macro and macro updates.",
        },
        {
            "variant": "w/o Micro Experience",
            "command": (
                "python -u run_pistachio_parallel.py hard --budget 100 --workers 2 "
                "--route-proposal-workers 1 --output-prefix \"output_Pistachio Hard/ablation_no_micro_B100\" "
                "--ablation-no-micro"
            ),
            "note": "Requires adding/confirming a planner switch that disables node-level micro memory injection and updates.",
        },
        {
            "variant": "w/o Candidate Cuts",
            "command": (
                "python -u run_pistachio_parallel.py hard --budget 100 --workers 2 "
                "--route-proposal-workers 1 --output-prefix \"output_Pistachio Hard/ablation_no_candidate_cuts_B100\" "
                "--ablation-no-candidate-cuts"
            ),
            "note": "Requires adding/confirming a planner switch that hides candidate graph context from LLM prompts.",
        },
        {
            "variant": "w/o Initial Macro Debate",
            "command": (
                "python -u run_pistachio_parallel.py hard --budget 100 --workers 2 "
                "--route-proposal-workers 1 --output-prefix \"output_Pistachio Hard/ablation_no_initial_macro_debate_B100\" "
                "--ablation-no-initial-macro-debate"
            ),
            "note": "Recommended optional ablation; keep macro memory but replace debate with a single strategist draft.",
        },
    ]}
    path.write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")


def success_indices(records: Iterable[Dict[str, Any]]) -> set[int]:
    return {record_index(record) for record in records if record.get("is_successful")}


def failure_indices(records: Iterable[Dict[str, Any]]) -> set[int]:
    return {record_index(record) for record in records if not record.get("is_successful")}


def record_map(records: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {record_index(record): record for record in records}


def merge_preferred_maps(*maps: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for source_map in maps:
        for index, record in source_map.items():
            merged[index] = prefer_better_record(merged.get(index), record) or record
    return merged


def solve_by_iteration(record: Optional[Dict[str, Any]], budget: int) -> bool:
    if not record or not record.get("is_successful"):
        return False
    iterations = record.get("actual_iterations")
    return iterations is not None and int(iterations) <= budget


def provenance_row(
    dataset: str,
    total: int,
    legacy_b100_solved: int,
    budget_solved: Dict[int, set[int]],
    failed_at_500: set[int],
    post_change_coverage: str,
    source_note: str,
) -> Dict[str, Any]:
    row = {
        "Dataset": dataset,
        "Legacy V3_W5 B100": f"{legacy_b100_solved}/{total} ({legacy_b100_solved / total:.1%})",
        "Post-change coverage": post_change_coverage,
        "B100": f"{len(budget_solved[100])}/{total} ({len(budget_solved[100]) / total:.1%})",
        "B300": f"{len(budget_solved[300])}/{total} ({len(budget_solved[300]) / total:.1%})",
        "B500": f"{len(budget_solved[500])}/{total} ({len(budget_solved[500]) / total:.1%})",
        "Failed@500": ",".join(map(str, sorted(failed_at_500))),
        "Source note": source_note,
    }
    return row


def write_provenance_consolidated(
    output_dir: Path,
    dataset: str,
    total: int,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    safe_name = dataset.lower().replace("-", "_").replace(" ", "_")
    json_path = output_dir / f"{safe_name}_provenance_consolidated.json"
    csv_path = output_dir / f"{safe_name}_provenance_consolidated.csv"
    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = [
        "target_index",
        "legacy_b100_success",
        "post_change_tested",
        "success_at_budget_100",
        "success_at_budget_300",
        "success_at_budget_500",
        "final_source",
        "final_actual_iterations",
        "target",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def build_provenance_outputs(output_dir: Path) -> Dict[str, Any]:
    """Build corrected experiment tables that preserve code-version provenance.

    V3_W5 files are treated as the legacy B100 main experiments. Prompt-compact
    files are treated as post-change retests or failed-target continuations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # USPTO-190: legacy B100 combined run plus post-change B500 on the 8 failures.
    uspto_base = load_records(project_path("output_USPTO/uspto190_V3_W5_B100_combined_with_initialmacro_retry.json"))
    uspto_supp = load_records(project_path("output_USPTO/uspto190_prompt_compact_failed8_B500_merged.json"))
    uspto_root_fix = load_records_optional(project_path("output_USPTO/uspto190_root_init_fix_retry_merged.json"))
    uspto_base_success = success_indices(uspto_base)
    uspto_supp_map = merge_preferred_maps(record_map(uspto_supp), record_map(uspto_root_fix))
    uspto_budget: Dict[int, set[int]] = {}
    for budget in [100, 300, 500]:
        solved = set(uspto_base_success)
        solved.update(idx for idx, record in uspto_supp_map.items() if solve_by_iteration(record, budget))
        uspto_budget[budget] = solved
    uspto_records = []
    uspto_base_map = record_map(uspto_base)
    for idx in range(1, 191):
        supp = uspto_supp_map.get(idx)
        final = supp if supp is not None else uspto_base_map[idx]
        uspto_records.append(
            {
                "target_index": idx,
                "target": final.get("target"),
                "legacy_b100_success": idx in uspto_base_success,
                "post_change_tested": supp is not None,
                "success_at_budget_100": idx in uspto_budget[100],
                "success_at_budget_300": idx in uspto_budget[300],
                "success_at_budget_500": idx in uspto_budget[500],
                "final_source": (supp.get("source_stage") if supp and supp.get("source_stage") else "failed8_or_rootfix_b500") if supp else "legacy_v3_w5_b100",
                "final_actual_iterations": final.get("actual_iterations"),
            }
        )
    uspto_failed_500 = set(range(1, 191)) - uspto_budget[500]
    uspto_summary = {
        "dataset": "USPTO-190",
        "legacy_b100_solved": len(uspto_base_success),
        "post_change_coverage": "failed targets only: 8/190",
        "budget_stats": {
            str(budget): {
                "solved_count": len(solved),
                "failed_indices": sorted(set(range(1, 191)) - solved),
            }
            for budget, solved in uspto_budget.items()
        },
        "source_note": (
            "Legacy V3_W5_B100 had 8 failures; prompt-compact B500 was run only on those failures. "
            "Root-initialization bugfix retries are treated as replacements for affected root-init failures."
        ),
    }
    write_provenance_consolidated(output_dir, "USPTO-190", 190, uspto_records, uspto_summary)

    # Pistachio-Hard: legacy split inferred from original 88 success set and 12 failure set;
    # post-change experiments cover all 100 targets.
    hard_old88 = {}
    for path in [
        project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_oldsuccess88_B500_merged.json"),
        project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_oldsuccess88_B500_tail20_merged.json"),
    ]:
        hard_old88.update(record_map(load_records(path)))
    hard_failed12 = record_map(
        load_records(project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_failed12_merged.json"))
    )
    hard_failed6 = record_map(
        load_records(project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_failed6_B500_merged.json"))
    )
    hard_root_fix = record_map(
        load_records_optional(project_path("output_Pistachio Hard/pistachio_hard_root_init_fix_retry_merged.json"))
    )
    hard_legacy_success = set(hard_old88)
    hard_budget: Dict[int, set[int]] = {100: set(), 300: set(), 500: set()}
    hard_records = []
    for idx in range(1, 101):
        if idx in hard_old88:
            final = hard_old88[idx]
            for budget in [100, 300, 500]:
                if solve_by_iteration(final, budget):
                    hard_budget[budget].add(idx)
            final_source = "oldsuccess88_b500"
            post_change_tested = True
        else:
            b100_record = hard_failed12[idx]
            b500_record = merge_preferred_maps(hard_failed6, hard_root_fix).get(idx)
            final = b500_record or b100_record
            for budget in [100, 300, 500]:
                if b100_record.get("is_successful") and int(b100_record.get("actual_iterations") or 999999) <= budget:
                    hard_budget[budget].add(idx)
                elif b500_record and solve_by_iteration(b500_record, budget):
                    hard_budget[budget].add(idx)
            final_source = "failed6_or_rootfix_b500" if b500_record else "failed12_b100"
            post_change_tested = True
        hard_records.append(
            {
                "target_index": idx,
                "target": final.get("target"),
                "legacy_b100_success": idx in hard_legacy_success,
                "post_change_tested": post_change_tested,
                "success_at_budget_100": idx in hard_budget[100],
                "success_at_budget_300": idx in hard_budget[300],
                "success_at_budget_500": idx in hard_budget[500],
                "final_source": final_source,
                "final_actual_iterations": final.get("actual_iterations"),
            }
        )
    hard_failed_500 = set(range(1, 101)) - hard_budget[500]
    hard_summary = {
        "dataset": "Pistachio-Hard",
        "legacy_b100_solved": len(hard_legacy_success),
        "post_change_coverage": "full retest/continuation coverage: 100/100",
        "budget_stats": {
            str(budget): {
                "solved_count": len(solved),
                "failed_indices": sorted(set(range(1, 101)) - solved),
            }
            for budget, solved in hard_budget.items()
        },
        "source_note": (
            "Legacy V3_W5_B100 split is inferred from the old-success88 and failed12 target lists. "
            "Post-change experiments cover all 100 targets; B100/B300/B500 are computed from actual_iterations "
            "in available post-change B100/B500 runs. Root-initialization bugfix retries replace affected "
            "root-init failures when present."
        ),
    }
    write_provenance_consolidated(output_dir, "Pistachio-Hard", 100, hard_records, hard_summary)

    # Pistachio-Reachable: legacy B100 full run plus post-change B500 on the 9 failures.
    reachable_base = load_records(project_path("output_Pistachio Reachable/pistachio_reachable_V3_W5_B100_merged.json"))
    reachable_supp = load_records(
        project_path("output_Pistachio Reachable/pistachio_reachable_prompt_compact_failed9_B500_w3_merged.json")
    )
    reachable_root_fix = load_records_optional(
        project_path("output_Pistachio Reachable/pistachio_reachable_root_init_fix_retry_merged.json")
    )
    reachable_base_success = success_indices(reachable_base)
    reachable_supp_map = merge_preferred_maps(record_map(reachable_supp), record_map(reachable_root_fix))
    reachable_budget: Dict[int, set[int]] = {}
    for budget in [100, 300, 500]:
        solved = set(reachable_base_success)
        solved.update(idx for idx, record in reachable_supp_map.items() if solve_by_iteration(record, budget))
        reachable_budget[budget] = solved
    reachable_records = []
    reachable_base_map = record_map(reachable_base)
    for idx in range(1, 151):
        supp = reachable_supp_map.get(idx)
        final = supp if supp is not None else reachable_base_map[idx]
        reachable_records.append(
            {
                "target_index": idx,
                "target": final.get("target"),
                "legacy_b100_success": idx in reachable_base_success,
                "post_change_tested": supp is not None,
                "success_at_budget_100": idx in reachable_budget[100],
                "success_at_budget_300": idx in reachable_budget[300],
                "success_at_budget_500": idx in reachable_budget[500],
                "final_source": "failed9_or_rootfix_b500" if supp else "legacy_v3_w5_b100",
                "final_actual_iterations": final.get("actual_iterations"),
            }
        )
    reachable_failed_500 = set(range(1, 151)) - reachable_budget[500]
    reachable_summary = {
        "dataset": "Pistachio-Reachable",
        "legacy_b100_solved": len(reachable_base_success),
        "post_change_coverage": "failed targets only: 9/150",
        "budget_stats": {
            str(budget): {
                "solved_count": len(solved),
                "failed_indices": sorted(set(range(1, 151)) - solved),
            }
            for budget, solved in reachable_budget.items()
        },
        "source_note": (
            "Legacy V3_W5_B100 had 9 failures; prompt-compact B500 was run only on those failures. "
            "Root-initialization bugfix retries are treated as replacements for affected root-init failures."
        ),
    }
    write_provenance_consolidated(output_dir, "Pistachio-Reachable", 150, reachable_records, reachable_summary)

    rows = [
        provenance_row(
            "USPTO-190",
            190,
            len(uspto_base_success),
            uspto_budget,
            uspto_failed_500,
            "failed 8 only",
            uspto_summary["source_note"],
        ),
        provenance_row(
            "Pistachio-Hard",
            100,
            len(hard_legacy_success),
            hard_budget,
            hard_failed_500,
            "all 100 retested/continued",
            hard_summary["source_note"],
        ),
        provenance_row(
            "Pistachio-Reachable",
            150,
            len(reachable_base_success),
            reachable_budget,
            reachable_failed_500,
            "failed 9 only",
            reachable_summary["source_note"],
        ),
    ]
    write_rows_csv(output_dir / "provenance_corrected_main_results_table.csv", rows)
    write_markdown_table(
        output_dir / "provenance_corrected_main_results_table.md",
        rows,
        [
            "Dataset",
            "Legacy V3_W5 B100",
            "Post-change coverage",
            "B100",
            "B300",
            "B500",
            "Failed@500",
        ],
    )
    (output_dir / "provenance_corrected_main_results_table.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "USPTO-190": uspto_summary,
        "Pistachio-Hard": hard_summary,
        "Pistachio-Reachable": reachable_summary,
    }


def write_global_outputs(payloads: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for payload in payloads.values():
        summary = payload["summary"]
        row = {"Dataset": summary["dataset"]}
        for budget in ["100", "300", "500"]:
            stats = summary["budget_stats"][budget]
            row[f"B{budget}"] = f"{stats['solved_count']}/{summary['total_targets']} ({stats['success_rate']:.1%})"
        row["Failed@500"] = ",".join(map(str, summary["final_failed_at_500"]))
        result_rows.append(row)
    write_rows_csv(output_dir / "main_results_table.csv", result_rows)
    write_markdown_table(output_dir / "main_results_table.md", result_rows, ["Dataset", "B100", "B300", "B500", "Failed@500"])

    efficiency_rows = [efficiency_row(payload) for payload in payloads.values()]
    write_rows_csv(output_dir / "efficiency_table.csv", efficiency_rows)
    write_markdown_table(
        output_dir / "efficiency_table.md",
        efficiency_rows,
        [
            "dataset",
            "solved_at_500",
            "total_targets",
            "total_llm_calls_from_records",
            "total_tokens_from_records",
            "avg_tokens_per_solved",
            "avg_iterations_per_solved",
            "legal_prefix_acceptance_rate",
            "template_exact",
            "template_topk",
            "template_unmatched",
            "llm_rejected",
            "root_init_failures",
        ],
    )

    failures = error_rows(payloads)
    write_rows_csv(output_dir / "error_analysis.csv", failures)
    grouped: Dict[str, int] = {}
    for row in failures:
        key = f"{row['dataset']}::{row['failure_type']}"
        grouped[key] = grouped.get(key, 0) + 1
    error_summary = [
        {"dataset": key.split("::", 1)[0], "failure_type": key.split("::", 1)[1], "count": count}
        for key, count in sorted(grouped.items())
    ]
    write_rows_csv(output_dir / "error_analysis_summary.csv", error_summary)
    write_markdown_table(output_dir / "error_analysis_summary.md", error_summary, ["dataset", "failure_type", "count"])

    case_specs = [
        ("Pistachio-Hard", 15),
        ("Pistachio-Hard", 46),
        ("Pistachio-Hard", 42),
        ("Pistachio-Reachable", 1),
        ("USPTO-190", 73),
    ]
    case_rows = []
    by_dataset = {payload["summary"]["dataset"]: payload for payload in payloads.values()}
    for dataset, index in case_specs:
        payload = by_dataset.get(dataset)
        if not payload:
            continue
        record = find_record(payload, index)
        if record:
            row = {"dataset": dataset}
            row.update(extract_case_summary(record))
            case_rows.append(row)
    (output_dir / "case_studies.json").write_text(json.dumps(case_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(output_dir / "case_studies.csv", case_rows)
    write_ablation_commands(output_dir / "ablation_commands.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate HSE-Retro experiment outputs for paper tables.")
    parser.add_argument("--output-dir", default="output/experiment_analysis")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), choices=sorted(DATASETS))
    args = parser.parse_args()

    output_dir = project_path(args.output_dir)
    payloads: Dict[str, Dict[str, Any]] = {}
    for dataset_key in args.datasets:
        config = DATASETS[dataset_key]
        payload = consolidate_dataset(config)
        write_dataset_outputs(config, payload)
        payloads[dataset_key] = payload

    write_global_outputs(payloads, output_dir)
    provenance = build_provenance_outputs(output_dir)
    print(f"Wrote consolidated experiment analysis to {output_dir}")
    for payload in payloads.values():
        summary = payload["summary"]
        b500 = summary["budget_stats"]["500"]
        print(
            f"{summary['dataset']}: B500 {b500['solved_count']}/{summary['total_targets']} "
            f"failed={b500['failed_indices']}"
        )
    print("Corrected provenance-aware main table:")
    for name, summary in provenance.items():
        b500 = summary["budget_stats"]["500"]
        print(f"{name}: B500 {b500['solved_count']} failed={b500['failed_indices']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
