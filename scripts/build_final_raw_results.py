from __future__ import annotations

import csv
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output_final_main_results"


def project_path(path: str) -> Path:
    return ROOT / path


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    return records


def record_index(record: Dict[str, Any]) -> int:
    return int(record["target_index"])


def record_map(path: Path, source_stage: str) -> Dict[int, Dict[str, Any]]:
    mapped: Dict[int, Dict[str, Any]] = {}
    for record in load_records(path):
        idx = record_index(record)
        cloned = copy.deepcopy(record)
        cloned["_final_source_stage"] = source_stage
        cloned["_final_source_file"] = str(path.relative_to(ROOT))
        mapped[idx] = cloned
    return mapped


def solved_within(record: Optional[Dict[str, Any]], budget: int) -> bool:
    if not record or not record.get("is_successful"):
        return False
    iterations = record.get("actual_iterations")
    if iterations is None:
        return False
    return int(iterations) <= budget


def success_rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def attach_final_metadata(
    dataset: str,
    record: Dict[str, Any],
    integration_rule: str,
) -> Dict[str, Any]:
    record = copy.deepcopy(record)
    source_stage = record.pop("_final_source_stage")
    source_file = record.pop("_final_source_file")
    record["final_metadata"] = {
        "dataset": dataset,
        "target_index": record_index(record),
        "selected_source_stage": source_stage,
        "selected_source_file": source_file,
        "integration_rule": integration_rule,
        "success_at_budget_100": solved_within(record, 100),
        "success_at_budget_300": solved_within(record, 300),
        "success_at_budget_500": solved_within(record, 500),
        "final_success": bool(record.get("is_successful")),
        "final_actual_iterations": record.get("actual_iterations"),
        "root_init_bugfix_record": source_stage == "root_init_fix_retry",
    }
    return record


def require_coverage(records: Dict[int, Dict[str, Any]], total_targets: int, dataset: str) -> None:
    expected = set(range(1, total_targets + 1))
    actual = set(records)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{dataset} coverage mismatch: missing={missing}, extra={extra}")


def summarize(
    dataset: str,
    total_targets: int,
    records: List[Dict[str, Any]],
    source_files: List[str],
    integration_rule: str,
) -> Dict[str, Any]:
    budget_stats: Dict[str, Dict[str, Any]] = {}
    for budget in [100, 300, 500]:
        solved = [
            record_index(record)
            for record in records
            if record["final_metadata"][f"success_at_budget_{budget}"]
        ]
        failed = sorted(set(range(1, total_targets + 1)) - set(solved))
        budget_stats[str(budget)] = {
            "solved_count": len(solved),
            "success_rate": success_rate(len(solved), total_targets),
            "failed_indices": failed,
        }

    rootfix_indices = [
        record_index(record)
        for record in records
        if record["final_metadata"]["root_init_bugfix_record"]
        and record["final_metadata"]["success_at_budget_500"]
    ]
    source_stage_counts: Dict[str, int] = {}
    for record in records:
        stage = record["final_metadata"]["selected_source_stage"]
        source_stage_counts[stage] = source_stage_counts.get(stage, 0) + 1

    return {
        "dataset": dataset,
        "framework": "HSE-Retro",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_targets": total_targets,
        "integration_rule": integration_rule,
        "source_files": source_files,
        "source_stage_counts": source_stage_counts,
        "root_init_bugfix_success_indices": sorted(rootfix_indices),
        "budget_stats": budget_stats,
        "final_failed_at_500": budget_stats["500"]["failed_indices"],
    }


def write_dataset(
    slug: str,
    dataset: str,
    total_targets: int,
    selected_records: Dict[int, Dict[str, Any]],
    source_files: List[Path],
    integration_rule: str,
) -> Dict[str, Any]:
    require_coverage(selected_records, total_targets, dataset)
    records = [
        attach_final_metadata(dataset, selected_records[idx], integration_rule)
        for idx in range(1, total_targets + 1)
    ]
    source_file_names = [str(path.relative_to(ROOT)) for path in source_files]
    summary = summarize(dataset, total_targets, records, source_file_names, integration_rule)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": records}
    (OUTPUT_DIR / f"{slug}_final_raw_records.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / f"{slug}_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_rows = []
    for record in records:
        meta = record["final_metadata"]
        csv_rows.append(
            {
                "target_index": meta["target_index"],
                "target": record.get("target", ""),
                "source_stage": meta["selected_source_stage"],
                "source_file": meta["selected_source_file"],
                "is_successful": record.get("is_successful"),
                "actual_iterations": record.get("actual_iterations"),
                "success_at_budget_100": meta["success_at_budget_100"],
                "success_at_budget_300": meta["success_at_budget_300"],
                "success_at_budget_500": meta["success_at_budget_500"],
                "root_init_bugfix_record": meta["root_init_bugfix_record"],
                "steps": len(record.get("steps") or []),
            }
        )
    write_csv(OUTPUT_DIR / f"{slug}_final_per_target.csv", csv_rows)
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_uspto() -> Dict[str, Any]:
    base_path = project_path("output_USPTO/uspto190_V3_W5_B100_combined_with_initialmacro_retry.json")
    retry_path = project_path("output_USPTO/uspto190_prompt_compact_failed8_B500_merged.json")
    rootfix_path = project_path("output_USPTO/uspto190_root_init_fix_retry_merged.json")

    selected = record_map(base_path, "base_b100_combined")
    retry = record_map(retry_path, "failed8_b500_retry")
    rootfix = record_map(rootfix_path, "root_init_fix_retry")

    for idx, record in retry.items():
        selected[idx] = record
    for idx, record in rootfix.items():
        if record.get("is_successful") or idx not in selected:
            selected[idx] = record

    return write_dataset(
        "uspto190",
        "USPTO-190",
        190,
        selected,
        [base_path, retry_path, rootfix_path],
        (
            "Base records come from the combined B100 run. Records for the original "
            "failed targets are replaced by the B500 retry, and root-initialization "
            "bugfix retry records replace corresponding failed retry records when successful."
        ),
    )


def build_hard() -> Dict[str, Any]:
    old88_path = project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_oldsuccess88_B500_merged.json")
    old88_tail_path = project_path(
        "output_Pistachio Hard/pistachio_hard_prompt_compact_oldsuccess88_B500_tail20_merged.json"
    )
    failed12_path = project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_failed12_merged.json")
    failed6_path = project_path("output_Pistachio Hard/pistachio_hard_prompt_compact_failed6_B500_merged.json")
    rootfix_path = project_path("output_Pistachio Hard/pistachio_hard_root_init_fix_retry_merged.json")

    selected: Dict[int, Dict[str, Any]] = {}
    for stage_map in [
        record_map(old88_path, "old_success88_current_b500_retest"),
        record_map(old88_tail_path, "old_success88_current_b500_tail_retest"),
        record_map(failed12_path, "old_failed12_current_b100_retest"),
    ]:
        overlap = set(selected) & set(stage_map)
        if overlap:
            raise ValueError(f"Pistachio-Hard duplicate records before retry replacement: {sorted(overlap)}")
        selected.update(stage_map)

    failed6 = record_map(failed6_path, "failed6_b500_retry")
    rootfix = record_map(rootfix_path, "root_init_fix_retry")
    for idx, record in failed6.items():
        selected[idx] = record
    for idx, record in rootfix.items():
        if record.get("is_successful") or idx not in selected:
            selected[idx] = record

    return write_dataset(
        "pistachio_hard",
        "Pistachio-Hard",
        100,
        selected,
        [old88_path, old88_tail_path, failed12_path, failed6_path, rootfix_path],
        (
            "The current implementation covers all 100 targets. The originally solved "
            "88 targets use the current B500 retest records. The originally failed 12 "
            "targets use the current B100 retest, then failed-target B500 retry records, "
            "with successful root-initialization bugfix retry records taking priority."
        ),
    )


def build_reachable() -> Dict[str, Any]:
    base_path = project_path("output_Pistachio Reachable/pistachio_reachable_V3_W5_B100_merged.json")
    retry_path = project_path(
        "output_Pistachio Reachable/pistachio_reachable_prompt_compact_failed9_B500_w3_merged.json"
    )
    rootfix_path = project_path(
        "output_Pistachio Reachable/pistachio_reachable_root_init_fix_retry_merged.json"
    )

    selected = record_map(base_path, "base_b100")
    retry = record_map(retry_path, "failed9_b500_retry")
    rootfix = record_map(rootfix_path, "root_init_fix_retry")
    for idx, record in retry.items():
        selected[idx] = record
    for idx, record in rootfix.items():
        if record.get("is_successful") or idx not in selected:
            selected[idx] = record

    return write_dataset(
        "pistachio_reachable",
        "Pistachio-Reachable",
        150,
        selected,
        [base_path, retry_path, rootfix_path],
        (
            "Base records come from the B100 run. Records for the original failed "
            "targets are replaced by the B500 retry, and root-initialization bugfix "
            "retry records replace corresponding failed retry records when successful."
        ),
    )


def main() -> int:
    summaries = [build_uspto(), build_hard(), build_reachable()]
    rows = []
    for summary in summaries:
        total = summary["total_targets"]
        row = {
            "Dataset": summary["dataset"],
            "B100": f"{summary['budget_stats']['100']['solved_count']}/{total} ({summary['budget_stats']['100']['success_rate']:.1%})",
            "B300": f"{summary['budget_stats']['300']['solved_count']}/{total} ({summary['budget_stats']['300']['success_rate']:.1%})",
            "B500": f"{summary['budget_stats']['500']['solved_count']}/{total} ({summary['budget_stats']['500']['success_rate']:.1%})",
            "RootInitFix": ",".join(map(str, summary["root_init_bugfix_success_indices"])),
            "Failed@500": ",".join(map(str, summary["final_failed_at_500"])),
        }
        rows.append(row)

    write_csv(OUTPUT_DIR / "final_main_results_table.csv", rows)
    write_markdown_table(
        OUTPUT_DIR / "final_main_results_table.md",
        rows,
        ["Dataset", "B100", "B300", "B500", "RootInitFix", "Failed@500"],
    )
    (OUTPUT_DIR / "final_main_results_table.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote final raw main-result records to {OUTPUT_DIR}")
    for row in rows:
        print(
            f"{row['Dataset']}: B100={row['B100']}, B300={row['B300']}, "
            f"B500={row['B500']}, rootfix={row['RootInitFix']}, failed={row['Failed@500']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
