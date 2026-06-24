from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, List

from planner.paroutes_io import load_smiles_file, save_json
from planner.route import SynthesisRoute


def experience_snapshot(planner=None) -> Dict[str, Any]:
    if planner is None:
        return {
            "final_macro_experience": "",
            "final_macro_experience_chars": 0,
            "micro_experience_nodes": [],
            "micro_experience_node_count": 0,
            "micro_experience_total_chars": 0,
        }

    micro_nodes = []
    for smiles, node in sorted(getattr(planner, "node_registry", {}).items()):
        micro_experience = getattr(node, "micro_experience", "") or ""
        if not micro_experience.strip():
            continue
        micro_nodes.append(
            {
                "smiles": smiles,
                "is_purchasable": getattr(node, "is_purchasable", False),
                "is_solved": getattr(node, "is_solved", False),
                "visit_count": getattr(node, "visit_count", 0),
                "value": getattr(node, "value", 0.0),
                "children_count": len(getattr(node, "children", [])),
                "parents_count": len(getattr(node, "parents", [])),
                "micro_experience": micro_experience,
                "micro_experience_chars": len(micro_experience),
            }
        )

    macro_experience = getattr(planner, "macro_experience", "") or ""
    return {
        "final_macro_experience": macro_experience,
        "final_macro_experience_chars": len(macro_experience),
        "micro_experience_nodes": micro_nodes,
        "micro_experience_node_count": len(micro_nodes),
        "micro_experience_total_chars": sum(
            item["micro_experience_chars"] for item in micro_nodes
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the research retrosynthesis planner on a plain target SMILES file."
    )
    parser.add_argument(
        "--targets",
        default="data/test_sets/USPTO-190.smi",
        help="One-SMILES-per-line target file.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="output/aot_strict_results.json")
    parser.add_argument(
        "--provider",
        default="qian_duo_duo",
        help="LLM provider. Current experiments use qian_duo_duo.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v3",
        help=(
            "Main LLM model or alias, e.g. gpt-4o, deepseek-v3, "
            "deepseek-v4-flash, deepseek-v4-pro."
        ),
    )
    parser.add_argument(
        "--critic-provider",
        default="qian_duo_duo",
        help="Provider for the debate critic model.",
    )
    parser.add_argument(
        "--critic-model",
        default="deepseek-v4-flash",
        help="Model or alias for the debate critic model.",
    )
    parser.add_argument("--budget", type=int, default=None, help="Search iterations.")
    parser.add_argument(
        "--route-width",
        "--expansion-width",
        dest="route_width",
        type=int,
        default=None,
        help="LLM route proposals per expansion.",
    )
    parser.add_argument(
        "--rag-top-k",
        "--retrieval-size",
        dest="rag_top_k",
        type=int,
        default=None,
        help="Retrieved route examples per prompt.",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--template-top-k", type=int, default=None)
    parser.add_argument(
        "--candidate-width",
        type=int,
        default=None,
        help="Candidate cuts shown per active molecule.",
    )
    parser.add_argument("--candidate-template-top-k", type=int, default=None)
    return parser.parse_args()


def route_record(target: str, route: SynthesisRoute | None, planner=None) -> Dict[str, Any]:
    diagnostics = {
        "validation_stats": getattr(planner, "validation_stats", {}),
        "invalid_route_examples": getattr(planner, "invalid_route_examples", []),
        "valid_route_examples": getattr(planner, "valid_route_examples", []),
        "experience_snapshot": experience_snapshot(planner),
        "frontier_snapshot": planner.frontier_snapshot() if planner else {},
        "dead_end_snapshot": planner.dead_end_snapshot() if planner else [],
    }
    if route is None:
        return {
            "target": target,
            "is_successful": False,
            "reward": 0.0,
            "steps": [],
            **diagnostics,
        }

    report = getattr(route, "validation_report", None)
    return {
        "target": target,
        "is_successful": route.is_successful,
        "reward": route.reward,
        "steps": [asdict(step) for step in route.steps],
        "validation": {
            "steps_valid": getattr(report, "steps_valid", False),
            "terminal_molecules": getattr(report, "terminal_molecules", []),
            "non_purchasable": getattr(report, "non_purchasable", []),
            "first_invalid_index": getattr(report, "first_invalid_index", -1),
        },
        **diagnostics,
    }


def main() -> None:
    args = parse_args()
    targets = load_smiles_file(args.targets, limit=args.limit)

    from config import default_config

    config_snapshot = default_config.configure_experiment(
        provider=args.provider,
        model=args.model,
        critic_provider=args.critic_provider,
        critic_model=args.critic_model,
        budget=args.budget,
        route_width=args.route_width,
        rag_top_k=args.rag_top_k,
        max_search_depth=args.max_depth,
        temperature=args.temperature,
        template_top_k=args.template_top_k,
        candidate_width=args.candidate_width,
        candidate_template_top_k=args.candidate_template_top_k,
    )
    print(f"[Runner][INFO] config={config_snapshot}")

    from planner.strategic_search import StrategicRetrosynthesisPlanner

    records: List[Dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        print(f"[Runner][INFO] target={index}/{len(targets)}, smiles={target}")
        planner = StrategicRetrosynthesisPlanner(target)
        route = planner.run()
        records.append(route_record(target, route, planner))
        print(f"[Runner][INFO] target={index}, solved={bool(route and route.is_successful)}")

    save_json(records, args.output)
    solved_count = sum(1 for item in records if item["is_successful"])
    print(f"[Runner][INFO] saved={args.output}, targets={len(records)}, solved={solved_count}")


if __name__ == "__main__":
    main()
