from __future__ import annotations

import json
from typing import Dict, List, Optional

from config import default_config
from data.database import db_instance
from planner.candidate_guided import (
    CandidateGraph,
    TemplateCandidate,
    build_experience_guided_candidate_graph,
    format_candidate_graph_context,
)
from planner.experience_memory import StructuredExperienceMemory
from planner.frontier import (
    compute_and_frontier_metrics,
    compute_or_frontier_metrics,
    format_frontier_context,
)
from planner.operators import (
    canonicalize_smiles,
    generate_initial_macro_experience,
    generate_pathways,
)
from planner.route import ReactionStep, SynthesisRoute
from planner.tree_node import ANDNode, ORNode
from planner.validation import RouteValidationReport, validate_synthesis_route

try:
    from chem_utils.scoring import get_sc_score
except ImportError:
    def get_sc_score(smiles: str) -> float:
        return 5.0


def _policy(config, name: str) -> Dict:
    value = getattr(config, name, {})
    return value if isinstance(value, dict) else {}


class StrategicRetrosynthesisPlanner:
    """Experience-guided retrosynthesis planner with strict tree insertion.

    The LLM is allowed to propose routes, but a route can affect search only
    after it passes molecule/reaction/connectivity validation. The
    macro/micro experience mechanism then learns from accepted legal cuts and
    from rejected illegal proposals.
    """

    def __init__(self, target_molecule: str, config=default_config):
        self.target_smiles = canonicalize_smiles(target_molecule)
        self.config = config
        self.node_registry: Dict[str, ORNode] = {}
        self.root = self._get_or_create_or_node(self.target_smiles)

        search_policy = _policy(config, "SEARCH_POLICY")
        validation_policy = _policy(config, "VALIDATION_POLICY")
        candidate_policy = _policy(config, "CANDIDATE_GRAPH_POLICY")
        experience_policy = _policy(config, "EXPERIENCE_POLICY")
        ablation_policy = _policy(config, "ABLATION_POLICY")

        self.disable_macro_experience = bool(
            ablation_policy.get("disable_macro_experience", False)
        )
        self.disable_micro_experience = bool(
            ablation_policy.get("disable_micro_experience", False)
        )
        self.disable_candidate_cuts = bool(
            ablation_policy.get("disable_candidate_cuts", False)
        )
        self.disable_initial_macro_debate = bool(
            ablation_policy.get("disable_initial_macro_debate", False)
        )

        self.c_param = 0.5
        self.alpha = search_policy.get(
            "availability_weight",
            getattr(config, "AVAILABILITY_WEIGHT", 0.4),
        )
        self.prune_cycles = True
        self.prune_same_molecule_reactants = True
        self.prune_dead_end_branches = True
        self.dead_end_max_expansion_failures = search_policy.get(
            "dead_end_max_expansion_failures",
            getattr(config, "DEAD_END_MAX_EXPANSION_FAILURES", 999),
        )
        self.root_llm_retry_attempts = search_policy.get(
            "root_retry_attempts",
            getattr(config, "ROOT_LLM_RETRY_ATTEMPTS", 3),
        )
        self.llm_pathway_generation = True
        self.llm_pathway_width = getattr(
            config,
            "ROUTE_WIDTH",
            getattr(config, "LLM_PATHWAY_WIDTH", getattr(config, "EXPANSION_WIDTH", 3)),
        )
        self.llm_pathway_use_rag = True
        self.candidate_graph_enabled = not self.disable_candidate_cuts
        self.candidate_graph_depth = candidate_policy.get(
            "depth",
            getattr(config, "CANDIDATE_GRAPH_DEPTH", 0),
        )
        self.candidate_graph_width = getattr(
            config,
            "CANDIDATE_WIDTH",
            getattr(config, "CANDIDATE_GRAPH_WIDTH", 5),
        )
        self.candidate_graph_max_nodes = candidate_policy.get(
            "max_nodes",
            getattr(config, "CANDIDATE_GRAPH_MAX_NODES", 6),
        )
        self.candidate_graph_retrieval_size = candidate_policy.get(
            "rag_top_k",
            getattr(config, "CANDIDATE_GRAPH_RETRIEVAL_SIZE", 8),
        )
        self.candidate_graph_template_top_k = getattr(
            config,
            "CANDIDATE_TEMPLATE_TOP_K",
            getattr(config, "CANDIDATE_GRAPH_TEMPLATE_TOP_K", 1000),
        )
        self.candidate_graph_direct_max_templates = candidate_policy.get(
            "direct_max_templates",
            getattr(config, "CANDIDATE_GRAPH_DIRECT_MAX_TEMPLATES", 0),
        )
        self.candidate_graph_applicable_template_limit = candidate_policy.get(
            "applicable_template_limit",
            getattr(config, "CANDIDATE_GRAPH_APPLICABLE_TEMPLATE_LIMIT", 2000),
        )
        self.candidate_graph_max_outcomes = candidate_policy.get(
            "max_outcomes",
            getattr(config, "CANDIDATE_GRAPH_MAX_OUTCOMES", 20),
        )
        self.candidate_graph_max_chars = candidate_policy.get(
            "max_chars",
            getattr(config, "CANDIDATE_GRAPH_MAX_CHARS", 5200),
        )
        self.candidate_graph_prune_nonprogressive = True
        self.candidate_graph_max_lateral_risk = candidate_policy.get(
            "max_lateral_risk",
            getattr(config, "CANDIDATE_GRAPH_MAX_LATERAL_RISK", 0.88),
        )
        self.max_depth = getattr(config, "MAX_SEARCH_DEPTH", 16)
        self.max_accepted_prefix_steps = search_policy.get(
            "max_accepted_prefix_steps",
            getattr(config, "MAX_ACCEPTED_PREFIX_STEPS", 3),
        )
        self.expansion_num = getattr(config, "ROUTE_WIDTH", getattr(config, "EXPANSION_WIDTH", 3))
        self.template_backed_validation = True
        self.template_match_top_k = getattr(
            config,
            "TEMPLATE_TOP_K",
            getattr(config, "TEMPLATE_MATCH_TOP_K", 100),
        )
        self.require_template_db_match = True
        self.validation_max_outcomes = validation_policy.get(
            "max_outcomes",
            getattr(config, "VALIDATION_MAX_OUTCOMES", 5000),
        )
        self.template_generated_reactants_as_source_of_truth = True
        self.experience_max_context_chars = experience_policy.get(
            "max_context_chars",
            getattr(config, "EXPERIENCE_MAX_CONTEXT_CHARS", 5200),
        )
        self.memory_hard_gate_enabled = True
        self.node_backoff_enabled = False

        self.macro_experience = ""
        self.initial_macro_experience = ""
        self.experience_memory = StructuredExperienceMemory(
            max_valid_cuts=experience_policy.get("top_valid_cuts", 3),
            max_failed_cuts=experience_policy.get("top_failed_cuts", 5),
            max_taboos=experience_policy.get("top_taboos", 3),
            max_macro_patterns=experience_policy.get("top_macro_patterns", 5),
        )
        self.experience_trace: List[Dict] = []
        self.all_created_and_nodes: List[ANDNode] = []
        self.valid_route_examples: List[str] = []
        self.invalid_route_examples: List[str] = []
        self._candidate_graph_cache: Dict[str, str] = {}
        self._candidate_graph_object_cache: Dict[str, CandidateGraph] = {}
        self.validation_stats = {
            "generated": 0,
            "accepted": 0,
            "rejected": 0,
            "successful": 0,
            "template_exact": 0,
            "template_substructure": 0,
            "template_topk": 0,
            "template_unmatched": 0,
            "candidate_graph_calls": 0,
            "candidate_graph_nodes": 0,
            "candidate_graph_candidates": 0,
            "candidate_graph_empty": 0,
            "llm_pathway_generated": 0,
            "llm_pathway_accepted": 0,
            "llm_pathway_rejected": 0,
            "llm_pathway_prefix_accepted": 0,
            "accepted_prefix_routes": 0,
            "discarded_suffix_steps": 0,
            "llm_free_generated": 0,
            "llm_free_accepted": 0,
            "llm_free_rejected": 0,
            "cycle_rejected": 0,
            "same_molecule_rejected": 0,
            "pruned_routes": 0,
            "dead_end_marked": 0,
            "dead_end_expansion_failures": 0,
            "llm_no_valid_prefix_failures": 0,
            "soft_expansion_failures": 0,
            "depth_limited_routes": 0,
            "prefix_step_limited": 0,
            "memory_gate_rejected": 0,
            "memory_avoid_rejected": 0,
            "memory_self_reactant_rejected": 0,
            "memory_ancestor_rejected": 0,
            "backoff_applied": 0,
            "backoff_skips": 0,
            "root_product_repair_applied": 0,
            "root_candidate_fallback_generated": 0,
            "root_candidate_fallback_accepted": 0,
        }
        self.actual_iterations = 0
        self._last_mapped_and_nodes: List[ANDNode] = []

    def run(self) -> Optional[SynthesisRoute]:
        print(f"[Search][INFO] Started target={self.target_smiles}")
        from llm.client import get_llm_client

        client = get_llm_client()
        start_stats = client.get_usage_snapshot()

        root_candidate_graph_context = self._candidate_graph_context_for_generation(
            self.target_smiles
        )
        self.initial_macro_experience = self._initialize_macro_experience(
            root_candidate_graph_context
        )
        self.macro_experience = self.initial_macro_experience

        print("[Search][INFO] Root expansion with strict route validation.")
        initial_batch_nodes: List[ANDNode] = []
        root_routes_for_repair: List[SynthesisRoute] = []

        if self.llm_pathway_generation:
            initial_routes = generate_pathways(
                self.target_smiles,
                n_routes=self.llm_pathway_width,
                experience=self.macro_experience,
                context="",
                frontier_context=self._format_current_frontier_context(),
                candidate_graph_context=root_candidate_graph_context,
                enable_rag=self.llm_pathway_use_rag,
            )
            self._tag_routes(initial_routes, "llm_pathway")
            root_routes_for_repair.extend(initial_routes)
            batch_nodes, _ = self._ingest_routes(
                routes=initial_routes,
                start_or_node=self.root,
                start_smiles=self.target_smiles,
                base_depth=0,
            )
            initial_batch_nodes.extend(batch_nodes)
            if batch_nodes:
                print(
                    f"[Search][INFO] Accepted root branches: count={len(batch_nodes)}"
                )

        for attempt in range(max(0, self.root_llm_retry_attempts - 1)):
            if initial_batch_nodes:
                break
            try:
                retry_experience = self._format_initial_macro_context()
                if self.invalid_route_examples:
                    retry_experience = (
                        retry_experience
                        + ("\n\n" if retry_experience else "")
                        + "[Previous root proposals rejected by strict template validation]\n"
                        + "\n".join(self.invalid_route_examples[-8:])
                        + "\nAvoid repeating these Product -> Reactants cuts unless you can make them database-template-backed."
                    )
                initial_routes = generate_pathways(
                    self.target_smiles,
                    n_routes=self.llm_pathway_width,
                    experience=retry_experience,
                    candidate_graph_context=root_candidate_graph_context,
                    enable_rag=True,
                )
                self._tag_routes(initial_routes, "llm_pathway")
                root_routes_for_repair.extend(initial_routes)
                batch_nodes, _ = self._ingest_routes(
                    routes=initial_routes,
                    start_or_node=self.root,
                    start_smiles=self.target_smiles,
                    base_depth=0,
                )
                initial_batch_nodes.extend(batch_nodes)

                if initial_batch_nodes:
                    break
                print(f"[Search][WARN] Root attempt {attempt + 1} produced no legal route.")
                if self.invalid_route_examples:
                    print(f"[Search][DEBUG] Last rejection: {self.invalid_route_examples[-1][:240]}")
            except Exception as exc:
                print(f"[Search][ERROR] Root attempt {attempt + 1} failed: {exc}")

        if not initial_batch_nodes:
            repairable_routes = [
                route
                for route in root_routes_for_repair
                if route is not None and route.steps and not route.steps[0].product
            ]
            if repairable_routes:
                print(
                    "[Search][INFO] Trying root product-field repair: "
                    f"routes={len(repairable_routes)}"
                )
                batch_nodes, _ = self._ingest_routes(
                    routes=repairable_routes,
                    start_or_node=self.root,
                    start_smiles=self.target_smiles,
                    base_depth=0,
                    repair_missing_root_product=True,
                )
                initial_batch_nodes.extend(batch_nodes)

        if not initial_batch_nodes:
            fallback_routes = self._candidate_root_seed_routes(self.target_smiles)
            if fallback_routes:
                print(
                    "[Search][INFO] Trying candidate-cut root fallback: "
                    f"routes={len(fallback_routes)}"
                )
                batch_nodes, _ = self._ingest_routes(
                    routes=fallback_routes,
                    start_or_node=self.root,
                    start_smiles=self.target_smiles,
                    base_depth=0,
                    repair_missing_root_product=False,
                )
                initial_batch_nodes.extend(batch_nodes)
                self.validation_stats["root_candidate_fallback_accepted"] += len(
                    batch_nodes
                )

        if not initial_batch_nodes:
            print("[Search][FATAL] Root initialization failed after strict validation.")
            end_stats = client.get_usage_snapshot()
            self._save_logs(start_stats, end_stats)
            return self.extract_best_route()

        self._score_new_nodes(initial_batch_nodes)
        self.root.update_solved_status()
        legal_root_children = [
            child
            for child in self.root.children
            if child.is_valid
            and (
                child.is_solved
                or not self.prune_dead_end_branches
                or self._and_has_expandable_frontier(child)
            )
        ]
        if legal_root_children:
            self.root.value = max(child.value for child in legal_root_children)

        self.macro_experience = self._update_macro_from_root_validation(
            initial_batch_nodes,
            bad_examples=self.invalid_route_examples[-10:],
        )

        update_interval = getattr(self.config, "MACRO_UPDATE_INTERVAL", 10)

        for i in range(getattr(self.config, "BUDGET", 50)):
            self.actual_iterations = i + 1
            self.root.update_solved_status()
            if self.root.is_solved:
                print(f"[Search][INFO] Target solved at iteration {i}.")
                break

            if (
                not self.disable_macro_experience
                and i > 0
                and i % update_interval == 0
            ):
                print(f"[Experience][INFO] Updating macro strategy at iteration {i}.")
                self.macro_experience = self._update_macro_experience_periodic()

            selected_and = self._select()
            if not selected_and:
                frontier = self.frontier_snapshot()
                print(
                    f"[Search][INFO] Iter {i}: no viable legal nodes left "
                    f"(non_purch={frontier.get('non_purchasable_terminal_count', 0)}, "
                    f"dead_end_nodes={len(self.dead_end_snapshot())})."
                )
                break

            self._expand(selected_and)

            if i % 5 == 0:
                micro_exp_count = sum(
                    1
                    for node in self.node_registry.values()
                    if getattr(node, "micro_experience", "")
                )
                print(
                    f"[Search][METRIC] iter={i}, root_value={self.root.value:.4f}, "
                    f"macro_chars={len(self.macro_experience)}, "
                    f"micro_nodes={micro_exp_count}, "
                    f"Accepted/Rejected={self.validation_stats['accepted']}/"
                    f"{self.validation_stats['rejected']}, "
                    f"Template exact/sub/topk/unmatched="
                    f"{self.validation_stats['template_exact']}/"
                    f"{self.validation_stats['template_substructure']}/"
                    f"{self.validation_stats['template_topk']}/"
                    f"{self.validation_stats['template_unmatched']}, "
                    f"CGraph calls/nodes/cands/empty="
                    f"{self.validation_stats['candidate_graph_calls']}/"
                    f"{self.validation_stats['candidate_graph_nodes']}/"
                    f"{self.validation_stats['candidate_graph_candidates']}/"
                    f"{self.validation_stats['candidate_graph_empty']}, "
                    f"LLMPath gen/acc/prefix/rej="
                    f"{self.validation_stats['llm_pathway_generated']}/"
                    f"{self.validation_stats['llm_pathway_accepted']}/"
                    f"{self.validation_stats['llm_pathway_prefix_accepted']}/"
                    f"{self.validation_stats['llm_pathway_rejected']}, "
                    f"LLMFail soft/dead="
                    f"{self.validation_stats['soft_expansion_failures']}/"
                    f"{self.validation_stats['dead_end_expansion_failures']}, "
                    f"Pruned/cycle/dead="
                    f"{self.validation_stats['pruned_routes']}/"
                    f"{self.validation_stats['cycle_rejected']}/"
                    f"{self.validation_stats['dead_end_marked']}, "
                    f"MemGate/avoid/self/ancestor="
                    f"{self.validation_stats['memory_gate_rejected']}/"
                    f"{self.validation_stats['memory_avoid_rejected']}/"
                    f"{self.validation_stats['memory_self_reactant_rejected']}/"
                    f"{self.validation_stats['memory_ancestor_rejected']}"
                )

        end_stats = client.get_usage_snapshot()
        self._save_logs(start_stats, end_stats)
        return self.extract_best_route()

    def _validate_generated_route(
        self,
        route: SynthesisRoute,
        start_smiles: str,
    ) -> RouteValidationReport:
        return validate_synthesis_route(
            route=route,
            target_molecule=start_smiles,
            db=db_instance,
            require_template_db_match=self.require_template_db_match,
            template_backed_validation=self.template_backed_validation,
            template_match_top_k=self.template_match_top_k,
            max_outcomes=self.validation_max_outcomes,
            template_generated_reactants_as_source_of_truth=(
                self.template_generated_reactants_as_source_of_truth
            ),
        )

    def _candidate_graph_context_for_generation(self, start_smiles: str) -> str:
        if not self.candidate_graph_enabled or self.disable_candidate_cuts:
            return ""

        start = canonicalize_smiles(start_smiles)
        if not start:
            return ""
        cached = self._candidate_graph_cache.get(start)
        if cached is not None:
            return cached

        graph = self._build_candidate_graph(start)
        candidate_count = sum(len(node.candidates) for node in graph.nodes)
        self.validation_stats["candidate_graph_calls"] += 1
        self.validation_stats["candidate_graph_nodes"] += len(graph.nodes)
        self.validation_stats["candidate_graph_candidates"] += candidate_count
        if candidate_count == 0:
            self.validation_stats["candidate_graph_empty"] += 1

        context = format_candidate_graph_context(
            graph,
            max_nodes=self.candidate_graph_max_nodes,
            max_candidates_per_node=self.candidate_graph_width,
        )
        if len(context) > self.candidate_graph_max_chars:
            context = context[: self.candidate_graph_max_chars - 80] + "\n... [candidate graph truncated]"
        self._candidate_graph_cache[start] = context
        print(
            f"[CandidateGraph][INFO] product={start[:30]}, "
            f"nodes={len(graph.nodes)}, candidates={candidate_count}, "
            f"rag_routes={graph.rag_route_count}, "
            f"applicable_templates={graph.template_query_count}, "
            f"applied={graph.applied_template_count}."
        )
        return context

    def _build_candidate_graph(self, start_smiles: str) -> CandidateGraph:
        start = canonicalize_smiles(start_smiles)
        if not start:
            return CandidateGraph("", [], 0, 0, 0)

        cached = self._candidate_graph_object_cache.get(start)
        if cached is not None:
            return cached

        graph = build_experience_guided_candidate_graph(
            db=db_instance,
            product_smiles=start,
            graph_depth=self.candidate_graph_depth,
            per_node_width=self.candidate_graph_width,
            retrieval_size=self.candidate_graph_retrieval_size,
            template_top_k=self.candidate_graph_template_top_k,
            direct_max_templates=self.candidate_graph_direct_max_templates,
            applicable_template_limit=self.candidate_graph_applicable_template_limit,
            max_outcomes_per_template=self.candidate_graph_max_outcomes,
            max_nodes=self.candidate_graph_max_nodes,
            sc_score_fn=get_sc_score,
            experience_memory=self.experience_memory,
            prune_nonprogressive=self.candidate_graph_prune_nonprogressive,
            max_lateral_move_risk=self.candidate_graph_max_lateral_risk,
        )
        self._candidate_graph_object_cache[start] = graph
        return graph

    def _candidate_root_seed_routes(self, start_smiles: str) -> List[SynthesisRoute]:
        if not self.candidate_graph_enabled or self.disable_candidate_cuts:
            return []

        start = canonicalize_smiles(start_smiles)
        if not start:
            return []

        graph = self._candidate_graph_object_cache.get(start)
        if graph is None:
            graph = self._build_candidate_graph(start)

        root_nodes = [
            node
            for node in graph.nodes
            if canonicalize_smiles(node.product_smiles) == start
        ]
        if not root_nodes:
            return []

        routes: List[SynthesisRoute] = []
        for candidate in root_nodes[0].candidates[: self.candidate_graph_width]:
            route = self._route_from_root_candidate(candidate)
            if route is not None:
                routes.append(route)

        self.validation_stats["root_candidate_fallback_generated"] += len(routes)
        return routes

    @staticmethod
    def _route_from_root_candidate(
        candidate: TemplateCandidate,
    ) -> Optional[SynthesisRoute]:
        product = canonicalize_smiles(candidate.product_smiles)
        reactants = [
            canonicalize_smiles(item)
            for item in (candidate.reactants or [])
        ]
        reactants = [item for item in reactants if item]
        if not product or not reactants:
            return None

        step = ReactionStep(
            molecule_set=[product],
            rational=(
                "template-grounded root candidate "
                f"{candidate.candidate_id} from {candidate.source}"
            ),
            product=[product],
            reaction=candidate.template,
            reactants=reactants,
            updated_molecule_set=reactants,
        )
        route = SynthesisRoute(
            steps=[step],
            explanation=(
                "Root fallback seeded from a template-grounded candidate cut "
                "after LLM root initialization produced no legal prefix."
            ),
        )
        route.generation_source = "candidate_root_fallback"
        route.candidate_id = candidate.candidate_id
        return route

    def _ingest_routes(
        self,
        routes: List[SynthesisRoute],
        start_or_node: ORNode,
        start_smiles: str,
        base_depth: int = 0,
        repair_missing_root_product: bool = False,
    ) -> tuple[List[ANDNode], List[str]]:
        """Validate generated routes and insert only legal ones into the tree."""
        accepted_nodes: List[ANDNode] = []
        rejected_examples: List[str] = []

        for route in routes:
            if repair_missing_root_product:
                missing_first_product = bool(
                    route is not None
                    and route.steps
                    and not route.steps[0].product
                )
                repaired = self._repair_root_initialization_route(route, start_smiles)
                if repaired is not None:
                    route = repaired
                repaired_first_product = bool(
                    route is not None
                    and route.steps
                    and route.steps[0].product
                )
                if missing_first_product and repaired_first_product:
                    self.validation_stats["root_product_repair_applied"] += 1
            route_source = getattr(route, "generation_source", "llm_free")
            self.validation_stats["generated"] += 1
            if route_source == "llm_pathway":
                self.validation_stats["llm_pathway_generated"] += 1
            if route_source == "llm_free":
                self.validation_stats["llm_free_generated"] += 1
            pre_validation_gate_enabled = not (
                self.template_backed_validation
                and self.template_generated_reactants_as_source_of_truth
            )
            if pre_validation_gate_enabled:
                gate_reason = self._memory_gate_route(route, start_or_node)
                if gate_reason:
                    self.validation_stats["memory_gate_rejected"] += 1
                    if gate_reason.startswith("memory_avoid_reactant_set"):
                        self.validation_stats["memory_avoid_rejected"] += 1
                    elif gate_reason.startswith("memory_self_reactant"):
                        self.validation_stats["memory_self_reactant_rejected"] += 1
                    elif gate_reason.startswith("memory_ancestor_reactant"):
                        self.validation_stats["memory_ancestor_rejected"] += 1
                    self.validation_stats["rejected"] += 1
                    if route_source == "llm_pathway":
                        self.validation_stats["llm_pathway_rejected"] += 1
                    elif route_source == "llm_free":
                        self.validation_stats["llm_free_rejected"] += 1
                    example = self._memory_gate_experience(route, gate_reason, route_source)
                    rejected_examples.append(example)
                    self._remember(self.invalid_route_examples, [example], limit=80)
                    continue
            report = self._validate_generated_route(route, start_smiles)
            route.validation_report = report

            if report.invalid_steps:
                for invalid in report.invalid_steps:
                    if invalid.reason.startswith("no_database_template"):
                        self.validation_stats["template_unmatched"] += 1
                examples = report.bad_experience()
                rejected_examples.extend(examples)
                self._remember(self.invalid_route_examples, examples, limit=80)

            has_valid_prefix = (
                report.normalized_route is not None
                and bool(report.normalized_route.steps)
            )
            if not has_valid_prefix:
                self.validation_stats["rejected"] += 1
                if route_source == "llm_pathway":
                    self.validation_stats["llm_pathway_rejected"] += 1
                elif route_source == "llm_free":
                    self.validation_stats["llm_free_rejected"] += 1
                continue

            normalized_route, prune_reason = self._prune_route_to_legal_prefix(
                report.normalized_route,
                start_or_node,
                base_depth=base_depth,
            )
            if prune_reason:
                self.validation_stats["pruned_routes"] += 1
                if prune_reason.startswith("cycle"):
                    self.validation_stats["cycle_rejected"] += 1
                if prune_reason.startswith("same_molecule"):
                    self.validation_stats["same_molecule_rejected"] += 1
                if prune_reason.startswith("memory_avoid_reactant_set"):
                    self.validation_stats["memory_gate_rejected"] += 1
                    self.validation_stats["memory_avoid_rejected"] += 1
                if prune_reason.startswith("max_depth"):
                    self.validation_stats["depth_limited_routes"] += 1
                if prune_reason.startswith("max_prefix_steps"):
                    self.validation_stats["prefix_step_limited"] += 1
                example = (
                    f"PRUNED route suffix: Product={start_smiles}; "
                    f"Reason={prune_reason}; Source={route_source}"
                )
                rejected_examples.append(example)
                self._remember(self.invalid_route_examples, [example], limit=80)

            if normalized_route is None or not normalized_route.steps:
                self.validation_stats["rejected"] += 1
                if route_source == "llm_pathway":
                    self.validation_stats["llm_pathway_rejected"] += 1
                elif route_source == "llm_free":
                    self.validation_stats["llm_free_rejected"] += 1
                continue

            original_step_count = len(route.steps)
            accepted_step_count = len(normalized_route.steps)
            is_prefix_accept = (
                not report.steps_valid
                or bool(prune_reason)
                or accepted_step_count < original_step_count
            )
            if is_prefix_accept:
                self.validation_stats["accepted_prefix_routes"] += 1
                self.validation_stats["discarded_suffix_steps"] += max(
                    0,
                    original_step_count - accepted_step_count,
                )

            self.validation_stats["accepted"] += 1
            if route_source == "llm_pathway":
                self.validation_stats["llm_pathway_accepted"] += 1
                if is_prefix_accept:
                    self.validation_stats["llm_pathway_prefix_accepted"] += 1
            elif route_source == "llm_free":
                self.validation_stats["llm_free_accepted"] += 1
            for step_report in report.valid_steps[:accepted_step_count]:
                if step_report.match_source == "exact":
                    self.validation_stats["template_exact"] += 1
                elif step_report.match_source == "substructure":
                    self.validation_stats["template_substructure"] += 1
                elif step_report.match_source.startswith("top"):
                    self.validation_stats["template_topk"] += 1
            if report.is_successful and not is_prefix_accept:
                self.validation_stats["successful"] += 1

            self._remember(
                self.valid_route_examples,
                [item.as_experience() for item in report.valid_steps[:accepted_step_count]],
                limit=80,
            )
            normalized_route.generation_source = route_source
            normalized_route.candidate_id = getattr(route, "candidate_id", "")
            node = self._map_pathway_to_tree(
                normalized_route,
                start_or_node,
                base_depth=base_depth,
            )
            if node:
                accepted_nodes.append(node)
                for mapped_node in self._last_mapped_and_nodes or [node]:
                    if mapped_node not in self.all_created_and_nodes:
                        self.all_created_and_nodes.append(mapped_node)

        return accepted_nodes, rejected_examples

    @staticmethod
    def _repair_root_initialization_route(
        route: SynthesisRoute,
        start_smiles: str,
    ) -> Optional[SynthesisRoute]:
        if route is None or not route.steps:
            return route

        start = canonicalize_smiles(start_smiles)
        if not start:
            return route

        first_step = route.steps[0]
        if first_step.product:
            return route

        first_step.product = [start]
        if not first_step.molecule_set:
            first_step.molecule_set = [start]
        return route

    def _memory_gate_route(
        self,
        route: SynthesisRoute,
        start_or_node: ORNode,
    ) -> str:
        if not self.memory_hard_gate_enabled or route is None or not route.steps:
            return ""

        active_targets: Dict[str, Optional[ORNode]] = {start_or_node.smiles: start_or_node}
        expanded_products = set()

        for step in route.steps:
            raw_product = step.product[0] if step.product else ""
            product = canonicalize_smiles(raw_product)
            if not product or product not in active_targets:
                return ""

            current_product = active_targets.pop(product)
            reactants = [canonicalize_smiles(item) for item in step.reactants]
            reactants = [item for item in reactants if item]
            reactant_set = sorted(reactants)

            if product in reactant_set:
                return f"memory_self_reactant:{product}"

            if self._is_memory_avoid_reactant_set(product, reactant_set):
                return f"memory_avoid_reactant_set:{product}"

            ancestor_smiles = set(expanded_products)
            ancestor_smiles.add(product)
            if current_product is not None:
                ancestor_smiles.update(self._ancestor_smiles(current_product))
            for reactant in reactant_set:
                if reactant in ancestor_smiles:
                    return f"memory_ancestor_reactant:{reactant}"

            expanded_products.add(product)
            for reactant in reactant_set:
                if not self._is_purchasable(reactant):
                    active_targets[reactant] = self.node_registry.get(reactant)

        return ""

    def _memory_gate_experience(
        self,
        route: SynthesisRoute,
        reason: str,
        source: str,
    ) -> str:
        product = "UNKNOWN"
        reactants = "UNKNOWN"
        if route and route.steps:
            step = route.steps[0]
            product = canonicalize_smiles(step.product[0]) if step.product else "UNKNOWN"
            clean_reactants = [canonicalize_smiles(item) for item in step.reactants]
            clean_reactants = [item for item in clean_reactants if item]
            reactants = ".".join(sorted(clean_reactants)) if clean_reactants else "UNKNOWN"
        return (
            f"MEMORY-GATE rejected: Product={product} | Reactants={reactants} | "
            f"Reason={reason} | Source={source}"
        )

    @staticmethod
    def _tag_routes(routes: List[SynthesisRoute], source: str) -> None:
        for route in routes:
            if not getattr(route, "generation_source", ""):
                route.generation_source = source

    @staticmethod
    def _remember(target: List[str], examples: List[str], limit: int) -> None:
        for example in examples:
            if example and example not in target:
                target.append(example)
        if len(target) > limit:
            del target[: len(target) - limit]

    def _route_prune_reason(
        self,
        route: SynthesisRoute,
        start_or_node: ORNode,
    ) -> str:
        """Apply route-level hard constraints after template validation."""
        active_targets: Dict[str, Optional[ORNode]] = {start_or_node.smiles: start_or_node}
        expanded_products = set()

        for step in route.steps:
            raw_product = step.product[0] if step.product else ""
            product = canonicalize_smiles(raw_product)
            if not product or product not in active_targets:
                return ""

            current_product = active_targets.pop(product)
            reactants = [canonicalize_smiles(item) for item in step.reactants]
            reactants = [item for item in reactants if item]

            if self.prune_same_molecule_reactants and product in reactants:
                return f"same_molecule_reactant:{product}"

            if self.prune_cycles:
                ancestor_smiles = set(expanded_products)
                ancestor_smiles.add(product)
                if current_product is not None:
                    ancestor_smiles.update(self._ancestor_smiles(current_product))

                for reactant in reactants:
                    if reactant in ancestor_smiles:
                        return f"cycle_to_ancestor:{reactant}"
                    existing = self.node_registry.get(reactant)
                    if (
                        existing is not None
                        and current_product is not None
                        and self._is_ancestor_or_node(current_product, existing)
                    ):
                        return f"cycle_to_existing_ancestor:{reactant}"

            expanded_products.add(product)
            for reactant in reactants:
                if not self._is_purchasable(reactant):
                    active_targets[reactant] = self.node_registry.get(reactant)

        return ""

    def _prune_route_to_legal_prefix(
        self,
        route: SynthesisRoute,
        start_or_node: ORNode,
        base_depth: int = 0,
    ) -> tuple[Optional[SynthesisRoute], str]:
        """Keep the longest connected prefix that passes route-level pruning."""
        if route is None or not route.steps:
            return None, ""

        active_targets: Dict[str, Optional[ORNode]] = {start_or_node.smiles: start_or_node}
        expanded_products = set()
        legal_steps: List[ReactionStep] = []
        prune_reason = ""

        for step in route.steps:
            if (
                self.max_accepted_prefix_steps
                and len(legal_steps) >= self.max_accepted_prefix_steps
            ):
                prune_reason = f"max_prefix_steps:{self.max_accepted_prefix_steps}"
                break
            if base_depth + len(legal_steps) >= self.max_depth:
                prune_reason = f"max_depth_prefix_limit:{self.max_depth}"
                break

            raw_product = step.product[0] if step.product else ""
            product = canonicalize_smiles(raw_product)
            if not product or product not in active_targets:
                break

            current_product = active_targets.pop(product)
            reactants = [canonicalize_smiles(item) for item in step.reactants]
            reactants = [item for item in reactants if item]
            reactant_set = sorted(reactants)

            if self.prune_same_molecule_reactants and product in reactants:
                prune_reason = f"same_molecule_reactant:{product}"
                break

            if self.memory_hard_gate_enabled and self._is_memory_avoid_reactant_set(
                product,
                reactant_set,
            ):
                prune_reason = f"memory_avoid_reactant_set:{product}"
                break

            if self.prune_cycles:
                ancestor_smiles = set(expanded_products)
                ancestor_smiles.add(product)
                if current_product is not None:
                    ancestor_smiles.update(self._ancestor_smiles(current_product))

                for reactant in reactants:
                    if reactant in ancestor_smiles:
                        prune_reason = f"cycle_to_ancestor:{reactant}"
                        break
                    existing = self.node_registry.get(reactant)
                    if (
                        existing is not None
                        and current_product is not None
                        and self._is_ancestor_or_node(current_product, existing)
                    ):
                        prune_reason = f"cycle_to_existing_ancestor:{reactant}"
                        break
                if prune_reason:
                    break

            legal_steps.append(step)
            expanded_products.add(product)
            for reactant in reactants:
                if not self._is_purchasable(reactant):
                    active_targets[reactant] = self.node_registry.get(reactant)

        if len(legal_steps) == len(route.steps):
            return route, ""

        if not legal_steps:
            return None, prune_reason or "route_pruned_before_first_step"

        prefix = SynthesisRoute(
            steps=legal_steps,
            explanation=(
                route.explanation
                + (
                    f"\nAccepted longest legal prefix before pruning: {prune_reason}."
                    if prune_reason
                    else "\nAccepted longest legal prefix."
                )
            ),
            reward=route.reward,
            is_successful=False,
        )
        return prefix, prune_reason or "suffix_not_connected_or_pruned"

    def _is_memory_avoid_reactant_set(
        self,
        product: str,
        reactant_set: List[str],
    ) -> bool:
        if not self.memory_hard_gate_enabled or not reactant_set:
            return False
        micro = self.experience_memory.micro_snapshot(product, compact=True)
        avoid_sets = []
        for avoid in micro.get("avoid_reactant_sets", []):
            clean = [canonicalize_smiles(item) for item in avoid if item]
            clean = [item for item in clean if item]
            if clean:
                avoid_sets.append(sorted(clean))
        return any(reactant_set == avoid for avoid in avoid_sets)

    def _ancestor_smiles(self, node: ORNode) -> set:
        ancestors = set()
        queue = [node]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for parent_and in getattr(current, "parents", []):
                parent_or = parent_and.product
                if parent_or in visited:
                    continue
                ancestors.add(parent_or.smiles)
                queue.append(parent_or)
        return ancestors

    def _is_ancestor_or_node(self, node: ORNode, candidate_ancestor: ORNode) -> bool:
        if node is candidate_ancestor:
            return True
        queue = [node]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for parent_and in getattr(current, "parents", []):
                parent_or = parent_and.product
                if parent_or is candidate_ancestor:
                    return True
                queue.append(parent_or)
        return False

    def _mark_dead_end(self, node: ORNode, reason: str) -> None:
        if node.is_purchasable or node.is_solved:
            return
        if not getattr(node, "is_dead_end", False):
            node.is_dead_end = True
            node.dead_end_reason = reason
            self.validation_stats["dead_end_marked"] += 1
            if reason == "llm_no_valid_prefix":
                self.validation_stats["dead_end_expansion_failures"] += 1
        node.value = 0.0

    def _is_purchasable(self, smiles: str) -> bool:
        if hasattr(db_instance, "is_purchasable"):
            return bool(db_instance.is_purchasable(smiles))
        if hasattr(db_instance, "purchasable_db"):
            return smiles in db_instance.purchasable_db
        return False

    def _get_path_context(self, node: ANDNode) -> str:
        path_steps = []
        curr = node
        visited = set()

        while curr:
            if curr in visited:
                break
            visited.add(curr)
            reactants = ".".join(child.smiles for child in curr.children)
            path_steps.insert(
                0,
                f"Product={curr.product.smiles}; Reaction={curr.reaction_smarts}; Reactants={reactants}",
            )
            if curr.product.parents:
                curr = max(curr.product.parents, key=lambda parent: parent.value)
            else:
                break

        return " | ".join(path_steps)

    def _expand(self, and_node: ANDNode) -> None:
        unsolved_reactants = [
            child
            for child in and_node.children
            if not child.is_solved and not getattr(child, "is_dead_end", False)
        ]
        if not unsolved_reactants:
            self._record_failed_expansion(and_node)
            return

        target_or = self._choose_unsolved_reactant(unsolved_reactants)
        current_depth = and_node.depth + 1

        if getattr(target_or, "is_dead_end", False):
            self._record_failed_expansion(and_node)
            return

        if self._is_node_backed_off(target_or):
            self.validation_stats["backoff_skips"] += 1
            target_or.visit_count += 1
            remaining = max(
                0,
                getattr(target_or, "backoff_until_iteration", 0) - self.actual_iterations,
            )
            print(
                f"[Search][INFO] Backoff skip: molecule={target_or.smiles[:30]}, "
                f"remaining_iterations={remaining}"
            )
            self._record_failed_expansion(and_node)
            return

        if current_depth >= self.max_depth:
            self._mark_dead_end(target_or, "max_depth")
            self._record_failed_expansion(and_node)
            return

        if not target_or.smiles or not target_or.smiles.strip():
            print("[Search][WARN] Empty intermediate blocked before expansion.")
            target_or.visit_count += 1
            self._record_failed_expansion(and_node)
            return

        print(f"[Search][INFO] Expanding molecule={target_or.smiles[:30]}, depth={current_depth}")

        path_context = self._get_path_context(and_node)
        combined_experience = self._compose_experience(target_or)

        batch_new_nodes: List[ANDNode] = []
        rejected_examples: List[str] = []

        if self.llm_pathway_generation:
            new_routes = generate_pathways(
                target_or.smiles,
                n_routes=self.llm_pathway_width,
                experience=combined_experience,
                context=path_context,
                frontier_context=self._format_current_frontier_context(target_or),
                candidate_graph_context=self._candidate_graph_context_for_generation(
                    target_or.smiles
                ),
                enable_rag=self.llm_pathway_use_rag,
            )
            self._tag_routes(new_routes, "llm_pathway")
            batch_new_nodes, rejected_examples = self._ingest_routes(
                routes=new_routes,
                start_or_node=target_or,
                start_smiles=target_or.smiles,
                base_depth=current_depth,
            )
            if batch_new_nodes:
                print(
                    f"[Search][INFO] Accepted expansion branches: count={len(batch_new_nodes)}"
                )

        if batch_new_nodes:
            for node in batch_new_nodes:
                real_reward = self._compute_reward(node)
                if node.visit_count == 0:
                    node.value = real_reward

        if not batch_new_nodes:
            target_or.expansion_failures = getattr(target_or, "expansion_failures", 0) + 1
            target_or.visit_count += 1
            self.validation_stats["llm_no_valid_prefix_failures"] += 1
            remaining = max(
                0,
                self.dead_end_max_expansion_failures - target_or.expansion_failures,
            )
            rejected_examples.append(
                f"LLM-NO-VALID-PREFIX attempt {target_or.expansion_failures}/"
                f"{self.dead_end_max_expansion_failures}: Product={target_or.smiles}; "
                f"Depth={current_depth}; llm_pathway_width={self.llm_pathway_width}; "
                f"remaining_attempts_before_dead_end={remaining}"
            )
            self._update_micro_experience(
                target_or,
                batch_new_nodes,
                rejected_examples=rejected_examples,
            )
            self._maybe_apply_backoff(target_or)
            if target_or.expansion_failures >= self.dead_end_max_expansion_failures:
                self._mark_dead_end(target_or, "llm_no_valid_prefix")
                print(
                    f"[Search][WARN] Marked dead-end: molecule={target_or.smiles[:30]}, "
                    f"llm_expansion_failures={target_or.expansion_failures}"
                )
            else:
                self.validation_stats["soft_expansion_failures"] += 1
                print(
                    f"[Search][WARN] No legal LLM prefix: molecule={target_or.smiles[:30]}, "
                    f"remaining_attempts={remaining}"
                )
            self._record_failed_expansion(and_node)
            return

        self._update_micro_experience(
            target_or,
            batch_new_nodes,
            rejected_examples=rejected_examples,
        )
        target_or.expansion_failures = 0
        target_or.backoff_until_iteration = 0
        target_or.is_dead_end = False
        target_or.dead_end_reason = ""
        for new_and in batch_new_nodes:
            reward = self._compute_reward(new_and)
            self._backpropagate(new_and, reward=reward)

    def _score_new_nodes(self, nodes: List[ANDNode]) -> None:
        for node in nodes:
            if node.visit_count == 0:
                node.value = self._compute_reward(node)

    def _compose_experience(self, target_or: ORNode) -> str:
        local_memory = ""
        if not self.disable_micro_experience:
            local_memory = self.experience_memory.render_prompt_context(
                target_smiles=target_or.smiles,
                search_stats=self.validation_stats,
                max_chars=self.experience_max_context_chars,
                include_macro=False,
            )
        macro_strategy = "" if self.disable_macro_experience else (self.macro_experience or "").strip()
        if not macro_strategy:
            return local_memory

        max_chars = self.experience_max_context_chars
        macro_header = "[Global target-level macro strategy]\n"
        local_header = "\n\n[Current frontier molecule memory]\n"
        overhead = len(macro_header) + len(local_header) + 80

        macro_budget = min(len(macro_strategy), max(1000, max_chars // 2))
        local_budget = max(800, max_chars - overhead - macro_budget)
        if len(local_memory) > local_budget:
            local_memory = (
                local_memory[: max(0, local_budget - 40)].rstrip()
                + "\n... [local memory truncated]"
            )

        combined = f"{macro_header}{macro_strategy[:macro_budget]}{local_header}{local_memory}"
        if len(combined) > max_chars:
            combined = combined[: max_chars - 40].rstrip() + "\n... [experience truncated]"
        return combined

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        if not text or max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def _map_pathway_to_tree(
        self,
        route: SynthesisRoute,
        start_or_node: ORNode,
        base_depth: int = 0,
    ) -> Optional[ANDNode]:
        self._last_mapped_and_nodes = []
        current_product = start_or_node
        created_and_nodes: List[ANDNode] = []
        first_and_node: Optional[ANDNode] = None
        active_targets = {canonicalize_smiles(current_product.smiles): current_product}
        route_source = getattr(route, "generation_source", "validated_unknown")
        candidate_id = getattr(route, "candidate_id", "")

        for offset, step in enumerate(route.steps):
            if base_depth + offset >= self.max_depth:
                return first_and_node

            raw_product = step.product[0] if step.product else None
            clean_product = canonicalize_smiles(raw_product)
            if not clean_product or clean_product not in active_targets:
                return first_and_node

            current_product = active_targets.pop(clean_product)
            reactant_nodes = []
            for reactant in step.reactants:
                clean_reactant = canonicalize_smiles(reactant)
                if not clean_reactant:
                    return first_and_node
                reactant_node = self._get_or_create_or_node(clean_reactant)
                reactant_nodes.append(reactant_node)
                if not reactant_node.is_solved:
                    active_targets[reactant_node.smiles] = reactant_node

            step_reactants = {node.smiles for node in reactant_nodes}
            existing_and = None
            for child_and in current_product.children:
                child_reactants = {child.smiles for child in child_and.children}
                if (
                    child_and.reaction_smarts == step.reaction
                    and child_reactants == step_reactants
                ):
                    existing_and = child_and
                    break

            if existing_and:
                and_node = existing_and
            else:
                and_node = ANDNode(
                    step.reaction,
                    current_product,
                    reactant_nodes,
                    depth=base_depth + offset,
                )
                current_product.children.append(and_node)
                for reactant_node in reactant_nodes:
                    reactant_node.parents.append(and_node)

            and_node.is_valid = True
            feedback = step.feedback
            if route_source and f"source={route_source}" not in feedback:
                feedback += f" | source={route_source}"
            if candidate_id and f"candidate_id={candidate_id}" not in feedback:
                feedback += f" | candidate_id={candidate_id}"
            and_node.validation_feedback = feedback
            if not getattr(and_node, "generation_source", ""):
                and_node.generation_source = route_source
            if candidate_id and not getattr(and_node, "candidate_id", ""):
                and_node.candidate_id = candidate_id
            created_and_nodes.append(and_node)
            if first_and_node is None:
                first_and_node = and_node

        for node in reversed(created_and_nodes):
            node.update_solved_status()

        self._last_mapped_and_nodes = created_and_nodes
        return first_and_node

    def _get_or_create_or_node(self, smiles: str) -> ORNode:
        smiles = canonicalize_smiles(smiles)
        if smiles not in self.node_registry:
            is_purchasable = self._is_purchasable(smiles)
            self.node_registry[smiles] = ORNode(smiles, is_purchasable)
        return self.node_registry[smiles]

    def _choose_unsolved_reactant(self, reactants: List[ORNode]) -> ORNode:
        """Choose the least-explored unsolved leaf in the current search tree."""
        viable = [
            node
            for node in reactants
            if not getattr(node, "is_dead_end", False) and not node.is_solved
        ]
        if not viable:
            return reactants[0]
        ready = [node for node in viable if not self._is_node_backed_off(node)]
        candidates = ready or viable

        return min(
            candidates,
            key=lambda node: (
                1 if self._is_node_backed_off(node) else 0,
                getattr(node, "backoff_until_iteration", 0),
                getattr(node, "visit_count", 0),
                getattr(node, "expansion_failures", 0),
                -self._heavy_atom_count(node.smiles),
                node.smiles,
            ),
        )

    def _format_current_frontier_context(self, active_node: Optional[ORNode] = None) -> str:
        metrics = compute_or_frontier_metrics(
            self.root,
            db=db_instance,
            sc_score_fn=get_sc_score,
            max_depth=self.max_depth,
        )
        text = format_frontier_context(metrics)
        if active_node is not None:
            text += (
                "\nactive_intermediate="
                f"{active_node.smiles}; heavy_atoms={self._heavy_atom_count(active_node.smiles)}; "
                f"visit_count={active_node.visit_count}; value={active_node.value:.3f}; "
                f"children={len(active_node.children)}"
            )
        return text

    def _frontier_brief(self, node: ANDNode) -> str:
        metrics = compute_and_frontier_metrics(
            node,
            db=db_instance,
            sc_score_fn=get_sc_score,
            max_depth=self.max_depth,
        )
        return (
            f"stock_ratio={metrics.stock_ratio:.2f}; "
            f"non_purch={metrics.non_purchasable_terminal_count}; "
            f"max_non_purch_heavy={metrics.max_non_purchasable_heavy_atoms}; "
            f"completion={metrics.frontier_completion_score:.2f}"
        )

    def frontier_snapshot(self) -> Dict:
        return compute_or_frontier_metrics(
            self.root,
            db=db_instance,
            sc_score_fn=get_sc_score,
            max_depth=self.max_depth,
        ).to_dict()

    def dead_end_snapshot(self) -> List[Dict]:
        items = []
        for node in self.node_registry.values():
            if getattr(node, "is_dead_end", False):
                items.append(
                    {
                        "smiles": node.smiles,
                        "reason": getattr(node, "dead_end_reason", ""),
                        "visit_count": node.visit_count,
                        "expansion_failures": getattr(node, "expansion_failures", 0),
                        "backoff_until_iteration": getattr(
                            node,
                            "backoff_until_iteration",
                            0,
                        ),
                        "backoff_count": getattr(node, "backoff_count", 0),
                        "children": len(node.children),
                    }
                )
        return sorted(
            items,
            key=lambda item: (item["visit_count"], item["expansion_failures"]),
            reverse=True,
        )

    @staticmethod
    def _heavy_atom_count(smiles: str) -> int:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles or "")
            return mol.GetNumHeavyAtoms() if mol is not None else 0
        except Exception:
            return 0

    def _is_node_backed_off(self, node: ORNode) -> bool:
        if not self.node_backoff_enabled:
            return False
        return self.actual_iterations < getattr(node, "backoff_until_iteration", 0)

    def _maybe_apply_backoff(self, node: ORNode) -> None:
        if not self.node_backoff_enabled:
            return
        threshold = 5
        failures = getattr(node, "expansion_failures", 0)
        if failures < threshold:
            return
        node.backoff_count = getattr(node, "backoff_count", 0) + 1
        node.backoff_until_iteration = self.actual_iterations + max(
            1,
            5,
        )
        self.validation_stats["backoff_applied"] += 1
        event = {
            "iteration": self.actual_iterations,
            "event_type": "node_backoff",
            "product": node.smiles,
            "expansion_failures": failures,
            "backoff_until_iteration": node.backoff_until_iteration,
            "backoff_count": node.backoff_count,
        }
        self.experience_memory.note_event(event)
        self.experience_trace.append(event)
        print(
            f"[Search][INFO] Backoff applied: molecule={node.smiles[:30]}, "
            f"until_iteration={node.backoff_until_iteration}, failures={failures}"
        )

    def _or_has_expandable_frontier(self, node: ORNode, visited=None) -> bool:
        if visited is None:
            visited = set()
        if node in visited:
            return False
        visited.add(node)
        if node.is_solved or node.is_purchasable:
            return False
        if self.prune_dead_end_branches and getattr(node, "is_dead_end", False):
            return False
        if not node.children:
            return True
        return any(
            child.is_valid
            and not child.is_solved
            and self._and_has_expandable_frontier(child, visited)
            for child in node.children
        )

    def _and_has_expandable_frontier(self, node: ANDNode, visited=None) -> bool:
        if visited is None:
            visited = set()
        if node.is_solved or not node.is_valid:
            return False
        return any(
            self._or_has_expandable_frontier(child, set(visited))
            for child in node.children
        )

    def _select(self) -> Optional[ANDNode]:
        current_or = self.root
        visited_ors = set()
        last_and = None

        while True:
            current_or.update_solved_status()
            if current_or.is_solved:
                return None

            if current_or in visited_ors:
                return last_and
            visited_ors.add(current_or)

            viable_and_children = [
                and_node
                for and_node in current_or.children
                if and_node.is_valid
                and not and_node.is_solved
                and self._and_has_expandable_frontier(and_node)
            ]
            if not viable_and_children:
                return None

            best_and = max(
                viable_and_children,
                key=lambda node: node.ucb_score(self.c_param),
            )
            last_and = best_and

            unsolved_or_children = [
                child
                for child in best_and.children
                if not child.is_solved
                and not (
                    self.prune_dead_end_branches
                    and getattr(child, "is_dead_end", False)
                )
            ]
            if not unsolved_or_children:
                return None

            is_frontier = any(len(or_child.children) == 0 for or_child in unsolved_or_children)
            if is_frontier or best_and.depth >= self.max_depth - 1:
                return best_and

            current_or = self._choose_unsolved_reactant(unsolved_or_children)

    def _compute_reward(self, node: ANDNode) -> float:
        if not node.is_valid or not node.children:
            return 0.0

        node.update_solved_status()
        num_purchasable = sum(1 for child in node.children if child.is_purchasable)
        f_avail = num_purchasable / len(node.children)

        try:
            max_sc = max(get_sc_score(child.smiles) for child in node.children)
            f_chem = 1.0 - (max_sc - 1.0) / 4.0
            f_chem = max(0.0, min(1.0, f_chem))
        except Exception as exc:
            print(f"[Scoring][WARN] Reward calculation failed: {exc}")
            f_chem = 0.5

        final_reward = self.alpha * f_avail + (1.0 - self.alpha) * f_chem
        final_reward = max(0.0, min(1.0, final_reward))
        node.reward = final_reward
        node.reward_components = {
            "mode": "tree_search",
            "availability": f_avail,
            "chemistry": f_chem,
            "alpha": self.alpha,
            "final_reward": final_reward,
        }
        return final_reward

    def _record_failed_expansion(self, node: ANDNode) -> None:
        """Record a failed expansion attempt without erasing the route's value.

        A failed LLM proposal batch is evidence for the active frontier molecule,
        not evidence that the already validated parent reaction has zero local
        value. The dead-end/frontier checks below still let ancestor values drop
        to zero when a branch truly becomes non-viable.
        """
        queue = [node]
        visited_and = set()
        visited_or = set()

        while queue:
            curr_and = queue.pop(0)
            if curr_and in visited_and:
                continue
            visited_and.add(curr_and)

            curr_and.visit_count += 1
            curr_and.update_solved_status()

            parent_or = curr_and.product
            if parent_or in visited_or:
                continue
            visited_or.add(parent_or)
            parent_or.visit_count += 1

            legal_children = [
                child
                for child in parent_or.children
                if child.is_valid
                and (
                    child.is_solved
                    or not self.prune_dead_end_branches
                    or self._and_has_expandable_frontier(child)
                )
            ]
            parent_or.value = max((child.value for child in legal_children), default=0.0)
            parent_or.update_solved_status()

            for parent_and in parent_or.parents:
                if parent_and not in visited_and:
                    queue.append(parent_and)

    def _backpropagate(self, node: ANDNode, reward: float, force_reward: bool = False) -> None:
        queue = [node]
        visited_and = set()
        visited_or = set()

        while queue:
            curr_and = queue.pop(0)
            if curr_and in visited_and:
                continue
            visited_and.add(curr_and)

            curr_and.visit_count += 1
            curr_and.value = curr_and.value + (reward - curr_and.value) / curr_and.visit_count
            curr_and.update_solved_status()

            parent_or = curr_and.product
            if parent_or not in visited_or:
                visited_or.add(parent_or)
                parent_or.visit_count += 1
                legal_children = [
                    child
                    for child in parent_or.children
                    if child.is_valid
                    and (
                        child.is_solved
                        or not self.prune_dead_end_branches
                        or self._and_has_expandable_frontier(child)
                    )
                ]
                if legal_children:
                    parent_or.value = max(child.value for child in legal_children)
                else:
                    parent_or.value = 0.0
                parent_or.update_solved_status()
                for parent_and in parent_or.parents:
                    if parent_and not in visited_and:
                        queue.append(parent_and)

    def extract_best_route(self) -> Optional[SynthesisRoute]:
        self.root.update_solved_status()
        if not self.root.is_solved:
            return SynthesisRoute(
                steps=self.extract_best_partial_route().steps,
                explanation="Target not solved by a fully validated route.",
                is_successful=False,
                reward=self.root.value,
            )

        all_steps: List[ReactionStep] = []
        visited_nodes = set()

        def collect_steps(or_node: ORNode) -> None:
            if or_node.is_purchasable or or_node in visited_nodes:
                return
            visited_nodes.add(or_node)

            solved_children = [
                child for child in or_node.children if child.is_valid and child.is_solved
            ]
            if not solved_children:
                return
            best_and = max(solved_children, key=lambda child: child.value)

            all_steps.append(
                ReactionStep(
                    molecule_set=[or_node.smiles],
                    rational=(
                        "Validated solution tree step"
                        + (
                            f" | source={best_and.generation_source}"
                            if getattr(best_and, "generation_source", "")
                            else ""
                        )
                        + (
                            f" | candidate_id={best_and.candidate_id}"
                            if getattr(best_and, "candidate_id", "")
                            else ""
                        )
                    ),
                    product=[or_node.smiles],
                    reaction=best_and.reaction_smarts,
                    reactants=[child.smiles for child in best_and.children],
                    updated_molecule_set=[child.smiles for child in best_and.children],
                    is_valid=True,
                    feedback=best_and.validation_feedback,
                )
            )

            for child_or in best_and.children:
                collect_steps(child_or)

        collect_steps(self.root)

        route = SynthesisRoute(
            steps=all_steps,
            explanation="Convergent synthesis tree with strict reaction validation.",
            is_successful=True,
            reward=self.root.value,
        )
        report = self._validate_generated_route(route, self.target_smiles)
        route.validation_report = report
        if report.normalized_route is not None:
            route.steps = report.normalized_route.steps
        route.is_successful = report.is_successful
        return route

    def extract_best_partial_route(self) -> SynthesisRoute:
        all_steps: List[ReactionStep] = []
        visited_nodes = set()

        def collect(or_node: ORNode) -> None:
            if or_node.is_purchasable or or_node in visited_nodes:
                return
            visited_nodes.add(or_node)
            legal_children = [
                child
                for child in or_node.children
                if child.is_valid
                and (
                    child.is_solved
                    or not self.prune_dead_end_branches
                    or self._and_has_expandable_frontier(child)
                )
            ]
            if not legal_children:
                legal_children = [child for child in or_node.children if child.is_valid]
            if not legal_children:
                return
            best_and = max(legal_children, key=lambda child: child.value)
            all_steps.append(
                ReactionStep(
                    molecule_set=[or_node.smiles],
                    rational=(
                        "Best partial validated tree step"
                        + (
                            f" | source={best_and.generation_source}"
                            if getattr(best_and, "generation_source", "")
                            else ""
                        )
                        + (
                            f" | candidate_id={best_and.candidate_id}"
                            if getattr(best_and, "candidate_id", "")
                            else ""
                        )
                    ),
                    product=[or_node.smiles],
                    reaction=best_and.reaction_smarts,
                    reactants=[child.smiles for child in best_and.children],
                    updated_molecule_set=[child.smiles for child in best_and.children],
                    is_valid=True,
                    feedback=best_and.validation_feedback,
                )
            )
            for child in best_and.children:
                collect(child)

        collect(self.root)
        return SynthesisRoute(
            steps=all_steps,
            explanation="Best partial route from the validated search tree.",
            is_successful=False,
            reward=self.root.value,
        )

    def _save_logs(self, start_stats, end_stats) -> None:
        delta_calls = end_stats["calls"] - start_stats["calls"]
        delta_tokens = end_stats["total_tokens"] - start_stats["total_tokens"]

        self.last_log_record = {
            "molecule": self.target_smiles,
            "method": "strict_experience_guided_search",
            "calls": delta_calls,
            "total_tokens": delta_tokens,
            "is_success": self.root.is_solved,
            "root_value": self.root.value,
            "validation_stats": dict(self.validation_stats),
            "frontier_metrics": self.frontier_snapshot(),
            "dead_end_nodes": self.dead_end_snapshot(),
            "initial_macro_experience": self.initial_macro_experience,
            "initial_macro_experience_chars": len(self.initial_macro_experience or ""),
            "structured_experience": self.experience_memory.snapshot_all(
                self.validation_stats
            ),
            "experience_trace": list(self.experience_trace[-40:]),
            "reward_policy": {
                "mode": "tree_search",
                "availability_weight": self.alpha,
                "chemistry_weight": 1.0 - self.alpha,
                "max_search_depth": self.max_depth,
            },
            "generation_policy": {
                "llm_pathway_generation": self.llm_pathway_generation,
                "initial_macro_generation": True,
                "llm_pathway_width": self.llm_pathway_width,
                "llm_pathway_use_rag": self.llm_pathway_use_rag,
                "candidate_graph_enabled": self.candidate_graph_enabled,
                "candidate_graph_depth": self.candidate_graph_depth,
                "candidate_graph_width": self.candidate_graph_width,
                "candidate_graph_max_nodes": self.candidate_graph_max_nodes,
                "candidate_graph_retrieval_size": self.candidate_graph_retrieval_size,
                "candidate_graph_template_top_k": self.candidate_graph_template_top_k,
                "candidate_graph_direct_max_templates": (
                    self.candidate_graph_direct_max_templates
                ),
                "candidate_graph_applicable_template_limit": (
                    self.candidate_graph_applicable_template_limit
                ),
                "candidate_graph_max_outcomes": self.candidate_graph_max_outcomes,
                "candidate_graph_prune_nonprogressive": (
                    self.candidate_graph_prune_nonprogressive
                ),
                "candidate_graph_max_lateral_risk": self.candidate_graph_max_lateral_risk,
                "max_accepted_prefix_steps": self.max_accepted_prefix_steps,
                "root_llm_retry_attempts": self.root_llm_retry_attempts,
                "dead_end_max_expansion_failures": (
                    self.dead_end_max_expansion_failures
                ),
                "accept_longest_valid_prefix": True,
            },
            "ablation_policy": {
                "disable_macro_experience": self.disable_macro_experience,
                "disable_micro_experience": self.disable_micro_experience,
                "disable_candidate_cuts": self.disable_candidate_cuts,
                "disable_initial_macro_debate": self.disable_initial_macro_debate,
            },
        }

    def _initialize_macro_experience(self, candidate_graph_context: str) -> str:
        if self.disable_macro_experience:
            print("[Experience][INFO] Macro experience disabled by ablation policy.")
            return ""
        print("[Experience][INFO] Initializing macro strategy from target-level evidence.")
        exp = generate_initial_macro_experience(
            target_molecule=self.target_smiles,
            candidate_graph_context=candidate_graph_context,
            enable_rag=self.llm_pathway_use_rag,
            use_debate=not self.disable_initial_macro_debate,
        )
        self.experience_trace.append(
            {
                "event": "initial_macro",
                "target": self.target_smiles,
                "macro_experience": exp,
                "macro_experience_chars": len(exp),
            }
        )
        print("[Experience][INFO] Initial macro strategy ready.")
        return exp

    def _format_initial_macro_context(self) -> str:
        if self.disable_macro_experience:
            return ""
        if not self.initial_macro_experience.strip():
            return ""
        return (
            "[Initial target-level macro experience]\n"
            + self.initial_macro_experience.strip()
        )

    def _combine_macro_experience(self, validated_experience: str) -> str:
        initial_macro = self._format_initial_macro_context()
        if not initial_macro:
            return validated_experience
        combined = (
            f"{initial_macro}\n\n"
            "[Validated search memory]\n"
            f"{validated_experience.strip() or 'No validated search memory yet.'}"
        )
        if len(combined) > self.experience_max_context_chars:
            keep_initial = initial_macro[: min(len(initial_macro), 1800)]
            remaining = max(0, self.experience_max_context_chars - len(keep_initial) - 80)
            combined = (
                f"{keep_initial}\n\n[Validated search memory]\n"
                f"{validated_experience.strip()[:remaining]}\n... [macro context truncated]"
            )
        return combined

    def _update_macro_from_root_validation(
        self,
        initial_nodes: List[ANDNode],
        bad_examples: Optional[List[str]] = None,
    ) -> str:
        if self.disable_macro_experience:
            return ""
        print("[Experience][INFO] Updating macro strategy from validated root examples.")
        self._update_structured_experience(
            target_or=self.root,
            batch_nodes=initial_nodes,
            rejected_examples=bad_examples or [],
            event_type="root_validated_macro_update",
        )
        exp = self.experience_memory.render_prompt_context(
            target_smiles=self.target_smiles,
            search_stats=self.validation_stats,
            max_chars=self.experience_max_context_chars,
        )
        exp = self._combine_macro_experience(exp)
        print("[Experience][INFO] Macro strategy updated from validated root examples.")
        return exp

    def _update_macro_experience_periodic(self) -> str:
        if self.disable_macro_experience:
            return ""
        if not self.all_created_and_nodes and not self.invalid_route_examples:
            return self.macro_experience
        exp = self.experience_memory.render_prompt_context(
            target_smiles=self.target_smiles,
            search_stats=self.validation_stats,
            max_chars=self.experience_max_context_chars,
        )
        exp = self._combine_macro_experience(exp)
        print("[Experience][INFO] Structured macro memory updated.")
        return exp

    def _update_micro_experience(
        self,
        target_or: ORNode,
        batch_nodes: List[ANDNode],
        rejected_examples: Optional[List[str]] = None,
    ) -> None:
        if self.disable_micro_experience:
            return
        sorted_batch = sorted(batch_nodes, key=lambda node: node.value, reverse=True)
        bad_prompts = list(rejected_examples or [])
        if not sorted_batch and not bad_prompts:
            return

        print(f"[Experience][INFO] Updating structured micro memory for molecule={target_or.smiles[:30]}")
        self._update_structured_experience(
            target_or=target_or,
            batch_nodes=sorted_batch,
            rejected_examples=bad_prompts,
            event_type="micro_update",
        )
        print("[Experience][INFO] Structured micro memory updated.")

    def _update_structured_experience(
        self,
        target_or: ORNode,
        batch_nodes: List[ANDNode],
        rejected_examples: Optional[List[str]] = None,
        event_type: str = "micro_update",
    ) -> None:
        accepted = 0
        for node in batch_nodes or []:
            reactants = [child.smiles for child in node.children]
            source = getattr(node, "generation_source", "") or "unknown"
            match_source = "unknown"
            similarity = 0.0
            feedback = getattr(node, "validation_feedback", "") or ""
            if "template_backed_match:" in feedback:
                match_source = feedback.split("template_backed_match:", 1)[1].split(":", 1)[0].split("|", 1)[0]
            if "similarity=" in feedback:
                try:
                    similarity = float(feedback.split("similarity=", 1)[1].split("|", 1)[0])
                except ValueError:
                    similarity = 0.0
            self.experience_memory.record_valid_cut(
                product=node.product.smiles,
                reactants=reactants,
                reaction=node.reaction_smarts,
                reward=node.value,
                match_source=match_source,
                similarity=similarity,
                source=source,
            )
            self.experience_memory.update_node_stats(
                node.product.smiles,
                {
                    "visit_count": node.product.visit_count,
                    "children_count": len(node.product.children),
                    "is_solved": node.product.is_solved,
                    "is_purchasable": node.product.is_purchasable,
                    "best_value": node.product.value,
                    "backoff_until_iteration": getattr(
                        node.product,
                        "backoff_until_iteration",
                        0,
                    ),
                    "backoff_count": getattr(node.product, "backoff_count", 0),
                },
            )
            accepted += 1

        self.experience_memory.record_observation_texts(rejected_examples or [])
        self.experience_memory.update_node_stats(
            target_or.smiles,
            {
                "visit_count": target_or.visit_count,
                "children_count": len(target_or.children),
                "is_solved": target_or.is_solved,
                "is_purchasable": target_or.is_purchasable,
                "best_value": target_or.value,
                "expansion_failures": getattr(target_or, "expansion_failures", 0),
                "backoff_until_iteration": getattr(
                    target_or,
                    "backoff_until_iteration",
                    0,
                ),
                "backoff_count": getattr(target_or, "backoff_count", 0),
            },
        )
        target_or.structured_experience = self.experience_memory.micro_snapshot(target_or.smiles)
        target_or.micro_experience = json.dumps(
            target_or.structured_experience,
            ensure_ascii=False,
            indent=2,
        )
        event = {
            "iteration": self.actual_iterations,
            "event_type": event_type,
            "product": target_or.smiles,
            "accepted_cuts": accepted,
            "rejected_observations": len(rejected_examples or []),
            "micro_counts": {
                "viable_cuts": len(target_or.structured_experience.get("viable_cuts", [])),
                "failed_cuts": len(target_or.structured_experience.get("failed_cuts", [])),
                "avoid_reactant_sets": len(
                    target_or.structured_experience.get("avoid_reactant_sets", [])
                ),
                "local_taboos": len(target_or.structured_experience.get("local_taboos", [])),
            },
            "next_generation_constraints": target_or.structured_experience.get(
                "next_generation_constraints", []
            )[:5],
        }
        self.experience_memory.note_event(event)
        self.experience_trace.append(event)
