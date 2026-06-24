from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from chem_utils.reaction import apply_retrosynthetic_template
from chem_utils.reaction import canonicalize_smiles
from planner.frontier import compute_candidate_frontier_metrics
from planner.template_matcher import TemplateMatcher, expand_product_with_templates


@dataclass
class TemplateCandidate:
    candidate_id: str
    product_smiles: str
    template: str
    reactants: List[str]
    rank: int
    checked_templates: int
    source: str
    heuristic_score: float
    purchasable_count: int
    all_purchasable: bool
    max_reactant_heavy_atoms: int
    product_heavy_atoms: int
    non_purchasable_count: int
    max_non_purchasable_heavy_atoms: int
    total_non_purchasable_heavy_atoms: int
    heavy_atom_delta: int
    complexity_delta_score: float
    estimated_completion_score: float
    lateral_move_risk: float
    nonstock_charged_count: int = 0
    nonstock_radical_count: int = 0
    nonstock_unstable_count: int = 0
    rag_similarity: float = 0.0
    experience_score: float = 0.0
    final_score: float = 0.0
    evidence: str = ""


@dataclass
class CandidateGraphNode:
    product_smiles: str
    depth: int
    candidates: List[TemplateCandidate]


@dataclass
class CandidateGraph:
    root_smiles: str
    nodes: List[CandidateGraphNode]
    rag_route_count: int
    template_query_count: int
    applied_template_count: int


def build_template_candidates(
    db,
    product_smiles: str,
    pool_size: int,
    max_templates: int,
    max_outcomes_per_template: int = 25,
    sc_score_fn: Optional[Callable[[str], float]] = None,
    prune_nonprogressive: bool = True,
    max_lateral_move_risk: float = 0.78,
    strict_unstable_filter: bool = True,
) -> List[TemplateCandidate]:
    """Generate a ranked, database-template-backed candidate action set."""
    product = canonicalize_smiles(product_smiles)
    if not product:
        return []

    raw_pool_size = pool_size * 3 if prune_nonprogressive else pool_size
    expansions = expand_product_with_templates(
        db=db,
        product_smiles=product,
        max_routes=raw_pool_size,
        max_templates=max_templates,
        max_outcomes_per_template=max_outcomes_per_template,
    )

    product_heavy = _heavy_atom_count(product)
    purchasable = getattr(db, "is_purchasable", lambda _: False)
    candidates: List[TemplateCandidate] = []

    for index, expansion in enumerate(expansions, start=1):
        reactants = [canonicalize_smiles(item) for item in expansion.reactants]
        if not reactants or any(not item for item in reactants):
            continue
        reactant_heavies = [_heavy_atom_count(item) for item in reactants]
        purchasable_count = sum(1 for item in reactants if purchasable(item))
        frontier = compute_candidate_frontier_metrics(
            product_smiles=product,
            reactants=reactants,
            db=db,
            sc_score_fn=sc_score_fn,
        )
        nonstock_charged_count = _nonstock_charged_count(reactants, purchasable)
        nonstock_radical_count = _nonstock_radical_count(reactants, purchasable)
        nonstock_unstable_count = _nonstock_unstable_count(reactants, purchasable)
        if prune_nonprogressive and _is_nonprogressive_candidate(
            frontier=frontier,
            purchasable_count=purchasable_count,
            nonstock_charged_count=nonstock_charged_count,
            nonstock_radical_count=nonstock_radical_count,
            nonstock_unstable_count=nonstock_unstable_count,
            max_lateral_move_risk=max_lateral_move_risk,
            strict_unstable_filter=strict_unstable_filter,
        ):
            continue
        candidates.append(
            TemplateCandidate(
                candidate_id=f"cand_{index:03d}",
                product_smiles=product,
                template=expansion.template,
                reactants=reactants,
                rank=index,
                checked_templates=expansion.checked_templates,
                source=expansion.source,
                heuristic_score=_candidate_score(
                    product=product,
                    product_heavy=product_heavy,
                    reactants=reactants,
                    reactant_heavies=reactant_heavies,
                    purchasable_count=purchasable_count,
                    frontier=frontier,
                    nonstock_charged_count=nonstock_charged_count,
                    nonstock_radical_count=nonstock_radical_count,
                    nonstock_unstable_count=nonstock_unstable_count,
                ),
                purchasable_count=purchasable_count,
                all_purchasable=purchasable_count == len(reactants),
                max_reactant_heavy_atoms=max(reactant_heavies) if reactant_heavies else 0,
                product_heavy_atoms=product_heavy,
                non_purchasable_count=frontier.non_purchasable_count,
                max_non_purchasable_heavy_atoms=frontier.max_non_purchasable_heavy_atoms,
                total_non_purchasable_heavy_atoms=frontier.total_non_purchasable_heavy_atoms,
                heavy_atom_delta=frontier.heavy_atom_delta,
                complexity_delta_score=frontier.complexity_delta_score,
                estimated_completion_score=frontier.estimated_completion_score,
                lateral_move_risk=frontier.lateral_move_risk,
                nonstock_charged_count=nonstock_charged_count,
                nonstock_radical_count=nonstock_radical_count,
                nonstock_unstable_count=nonstock_unstable_count,
                final_score=_candidate_score(
                    product=product,
                    product_heavy=product_heavy,
                    reactants=reactants,
                    reactant_heavies=reactant_heavies,
                    purchasable_count=purchasable_count,
                    frontier=frontier,
                    nonstock_charged_count=nonstock_charged_count,
                    nonstock_radical_count=nonstock_radical_count,
                    nonstock_unstable_count=nonstock_unstable_count,
                ),
            )
        )

    candidates.sort(key=lambda item: (item.heuristic_score, -item.rank), reverse=True)
    candidates = candidates[:pool_size]
    for new_rank, candidate in enumerate(candidates, start=1):
        candidate.rank = new_rank
        candidate.candidate_id = f"cand_{new_rank:03d}"
    return candidates


def build_experience_guided_candidate_graph(
    db,
    product_smiles: str,
    graph_depth: int,
    per_node_width: int,
    retrieval_size: int,
    template_top_k: int,
    direct_max_templates: int = 0,
    applicable_template_limit: int = 2000,
    max_outcomes_per_template: int = 25,
    max_nodes: int = 8,
    sc_score_fn: Optional[Callable[[str], float]] = None,
    experience_memory=None,
    prune_nonprogressive: bool = True,
    max_lateral_move_risk: float = 0.88,
    strict_unstable_filter: bool = True,
) -> CandidateGraph:
    """Build a compact template-backed action graph for LLM context.

    The graph is evidence only: each edge is created by applying product-side
    applicable database templates, then ranking them with RAG, experience, and
    route-completion heuristics. It is not inserted into the search tree unless
    the LLM later proposes a route that passes the normal strict validator.
    """
    root = canonicalize_smiles(product_smiles)
    if not root:
        return CandidateGraph("", [], 0, 0, 0)

    graph_depth = max(0, int(graph_depth))
    per_node_width = max(1, int(per_node_width))
    max_nodes = max(1, int(max_nodes))

    queue: List[Tuple[str, int]] = [(root, 0)]
    visited = set()
    nodes: List[CandidateGraphNode] = []
    rag_route_count = 0
    template_query_count = 0
    applied_template_count = 0

    while queue and len(nodes) < max_nodes:
        product, depth = queue.pop(0)
        product = canonicalize_smiles(product)
        if not product or product in visited:
            continue
        visited.add(product)

        candidates, stats = build_retrieved_template_candidates(
            db=db,
            product_smiles=product,
            pool_size=per_node_width,
            retrieval_size=retrieval_size,
            template_top_k=template_top_k,
            direct_max_templates=direct_max_templates,
            applicable_template_limit=applicable_template_limit,
            max_outcomes_per_template=max_outcomes_per_template,
            sc_score_fn=sc_score_fn,
            experience_memory=experience_memory,
            prune_nonprogressive=prune_nonprogressive,
            max_lateral_move_risk=max_lateral_move_risk,
            strict_unstable_filter=strict_unstable_filter,
        )
        rag_route_count += stats.get("rag_route_count", 0)
        template_query_count += stats.get("template_query_count", 0)
        applied_template_count += stats.get("applied_template_count", 0)
        nodes.append(CandidateGraphNode(product, depth, candidates))

        if depth >= graph_depth:
            continue
        for candidate in candidates[:per_node_width]:
            for reactant in candidate.reactants:
                clean_reactant = canonicalize_smiles(reactant)
                if not clean_reactant or clean_reactant in visited:
                    continue
                if getattr(db, "is_purchasable", lambda _: False)(clean_reactant):
                    continue
                if len(nodes) + len(queue) >= max_nodes:
                    break
                queue.append((clean_reactant, depth + 1))

    return CandidateGraph(
        root_smiles=root,
        nodes=nodes,
        rag_route_count=rag_route_count,
        template_query_count=template_query_count,
        applied_template_count=applied_template_count,
    )


def build_retrieved_template_candidates(
    db,
    product_smiles: str,
    pool_size: int,
    retrieval_size: int,
    template_top_k: int,
    direct_max_templates: int = 0,
    applicable_template_limit: int = 2000,
    max_outcomes_per_template: int = 25,
    sc_score_fn: Optional[Callable[[str], float]] = None,
    experience_memory=None,
    prune_nonprogressive: bool = True,
    max_lateral_move_risk: float = 0.88,
    strict_unstable_filter: bool = True,
) -> Tuple[List[TemplateCandidate], Dict[str, int]]:
    product = canonicalize_smiles(product_smiles)
    if not product:
        return [], {"rag_route_count": 0, "template_query_count": 0, "applied_template_count": 0}

    product_heavy = _heavy_atom_count(product)
    seen_reactant_sets = set()
    candidates: List[TemplateCandidate] = []
    applied_template_count = 0

    matcher = TemplateMatcher(db)
    rag_routes = []
    try:
        rag_routes = list(db.retrieve_similar_routes(product, retrieval_size) or [])
    except Exception:
        rag_routes = []

    template_hits: Dict[str, Tuple[float, str]] = {}
    template_db = getattr(db, "template_rules", set()) or set()
    if template_top_k > 0:
        for route_index, route_text in enumerate(rag_routes, start=1):
            query = _retrosynthetic_query_from_route(route_text)
            if not query:
                continue
            source = f"rag_route_{route_index}"
            if query in template_db:
                template_hits[query] = max(
                    template_hits.get(query, (0.0, source)),
                    (1.0, source),
                    key=lambda item: item[0],
                )
            for template, similarity in matcher.similar_templates(query, template_top_k):
                old = template_hits.get(template)
                if old is None or similarity > old[0]:
                    template_hits[template] = (float(similarity), source)

    applicable_hits, applicability_stats = matcher.product_applicable_templates(product)
    applicable_template_limit = max(1, int(applicable_template_limit or 1))
    ranked_hits = sorted(
        applicable_hits,
        key=lambda hit: (
            template_hits.get(hit.template, (0.0, ""))[0],
            hit.product_template_atoms,
            hit.product_template_bonds,
            -hit.checked_templates,
        ),
        reverse=True,
    )
    ranked_hits = ranked_hits[:applicable_template_limit]
    candidate_pool_limit = max(pool_size * 8, pool_size + 20)

    for template_index, hit in enumerate(
        ranked_hits,
        start=1,
    ):
        template = hit.template
        rag_similarity, rag_source = template_hits.get(template, (0.0, ""))
        source = (
            f"product_applicable+{rag_source}"
            if rag_source
            else "product_applicable"
        )
        error, generated = apply_retrosynthetic_template(
            template,
            product,
            max_outcomes=max_outcomes_per_template,
        )
        if error or not generated:
            continue
        applied_template_count += 1

        for reactants in generated[:max_outcomes_per_template]:
            clean_reactants = [canonicalize_smiles(item) for item in reactants]
            if not clean_reactants or any(not item for item in clean_reactants):
                continue
            clean_reactants = sorted(clean_reactants)
            reactant_key = tuple(clean_reactants)
            if reactant_key in seen_reactant_sets:
                continue
            seen_reactant_sets.add(reactant_key)

            candidate = _make_candidate(
                db=db,
                product=product,
                product_heavy=product_heavy,
                template=template,
                reactants=clean_reactants,
                rank=len(candidates) + 1,
                checked_templates=hit.checked_templates,
                source=source,
                rag_similarity=rag_similarity,
                sc_score_fn=sc_score_fn,
                experience_memory=experience_memory,
                prune_nonprogressive=prune_nonprogressive,
                max_lateral_move_risk=max_lateral_move_risk,
                strict_unstable_filter=strict_unstable_filter,
            )
            if candidate is not None:
                candidates.append(candidate)

        if len(candidates) >= candidate_pool_limit:
            break

    if not candidates and direct_max_templates and direct_max_templates > 0:
        expansions = expand_product_with_templates(
            db=db,
            product_smiles=product,
            max_routes=max(pool_size * 3, pool_size),
            max_templates=direct_max_templates,
            max_outcomes_per_template=max_outcomes_per_template,
        )
        for expansion in expansions:
            clean_reactants = [canonicalize_smiles(item) for item in expansion.reactants]
            if not clean_reactants or any(not item for item in clean_reactants):
                continue
            clean_reactants = sorted(clean_reactants)
            reactant_key = tuple(clean_reactants)
            if reactant_key in seen_reactant_sets:
                continue
            seen_reactant_sets.add(reactant_key)
            candidate = _make_candidate(
                db=db,
                product=product,
                product_heavy=product_heavy,
                template=expansion.template,
                reactants=clean_reactants,
                rank=len(candidates) + 1,
                checked_templates=expansion.checked_templates,
                source="direct_template_scan_fallback",
                rag_similarity=0.0,
                sc_score_fn=sc_score_fn,
                experience_memory=experience_memory,
                prune_nonprogressive=prune_nonprogressive,
                max_lateral_move_risk=max_lateral_move_risk,
                strict_unstable_filter=strict_unstable_filter,
            )
            if candidate is not None:
                candidates.append(candidate)
        applied_template_count += len(expansions)

    candidates.sort(key=lambda item: (item.final_score, item.rag_similarity), reverse=True)
    candidates = candidates[:pool_size]
    for new_rank, candidate in enumerate(candidates, start=1):
        candidate.rank = new_rank
        candidate.candidate_id = f"cand_{new_rank:03d}"
    return candidates, {
        "rag_route_count": len(rag_routes),
        "template_query_count": len(applicable_hits),
        "applied_template_count": applied_template_count,
        "rag_template_count": len(template_hits),
        "screened_template_count": applicability_stats.get("screened_templates", 0),
        "fingerprint_candidate_count": applicability_stats.get(
            "fingerprint_candidates",
            0,
        ),
    }


def format_candidate_graph_context(
    graph: CandidateGraph,
    max_nodes: int = 6,
    max_candidates_per_node: int = 5,
    max_template_chars: int = 260,
) -> str:
    if not graph or not graph.nodes:
        return "No template-grounded candidate graph was built for this molecule."

    nodes = []
    candidate_total = 0
    for node in graph.nodes[:max_nodes]:
        candidates = []
        for candidate in node.candidates[:max_candidates_per_node]:
            candidate_total += 1
            candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "product_smiles": candidate.product_smiles,
                    "reactant_smiles": candidate.reactants,
                    "reaction_smarts": _truncate(candidate.template, max_template_chars),
                    "source": candidate.source,
                    "rank": candidate.rank,
                    "final_score": round(candidate.final_score, 3),
                    "template_similarity_to_rag": round(candidate.rag_similarity, 3),
                    "experience_score": round(candidate.experience_score, 3),
                    "all_purchasable": candidate.all_purchasable,
                    "purchasable_count": candidate.purchasable_count,
                    "non_purchasable_count": candidate.non_purchasable_count,
                    "heavy_atom_delta": candidate.heavy_atom_delta,
                    "estimated_completion_score": round(
                        candidate.estimated_completion_score,
                        3,
                    ),
                    "lateral_move_risk": round(candidate.lateral_move_risk, 3),
                    "evidence": candidate.evidence,
                }
            )
        nodes.append(
            {
                "product_smiles": node.product_smiles,
                "graph_depth": node.depth,
                "candidate_count": len(node.candidates),
                "candidates": candidates,
            }
        )

    payload = {
        "role": "template_grounded_candidate_graph",
        "usage": (
            "Evidence only. These Product -> Reactants cuts come from product-side "
            "applicable database templates, then RAG/experience/heuristic ranking. "
            "Use them as high-priority anchors when composing a connected multi-step "
            "route, but still output a normal route. The validator, not this graph, "
            "decides acceptance."
        ),
        "root_smiles": graph.root_smiles,
        "stats": {
            "nodes_rendered": len(nodes),
            "candidates_rendered": candidate_total,
            "rag_route_count": graph.rag_route_count,
            "product_applicable_template_count": graph.template_query_count,
            "applied_template_count": graph.applied_template_count,
        },
        "nodes": nodes,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def format_template_candidate_context(
    candidates: Sequence[TemplateCandidate],
    max_items: int = 8,
) -> str:
    """Compact template-aware context for LLM pathway generation."""
    if not candidates:
        return "No local database-template candidates were found for this molecule."

    payload = []
    for candidate in candidates[:max_items]:
        payload.append(
            {
                "candidate_id": candidate.candidate_id,
                "product_smiles": candidate.product_smiles,
                "reactant_smiles": candidate.reactants,
                "reaction_smarts": _truncate(candidate.template, 300),
                "all_purchasable": candidate.all_purchasable,
                "purchasable_count": candidate.purchasable_count,
                "non_purchasable_count": candidate.non_purchasable_count,
                "heavy_atom_delta": candidate.heavy_atom_delta,
                "estimated_completion_score": round(
                    candidate.estimated_completion_score,
                    3,
                ),
                "lateral_move_risk": round(candidate.lateral_move_risk, 3),
                "nonstock_charged_count": candidate.nonstock_charged_count,
                "nonstock_radical_count": candidate.nonstock_radical_count,
                "nonstock_unstable_count": candidate.nonstock_unstable_count,
            }
        )
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _candidate_score(
    product: str,
    product_heavy: int,
    reactants: Sequence[str],
    reactant_heavies: Sequence[int],
    purchasable_count: int,
    frontier,
    nonstock_charged_count: int = 0,
    nonstock_radical_count: int = 0,
    nonstock_unstable_count: int = 0,
) -> float:
    max_reactant_heavy = max(reactant_heavies) if reactant_heavies else 0
    heavy_gain = max(0.0, (product_heavy - max_reactant_heavy) / max(product_heavy, 1))
    stock_gain = purchasable_count / max(len(reactants), 1)
    score = 100.0 * frontier.estimated_completion_score
    score += 20.0 * stock_gain
    score += 15.0 * frontier.complexity_delta_score
    score += 8.0 * heavy_gain
    score -= 12.0 * frontier.non_purchasable_count
    score -= 20.0 * frontier.lateral_move_risk
    score -= 28.0 * nonstock_charged_count
    score -= 35.0 * nonstock_radical_count
    score -= 30.0 * nonstock_unstable_count
    if frontier.lateral_move_risk >= 0.50 and frontier.non_purchasable_count:
        score -= 15.0
    if frontier.heavy_atom_delta <= 0 and purchasable_count == 0:
        score -= 25.0
    if frontier.total_non_purchasable_heavy_atoms > product_heavy:
        score -= 0.35 * (frontier.total_non_purchasable_heavy_atoms - product_heavy)
    score -= 30.0 if any(item == product for item in reactants) else 0.0
    if len(reactants) == 1:
        score -= 12.0
    elif len(reactants) == 2:
        score += 10.0
    elif len(reactants) == 3:
        score -= 12.0
    else:
        score -= 28.0 * (len(reactants) - 2)
    return score


def _retrosynthetic_query_from_route(route_text: str) -> str:
    if not route_text or ">>" not in route_text:
        return ""
    left, right = route_text.split(">>", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return ""
    return f"{right}>>{left}"


def _experience_score(
    product: str,
    reactants: Sequence[str],
    template: str,
    experience_memory=None,
) -> Tuple[float, str]:
    if experience_memory is None:
        return 0.0, "no_experience_memory"

    try:
        micro = experience_memory.micro_snapshot(product, compact=True)
    except Exception:
        return 0.0, "experience_unavailable"

    reactant_key = _reactant_key(reactants)
    score = 0.0
    evidence = []

    for preferred in micro.get("preferred_cuts", []) or []:
        preferred_reactants = preferred.get("reactants", [])
        preferred_key = _reactant_key(preferred_reactants)
        if preferred_key and preferred_key == reactant_key:
            score += 45.0
            evidence.append("matches_preferred_cut")
        if template and preferred.get("reaction_smarts") == template:
            score += 20.0
            evidence.append("matches_preferred_template")

    for avoided in micro.get("avoid_reactant_sets", []) or []:
        avoided_key = _reactant_key(avoided)
        if avoided_key and avoided_key == reactant_key:
            score -= 80.0
            evidence.append("matches_avoid_reactant_set")

    if not evidence:
        evidence.append("neutral_experience")
    return score, ",".join(evidence)


def _make_candidate(
    db,
    product: str,
    product_heavy: int,
    template: str,
    reactants: Sequence[str],
    rank: int,
    checked_templates: int,
    source: str,
    rag_similarity: float,
    sc_score_fn: Optional[Callable[[str], float]],
    experience_memory=None,
    prune_nonprogressive: bool = True,
    max_lateral_move_risk: float = 0.88,
    strict_unstable_filter: bool = True,
) -> Optional[TemplateCandidate]:
    purchasable = getattr(db, "is_purchasable", lambda _: False)
    reactant_heavies = [_heavy_atom_count(item) for item in reactants]
    purchasable_count = sum(1 for item in reactants if purchasable(item))
    frontier = compute_candidate_frontier_metrics(
        product_smiles=product,
        reactants=reactants,
        db=db,
        sc_score_fn=sc_score_fn,
    )
    nonstock_charged_count = _nonstock_charged_count(reactants, purchasable)
    nonstock_radical_count = _nonstock_radical_count(reactants, purchasable)
    nonstock_unstable_count = _nonstock_unstable_count(reactants, purchasable)
    if prune_nonprogressive and _is_nonprogressive_candidate(
        frontier=frontier,
        purchasable_count=purchasable_count,
        nonstock_charged_count=nonstock_charged_count,
        nonstock_radical_count=nonstock_radical_count,
        nonstock_unstable_count=nonstock_unstable_count,
        max_lateral_move_risk=max_lateral_move_risk,
        strict_unstable_filter=strict_unstable_filter,
    ):
        return None

    base_score = _candidate_score(
        product=product,
        product_heavy=product_heavy,
        reactants=reactants,
        reactant_heavies=reactant_heavies,
        purchasable_count=purchasable_count,
        frontier=frontier,
        nonstock_charged_count=nonstock_charged_count,
        nonstock_radical_count=nonstock_radical_count,
        nonstock_unstable_count=nonstock_unstable_count,
    )
    experience_score, evidence = _experience_score(
        product=product,
        reactants=reactants,
        template=template,
        experience_memory=experience_memory,
    )
    final_score = base_score + 25.0 * rag_similarity + experience_score
    return TemplateCandidate(
        candidate_id=f"cand_{rank:03d}",
        product_smiles=product,
        template=template,
        reactants=list(reactants),
        rank=rank,
        checked_templates=checked_templates,
        source=source,
        heuristic_score=base_score,
        purchasable_count=purchasable_count,
        all_purchasable=purchasable_count == len(reactants),
        max_reactant_heavy_atoms=max(reactant_heavies) if reactant_heavies else 0,
        product_heavy_atoms=product_heavy,
        non_purchasable_count=frontier.non_purchasable_count,
        max_non_purchasable_heavy_atoms=frontier.max_non_purchasable_heavy_atoms,
        total_non_purchasable_heavy_atoms=frontier.total_non_purchasable_heavy_atoms,
        heavy_atom_delta=frontier.heavy_atom_delta,
        complexity_delta_score=frontier.complexity_delta_score,
        estimated_completion_score=frontier.estimated_completion_score,
        lateral_move_risk=frontier.lateral_move_risk,
        nonstock_charged_count=nonstock_charged_count,
        nonstock_radical_count=nonstock_radical_count,
        nonstock_unstable_count=nonstock_unstable_count,
        rag_similarity=rag_similarity,
        experience_score=experience_score,
        final_score=final_score,
        evidence=evidence,
    )


def _reactant_key(reactants: Sequence[str]) -> Tuple[str, ...]:
    clean = []
    for item in reactants:
        canonical = canonicalize_smiles(item)
        if canonical:
            clean.append(canonical)
    return tuple(sorted(clean))


def _is_nonprogressive_candidate(
    frontier,
    purchasable_count: int,
    nonstock_charged_count: int,
    nonstock_radical_count: int,
    nonstock_unstable_count: int,
    max_lateral_move_risk: float,
    strict_unstable_filter: bool,
) -> bool:
    if strict_unstable_filter and nonstock_radical_count > 0:
        return True
    if strict_unstable_filter and nonstock_unstable_count > 0 and purchasable_count == 0:
        return True
    if strict_unstable_filter and nonstock_charged_count > 0 and purchasable_count == 0:
        return True
    if frontier.lateral_move_risk >= max_lateral_move_risk and purchasable_count == 0:
        return True
    if frontier.heavy_atom_delta < 0 and purchasable_count == 0:
        return True
    if (
        nonstock_charged_count > 0
        and purchasable_count == 0
        and frontier.lateral_move_risk >= 0.45
    ):
        return True
    return False


def _nonstock_charged_count(
    reactants: Sequence[str],
    purchasable: Callable[[str], bool],
) -> int:
    count = 0
    for item in reactants:
        if purchasable(item):
            continue
        if _has_nonstock_problematic_charge(item):
            count += 1
    return count


def _nonstock_radical_count(
    reactants: Sequence[str],
    purchasable: Callable[[str], bool],
) -> int:
    count = 0
    for item in reactants:
        if purchasable(item):
            continue
        if _has_radical(item):
            count += 1
    return count


def _nonstock_unstable_count(
    reactants: Sequence[str],
    purchasable: Callable[[str], bool],
) -> int:
    count = 0
    for item in reactants:
        if purchasable(item):
            continue
        if _has_radical(item) or _has_nonstock_problematic_charge(item):
            count += 1
    return count


def _has_formal_charge(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def _has_nonstock_problematic_charge(smiles: str) -> bool:
    clean = canonicalize_smiles(smiles)
    if clean in {"[OH-]", "[Cl-]", "[Br-]", "[I-]", "[F-]", "[Na+]", "[K+]", "[Li+]"}:
        return False
    mol = Chem.MolFromSmiles(clean or "")
    if mol is None:
        return False
    if not any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    heavy_atoms = mol.GetNumHeavyAtoms()
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    return has_carbon or heavy_atoms > 2


def _has_radical(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()
