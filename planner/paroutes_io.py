from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from chem_utils.reaction import canonicalize_smiles
from planner.route import ReactionStep, SynthesisRoute


def load_smiles_file(
    path: str | Path,
    limit: Optional[int] = None,
    reaction_side: Optional[str] = "left",
) -> List[str]:
    """Load target or stock files.

    If a line contains a retrosynthesis record like ``target>>precursors``,
    ``reaction_side="left"`` keeps the target side. Use ``None`` to require
    plain one-SMILES-per-line input.
    """
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip().split()[0] if line.strip() else ""
            if not raw or raw.startswith("#"):
                continue
            if reaction_side and ">>" in raw:
                sides = raw.split(">>", 1)
                if reaction_side == "left":
                    raw = sides[0]
                elif reaction_side == "right":
                    raw = sides[1]
            clean = canonicalize_smiles(raw)
            if clean:
                items.append(clean)
            if limit is not None and len(items) >= limit:
                break
    return items


def load_stock(path: str | Path) -> Set[str]:
    return set(load_smiles_file(path))


def apply_stock_to_db(db, stock_set: Iterable[str]) -> None:
    """Replace the active purchasable set for PaRoutes-style experiments."""
    canonical_stock = {canonicalize_smiles(item) for item in stock_set}
    canonical_stock.discard(None)
    db.purchasable_set = set(canonical_stock)
    db.purchasable_db = set(canonical_stock)


def route_to_paroutes_json(
    target_smiles: str,
    synthesis_route: SynthesisRoute,
    stock_set: Iterable[str],
    include_reactions: bool = True,
) -> Dict[str, Any]:
    """Convert a validated sequential route into PaRoutes reaction-tree JSON."""
    stock = {canonicalize_smiles(item) for item in stock_set}
    stock.discard(None)
    target = canonicalize_smiles(target_smiles) or target_smiles
    steps_by_product = _steps_by_product(synthesis_route.steps if synthesis_route else [])
    return _build_mol_node(
        smiles=target,
        steps_by_product=steps_by_product,
        stock=stock,
        include_reactions=include_reactions,
        trail=set(),
    )


def routes_to_target_dict(
    targets: Sequence[str],
    routes: Sequence[Optional[SynthesisRoute]],
    stock_set: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for target, route in zip(targets, routes):
        clean_target = canonicalize_smiles(target) or target
        if route and route.is_successful:
            output[clean_target] = [route_to_paroutes_json(clean_target, route, stock_set)]
        else:
            output[clean_target] = []
    return output


def routes_to_official_list(
    targets: Sequence[str],
    routes: Sequence[Optional[SynthesisRoute]],
    stock_set: Iterable[str],
) -> List[List[Dict[str, Any]]]:
    """Format expected by PaRoutes analysis: one route-list per target."""
    output: List[List[Dict[str, Any]]] = []
    for target, route in zip(targets, routes):
        clean_target = canonicalize_smiles(target) or target
        if route and route.is_successful:
            output.append([route_to_paroutes_json(clean_target, route, stock_set)])
        else:
            output.append([])
    return output


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def validate_paroutes_tree(tree: Dict[str, Any], stock_set: Iterable[str]) -> bool:
    stock = {canonicalize_smiles(item) for item in stock_set}
    stock.discard(None)
    return _tree_is_solved(tree, stock)


def _steps_by_product(steps: Sequence[ReactionStep]) -> Dict[str, ReactionStep]:
    output: Dict[str, ReactionStep] = {}
    for step in steps:
        product = canonicalize_smiles(step.product[0]) if step.product else None
        if product:
            output[product] = step
    return output


def _build_mol_node(
    smiles: str,
    steps_by_product: Dict[str, ReactionStep],
    stock: Set[str],
    include_reactions: bool,
    trail: Set[str],
) -> Dict[str, Any]:
    clean = canonicalize_smiles(smiles) or smiles
    node: Dict[str, Any] = {
        "smiles": clean,
        "type": "mol",
        "in_stock": clean in stock,
    }

    if clean in trail:
        return node

    step = steps_by_product.get(clean)
    if step is None:
        return node

    reaction_node: Dict[str, Any] = {
        "type": "reaction",
        "children": [
            _build_mol_node(
                reactant,
                steps_by_product,
                stock,
                include_reactions,
                trail | {clean},
            )
            for reactant in step.reactants
        ],
    }

    if include_reactions:
        reaction_node["smiles"] = ""
        reaction_node["metadata"] = {
            "reaction_smarts": step.reaction,
            "rationale": step.rational,
            "is_valid": step.is_valid,
            "feedback": step.feedback,
        }

    node["children"] = [reaction_node]
    return node


def _tree_is_solved(tree: Dict[str, Any], stock: Set[str]) -> bool:
    if tree.get("type") == "mol":
        clean = canonicalize_smiles(tree.get("smiles", "")) or tree.get("smiles", "")
        children = tree.get("children") or []
        if not children:
            return clean in stock and bool(tree.get("in_stock", False))
        return all(_tree_is_solved(child, stock) for child in children)

    return all(_tree_is_solved(child, stock) for child in tree.get("children", []))
