from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from rdkit import Chem

from chem_utils.reaction import canonicalize_smiles


ScoreFn = Callable[[str], float]


@dataclass
class MoleculeFrontierItem:
    smiles: str
    is_purchasable: bool
    heavy_atoms: int
    sc_score: float
    visit_count: int = 0
    value: float = 0.0


@dataclass
class FrontierMetrics:
    route_steps: int = 0
    terminal_count: int = 0
    purchasable_terminal_count: int = 0
    non_purchasable_terminal_count: int = 0
    stock_ratio: float = 0.0
    max_non_purchasable_heavy_atoms: int = 0
    max_non_purchasable_scscore: float = 0.0
    total_non_purchasable_heavy_atoms: int = 0
    total_non_purchasable_scscore: float = 0.0
    max_non_purchasable_complexity_norm: float = 0.0
    frontier_completion_score: float = 0.0
    terminal_molecules: List[MoleculeFrontierItem] = field(default_factory=list)
    non_purchasable_molecules: List[MoleculeFrontierItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateFrontierMetrics:
    reactant_count: int
    purchasable_count: int
    non_purchasable_count: int
    stock_ratio: float
    product_heavy_atoms: int
    max_reactant_heavy_atoms: int
    max_non_purchasable_heavy_atoms: int
    total_non_purchasable_heavy_atoms: int
    heavy_atom_delta: int
    max_non_purchasable_scscore: float
    complexity_delta_score: float
    estimated_completion_score: float
    lateral_move_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_candidate_frontier_metrics(
    product_smiles: str,
    reactants: Sequence[str],
    db,
    sc_score_fn: Optional[ScoreFn] = None,
) -> CandidateFrontierMetrics:
    product = canonicalize_smiles(product_smiles)
    clean_reactants = [canonicalize_smiles(item) for item in reactants]
    product_heavy = heavy_atom_count(product)
    reactant_heavies = [heavy_atom_count(item) for item in clean_reactants]
    purchasable = getattr(db, "is_purchasable", lambda _: False)
    non_purchasable = [item for item in clean_reactants if not purchasable(item)]
    non_purch_heavies = [heavy_atom_count(item) for item in non_purchasable]
    sc_fn = sc_score_fn or (lambda _: 5.0)
    non_purch_scs = [_safe_scscore(item, sc_fn) for item in non_purchasable]

    max_reactant_heavy = max(reactant_heavies) if reactant_heavies else 0
    max_non_purch_heavy = max(non_purch_heavies) if non_purch_heavies else 0
    max_non_purch_sc = max(non_purch_scs) if non_purch_scs else 0.0
    purchasable_count = len(clean_reactants) - len(non_purchasable)
    stock_ratio = purchasable_count / len(clean_reactants) if clean_reactants else 0.0
    heavy_delta = product_heavy - max_reactant_heavy
    complexity_delta_score = max(0.0, min(1.0, heavy_delta / max(product_heavy, 1)))

    complexity_norm = _complexity_norm(max_non_purch_heavy, max_non_purch_sc)
    unresolved_penalty = min(1.0, len(non_purchasable) / 4.0)
    lateral_move_risk = _lateral_move_risk(
        product=product,
        product_heavy=product_heavy,
        reactants=clean_reactants,
        max_reactant_heavy=max_reactant_heavy,
    )
    completion = (
        0.50 * stock_ratio
        + 0.25 * (1.0 - complexity_norm)
        + 0.15 * complexity_delta_score
        + 0.10 * (1.0 - unresolved_penalty)
        - 0.15 * lateral_move_risk
    )
    completion = max(0.0, min(1.0, completion))

    return CandidateFrontierMetrics(
        reactant_count=len(clean_reactants),
        purchasable_count=purchasable_count,
        non_purchasable_count=len(non_purchasable),
        stock_ratio=stock_ratio,
        product_heavy_atoms=product_heavy,
        max_reactant_heavy_atoms=max_reactant_heavy,
        max_non_purchasable_heavy_atoms=max_non_purch_heavy,
        total_non_purchasable_heavy_atoms=sum(non_purch_heavies),
        heavy_atom_delta=heavy_delta,
        max_non_purchasable_scscore=max_non_purch_sc,
        complexity_delta_score=complexity_delta_score,
        estimated_completion_score=completion,
        lateral_move_risk=lateral_move_risk,
    )


def compute_or_frontier_metrics(
    or_node,
    db,
    sc_score_fn: Optional[ScoreFn] = None,
    max_depth: int = 16,
) -> FrontierMetrics:
    metrics = FrontierMetrics()
    _walk_or_frontier(
        or_node=or_node,
        metrics=metrics,
        db=db,
        sc_score_fn=sc_score_fn or (lambda _: 5.0),
        visited_or=set(),
        visited_and=set(),
        max_depth=max_depth,
    )
    return _finalize_metrics(metrics)


def compute_and_frontier_metrics(
    and_node,
    db,
    sc_score_fn: Optional[ScoreFn] = None,
    max_depth: int = 16,
) -> FrontierMetrics:
    metrics = FrontierMetrics(route_steps=1)
    for child in getattr(and_node, "children", []):
        _walk_or_frontier(
            or_node=child,
            metrics=metrics,
            db=db,
            sc_score_fn=sc_score_fn or (lambda _: 5.0),
            visited_or=set(),
            visited_and={and_node},
            max_depth=max_depth,
        )
    return _finalize_metrics(metrics)


def format_frontier_context(metrics: FrontierMetrics, max_items: int = 8) -> str:
    non_purch = sorted(
        metrics.non_purchasable_molecules,
        key=lambda item: (item.heavy_atoms, item.sc_score),
        reverse=True,
    )
    lines = [
        f"route_steps={metrics.route_steps}",
        (
            "terminal_count="
            f"{metrics.terminal_count}; purchasable={metrics.purchasable_terminal_count}; "
            f"non_purchasable={metrics.non_purchasable_terminal_count}; "
            f"stock_ratio={metrics.stock_ratio:.3f}"
        ),
        (
            "largest_non_purchasable="
            f"heavy_atoms={metrics.max_non_purchasable_heavy_atoms}; "
            f"scscore={metrics.max_non_purchasable_scscore:.3f}; "
            f"frontier_completion_score={metrics.frontier_completion_score:.3f}"
        ),
    ]
    if non_purch:
        lines.append("non_purchasable_frontier:")
        for item in non_purch[:max_items]:
            lines.append(
                f"- {item.smiles} | heavy={item.heavy_atoms} | "
                f"scscore={item.sc_score:.3f} | value={item.value:.3f} | "
                f"visits={item.visit_count}"
            )
    else:
        lines.append("non_purchasable_frontier: none")
    return "\n".join(lines)


def _walk_or_frontier(
    or_node,
    metrics: FrontierMetrics,
    db,
    sc_score_fn: ScoreFn,
    visited_or: Set[Any],
    visited_and: Set[Any],
    max_depth: int,
) -> None:
    if or_node in visited_or:
        _add_terminal(or_node, metrics, db, sc_score_fn)
        return
    visited_or.add(or_node)

    if getattr(or_node, "is_purchasable", False):
        _add_terminal(or_node, metrics, db, sc_score_fn)
        return

    legal_children = [
        child
        for child in getattr(or_node, "children", [])
        if getattr(child, "is_valid", False)
    ]
    if not legal_children:
        _add_terminal(or_node, metrics, db, sc_score_fn)
        return

    best_child = max(legal_children, key=lambda child: getattr(child, "value", 0.0))
    if best_child in visited_and or getattr(best_child, "depth", 0) >= max_depth:
        _add_terminal(or_node, metrics, db, sc_score_fn)
        return

    visited_and.add(best_child)
    metrics.route_steps += 1
    for child_or in getattr(best_child, "children", []):
        _walk_or_frontier(
            or_node=child_or,
            metrics=metrics,
            db=db,
            sc_score_fn=sc_score_fn,
            visited_or=set(visited_or),
            visited_and=set(visited_and),
            max_depth=max_depth,
        )


def _add_terminal(or_node, metrics: FrontierMetrics, db, sc_score_fn: ScoreFn) -> None:
    smiles = canonicalize_smiles(getattr(or_node, "smiles", ""))
    purchasable = bool(getattr(or_node, "is_purchasable", False))
    if not purchasable:
        purchasable = bool(getattr(db, "is_purchasable", lambda _: False)(smiles))
    item = MoleculeFrontierItem(
        smiles=smiles,
        is_purchasable=purchasable,
        heavy_atoms=heavy_atom_count(smiles),
        sc_score=_safe_scscore(smiles, sc_score_fn),
        visit_count=getattr(or_node, "visit_count", 0),
        value=getattr(or_node, "value", 0.0),
    )
    metrics.terminal_molecules.append(item)
    if not item.is_purchasable:
        metrics.non_purchasable_molecules.append(item)


def _finalize_metrics(metrics: FrontierMetrics) -> FrontierMetrics:
    metrics.terminal_count = len(metrics.terminal_molecules)
    metrics.purchasable_terminal_count = sum(
        1 for item in metrics.terminal_molecules if item.is_purchasable
    )
    metrics.non_purchasable_terminal_count = len(metrics.non_purchasable_molecules)
    if metrics.terminal_count:
        metrics.stock_ratio = metrics.purchasable_terminal_count / metrics.terminal_count

    if metrics.non_purchasable_molecules:
        metrics.max_non_purchasable_heavy_atoms = max(
            item.heavy_atoms for item in metrics.non_purchasable_molecules
        )
        metrics.max_non_purchasable_scscore = max(
            item.sc_score for item in metrics.non_purchasable_molecules
        )
        metrics.total_non_purchasable_heavy_atoms = sum(
            item.heavy_atoms for item in metrics.non_purchasable_molecules
        )
        metrics.total_non_purchasable_scscore = sum(
            item.sc_score for item in metrics.non_purchasable_molecules
        )

    metrics.max_non_purchasable_complexity_norm = _complexity_norm(
        metrics.max_non_purchasable_heavy_atoms,
        metrics.max_non_purchasable_scscore,
    )
    unresolved_penalty = min(1.0, metrics.non_purchasable_terminal_count / 5.0)
    metrics.frontier_completion_score = max(
        0.0,
        min(
            1.0,
            0.55 * metrics.stock_ratio
            + 0.30 * (1.0 - metrics.max_non_purchasable_complexity_norm)
            + 0.15 * (1.0 - unresolved_penalty),
        ),
    )
    return metrics


def heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def _safe_scscore(smiles: str, sc_score_fn: ScoreFn) -> float:
    try:
        value = float(sc_score_fn(smiles))
        if value != value:
            return 5.0
        return max(1.0, min(5.0, value))
    except Exception:
        return 5.0


def _complexity_norm(heavy_atoms: int, scscore: float) -> float:
    heavy_norm = max(0.0, min(1.0, heavy_atoms / 50.0))
    sc_norm = max(0.0, min(1.0, (scscore - 1.0) / 4.0)) if scscore else 0.0
    return max(heavy_norm, sc_norm)


def _lateral_move_risk(
    product: str,
    product_heavy: int,
    reactants: Sequence[str],
    max_reactant_heavy: int,
) -> float:
    if not reactants:
        return 1.0
    risk = 0.0
    if any(item == product for item in reactants):
        risk += 0.5
    if len(reactants) == 1:
        risk += 0.25
    if max_reactant_heavy >= max(1, product_heavy - 1):
        risk += 0.35
    if max_reactant_heavy > product_heavy:
        risk += 0.25
    return max(0.0, min(1.0, risk))
