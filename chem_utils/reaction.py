from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun


@dataclass
class ReactionMatchResult:
    is_valid: bool
    reason: str
    canonical_reactants: List[str]
    generated_reactants: List[List[str]]


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def canonicalize_reactant_set(smiles_list: Sequence[str]) -> Optional[List[str]]:
    canonical = []
    for smiles in smiles_list:
        clean = canonicalize_smiles(smiles)
        if clean is None:
            return None
        canonical.append(clean)
    return sorted(canonical)


def _format_rdchiral_template(reaction_smarts: str) -> str:
    left, right = reaction_smarts.split(">>", 1)
    left = left.strip()
    right = right.strip()
    if left.startswith("(") and left.endswith(")"):
        return f"{left}>>{right}"
    if "." in left:
        left = f"({left})"
    return f"{left}>>{right}"


def _append_unique_reactant_set(
    generated: List[List[str]],
    seen: set,
    reactants: Optional[Sequence[str]],
) -> None:
    if not reactants:
        return
    clean = canonicalize_reactant_set(reactants)
    if clean is None:
        return
    key = tuple(clean)
    if key not in seen:
        seen.add(key)
        generated.append(clean)


def _apply_template_with_rdchiral(
    reaction_smarts: str,
    product_smiles: str,
    max_outcomes: int,
) -> Tuple[Optional[str], List[List[str]]]:
    product = canonicalize_smiles(product_smiles)
    if product is None:
        return "invalid_product_smiles", []

    try:
        reactants = rdchiralReactants(product)
    except Exception:
        return "invalid_product_smiles", []

    try:
        reaction = rdchiralReaction(_format_rdchiral_template(reaction_smarts))
    except Exception:
        return "reaction_smarts_parse_error", []

    try:
        try:
            outcomes = rdchiralRun(reaction, reactants, combine_enantiomers=False)
        except TypeError:
            outcomes = rdchiralRun(reaction, reactants)
    except Exception:
        return "reaction_application_failed", []

    generated: List[List[str]] = []
    seen = set()
    for outcome in outcomes[:max_outcomes]:
        fragments = [item for item in str(outcome).split(".") if item]
        _append_unique_reactant_set(generated, seen, fragments)

    return None, generated


def apply_retrosynthetic_template(
    reaction_smarts: str,
    product_smiles: str,
    max_outcomes: int = 5000,
) -> Tuple[Optional[str], List[List[str]]]:
    """Apply a backward reaction SMARTS of the form product>>reactants.

    Strict template validation uses rdchiral to execute USPTO/Retro* templates.
    This function intentionally requires rdchiral so validation semantics stay
    aligned with the database-template execution protocol.
    """
    if not reaction_smarts or ">>" not in reaction_smarts:
        return "missing_or_malformed_reaction_smarts", []

    return _apply_template_with_rdchiral(
        reaction_smarts=reaction_smarts,
        product_smiles=product_smiles,
        max_outcomes=max_outcomes,
    )


def reaction_matches_reactants(
    reaction_smarts: str,
    product_smiles: str,
    reactant_smiles: Sequence[str],
    max_outcomes: int = 5000,
) -> ReactionMatchResult:
    expected = canonicalize_reactant_set(reactant_smiles)
    if expected is None:
        return ReactionMatchResult(False, "invalid_reactant_smiles", [], [])

    error, generated = apply_retrosynthetic_template(
        reaction_smarts=reaction_smarts,
        product_smiles=product_smiles,
        max_outcomes=max_outcomes,
    )
    if error:
        return ReactionMatchResult(False, error, expected, generated)

    if expected in generated:
        return ReactionMatchResult(True, "ok", expected, generated)

    return ReactionMatchResult(
        False,
        f"template_does_not_generate_declared_reactants:{len(generated)}_outcomes",
        expected,
        generated,
    )


def is_reaction_feasible(
    reaction_smarts: str,
    product_smiles: str,
    reactant_smiles: Optional[Sequence[str]] = None,
) -> bool:
    """Backward-compatible feasibility check.

    If reactants are provided, this keeps the legacy exact-reactant check. If
    reactants are omitted, it follows the strict validation gate and only checks
    whether the database template executes on the product and generates at least
    one sanitized precursor set.
    """
    if reactant_smiles is not None:
        return reaction_matches_reactants(
            reaction_smarts=reaction_smarts,
            product_smiles=product_smiles,
            reactant_smiles=reactant_smiles,
        ).is_valid

    error, generated = apply_retrosynthetic_template(
        reaction_smarts=reaction_smarts,
        product_smiles=product_smiles,
    )
    return error is None and bool(generated)
