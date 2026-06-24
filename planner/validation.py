from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from chem_utils import molecule
from chem_utils.reaction import canonicalize_smiles, reaction_matches_reactants
from planner.route import ReactionStep, SynthesisRoute
from planner.template_matcher import match_step_to_template


@dataclass
class StepValidationReport:
    is_valid: bool
    reason: str
    step_index: int
    product: List[str] = field(default_factory=list)
    reactants: List[str] = field(default_factory=list)
    reaction: str = ""
    feedback: str = ""
    matched_template: str = ""
    match_source: str = ""
    similarity: float = 0.0
    generated_reactants: List[List[str]] = field(default_factory=list)

    def as_experience(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        product = self.product[0] if self.product else "UNKNOWN"
        reactants = ".".join(self.reactants) if self.reactants else "UNKNOWN"
        template = self.matched_template or self.reaction or "NONE"
        source = f" | Source={self.match_source}" if self.match_source else ""
        similarity = f" | Similarity={self.similarity:.3f}" if self.similarity else ""
        return (
            f"{status} step {self.step_index}: Product={product} | "
            f"Reaction={template} | Reactants={reactants}{source}{similarity} | "
            f"Reason={self.reason}"
        )


@dataclass
class RouteValidationReport:
    steps_valid: bool
    is_successful: bool
    valid_steps: List[StepValidationReport] = field(default_factory=list)
    invalid_steps: List[StepValidationReport] = field(default_factory=list)
    terminal_molecules: List[str] = field(default_factory=list)
    non_purchasable: List[str] = field(default_factory=list)
    normalized_route: Optional[SynthesisRoute] = None

    @property
    def first_invalid_index(self) -> int:
        if not self.invalid_steps:
            return -1
        return self.invalid_steps[0].step_index

    def good_experience(self) -> List[str]:
        return [item.as_experience() for item in self.valid_steps]

    def bad_experience(self) -> List[str]:
        bad = [item.as_experience() for item in self.invalid_steps]
        if self.steps_valid and self.non_purchasable:
            bad.append(
                "PARTIAL route: chemically valid so far, but terminal molecules "
                f"are not in stock: {' | '.join(self.non_purchasable)}"
            )
        return bad


def _parse_reaction_scalar(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _canonicalize_list(smiles_list: Sequence[str]) -> Optional[List[str]]:
    canonical = []
    for item in smiles_list:
        clean = canonicalize_smiles(str(item))
        if clean is None:
            return None
        canonical.append(clean)
    return canonical


def validate_synthesis_route(
    route: SynthesisRoute,
    target_molecule: str,
    db,
    require_template_db_match: bool = False,
    template_backed_validation: bool = True,
    template_match_top_k: int = 100,
    max_outcomes: int = 5000,
    template_generated_reactants_as_source_of_truth: bool = True,
) -> RouteValidationReport:
    """Validate a sequential retrosynthesis route before it can enter search.

    A route may be chemically valid but incomplete. Such a route is accepted for
    tree expansion, but it is not marked successful until every terminal
    molecule is in the active stock.

    In strict template-backed mode, database templates are treated as the source
    of truth: after a template is matched and executed on the product, the route
    is normalized with the generated reactants.
    """
    target = canonicalize_smiles(target_molecule)
    if target is None:
        report = StepValidationReport(
            is_valid=False,
            reason="invalid_target_smiles",
            step_index=0,
            product=[str(target_molecule)],
        )
        return RouteValidationReport(False, False, invalid_steps=[report])

    if route is None or not route.steps:
        report = StepValidationReport(False, "empty_route", 0)
        return RouteValidationReport(False, False, invalid_steps=[report])

    active: Set[str] = {target}
    normalized_steps: List[ReactionStep] = []
    valid_reports: List[StepValidationReport] = []
    invalid_reports: List[StepValidationReport] = []

    for index, step in enumerate(route.steps):
        products = _canonicalize_list(step.product or [])
        reactants = _canonicalize_list(step.reactants or [])
        reaction = _parse_reaction_scalar(step.reaction)
        use_generated_reactants = (
            template_backed_validation
            and template_generated_reactants_as_source_of_truth
        )

        if not products or len(products) != 1:
            report = StepValidationReport(
                False,
                "step_must_have_exactly_one_product",
                index,
                products or [],
                reactants or [],
                reaction,
            )
            invalid_reports.append(report)
            step.is_valid = False
            step.feedback = report.reason
            break

        if reactants is None or (not reactants and not use_generated_reactants):
            report = StepValidationReport(
                False,
                "invalid_or_empty_reactants",
                index,
                products,
                [],
                reaction,
            )
            invalid_reports.append(report)
            step.is_valid = False
            step.feedback = report.reason
            break

        product = products[0]
        if product not in active:
            report = StepValidationReport(
                False,
                f"route_connectivity_error:product_not_active:{product}",
                index,
                products,
                reactants,
                reaction,
            )
            invalid_reports.append(report)
            step.is_valid = False
            step.feedback = report.reason
            break

        if not all(molecule.is_valid(item) for item in products + reactants):
            report = StepValidationReport(
                False,
                "rdkit_invalid_smiles",
                index,
                products,
                reactants,
                reaction,
            )
            invalid_reports.append(report)
            step.is_valid = False
            step.feedback = report.reason
            break

        matched_template = reaction
        match_source = "llm_direct"
        similarity = 0.0
        generated_reactants = []
        validated_reactants = reactants or []
        reactants_match_declared = True

        if template_backed_validation:
            template_match = match_step_to_template(
                db=db,
                product_smiles=product,
                reactant_smiles=reactants,
                draft_reaction_smarts=reaction,
                top_k=template_match_top_k,
                max_outcomes=max_outcomes,
                require_declared_reactants=not use_generated_reactants,
            )
            if not template_match.is_matched:
                report = StepValidationReport(
                    False,
                    template_match.reason,
                    index,
                    products,
                    reactants,
                    reaction,
                    match_source=template_match.source,
                    similarity=template_match.similarity,
                    generated_reactants=template_match.generated_reactants[:5],
                )
                invalid_reports.append(report)
                step.is_valid = False
                step.feedback = report.reason
                break

            matched_template = template_match.matched_template
            match_source = template_match.source
            similarity = template_match.similarity
            generated_reactants = template_match.generated_reactants[:5]
            validated_reactants = template_match.selected_reactants or reactants or []
            reactants_match_declared = template_match.reactants_match_declared
            if not validated_reactants:
                report = StepValidationReport(
                    False,
                    "template_matched_but_no_generated_reactants",
                    index,
                    products,
                    reactants or [],
                    reaction,
                    match_source=match_source,
                    similarity=similarity,
                    generated_reactants=generated_reactants,
                )
                invalid_reports.append(report)
                step.is_valid = False
                step.feedback = report.reason
                break

        else:
            if require_template_db_match and hasattr(db, "match_reaction"):
                if db.match_reaction(reaction) is None:
                    report = StepValidationReport(
                        False,
                        "reaction_template_not_in_database",
                        index,
                        products,
                        reactants,
                        reaction,
                    )
                    invalid_reports.append(report)
                    step.is_valid = False
                    step.feedback = report.reason
                    break

            match = reaction_matches_reactants(
                reaction,
                product,
                reactants,
                max_outcomes=max_outcomes,
            )
            if not match.is_valid:
                report = StepValidationReport(
                    False,
                    match.reason,
                    index,
                    products,
                    reactants,
                    reaction,
                    generated_reactants=match.generated_reactants[:5],
                )
                invalid_reports.append(report)
                step.is_valid = False
                step.feedback = report.reason
                break
            generated_reactants = match.generated_reactants[:5]
            validated_reactants = reactants

        active.remove(product)
        active.update(validated_reactants)
        if template_backed_validation:
            feedback = (
                f"template_backed_match:{match_source}"
                + (f":similarity={similarity:.3f}" if similarity else "")
            )
            if use_generated_reactants and not reactants_match_declared:
                feedback += ":reactants_rewritten_from_template"
        else:
            feedback = "valid_llm_direct_reaction_match"
        normalized_step = ReactionStep(
            molecule_set=sorted(active | {product}),
            rational=step.rational,
            product=products,
            reaction=matched_template,
            reactants=validated_reactants,
            updated_molecule_set=sorted(active),
            is_valid=True,
            feedback=feedback,
        )
        normalized_steps.append(normalized_step)
        valid_reports.append(
            StepValidationReport(
                True,
                "valid_template_generated_reactants_match"
                if use_generated_reactants
                else "valid_template_backed_reaction_match",
                index,
                products,
                validated_reactants,
                matched_template,
                feedback=feedback,
                matched_template=matched_template,
                match_source=match_source,
                similarity=similarity,
                generated_reactants=generated_reactants,
            )
        )

    steps_valid = not invalid_reports and bool(normalized_steps)
    terminal_molecules = sorted(active)
    is_purchasable = getattr(db, "is_purchasable", lambda _: False)
    non_purchasable = [
        item for item in terminal_molecules
        if not is_purchasable(item)
    ]
    is_successful = steps_valid and not non_purchasable

    normalized_route = SynthesisRoute(
        steps=normalized_steps,
        explanation=route.explanation,
        reward=route.reward,
        is_successful=is_successful,
    )

    return RouteValidationReport(
        steps_valid=steps_valid,
        is_successful=is_successful,
        valid_steps=valid_reports,
        invalid_steps=invalid_reports,
        terminal_molecules=terminal_molecules,
        non_purchasable=non_purchasable,
        normalized_route=normalized_route,
    )
