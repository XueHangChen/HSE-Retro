from __future__ import annotations

from rdkit import Chem

from config import default_config
from data.database import db_instance
from planner.route import SynthesisRoute
from planner.validation import validate_synthesis_route


def canonicalize_smiles(smi: str) -> str:
    if not smi:
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return smi
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smi


class ThreeLevelEvaluator:
    """Wang-style molecule/reaction/route validator.

    The route may be valid but incomplete. In that case it remains unsuccessful
    and carries a validation report with non-purchasable terminal molecules.
    """

    def __init__(self, db):
        self.db = db
        print("[Evaluation][INFO] Strict three-level evaluator initialized.")

    def evaluate_route(self, route: SynthesisRoute, target_molecule: str) -> SynthesisRoute:
        report = validate_synthesis_route(
            route=route,
            target_molecule=target_molecule,
            db=self.db,
            require_template_db_match=getattr(
                default_config, "REQUIRE_REACTION_DB_MATCH", False
            ),
            template_backed_validation=getattr(
                default_config, "TEMPLATE_BACKED_VALIDATION", True
            ),
            template_match_top_k=getattr(default_config, "TEMPLATE_MATCH_TOP_K", 100),
            max_outcomes=getattr(default_config, "VALIDATION_MAX_OUTCOMES", 5000),
            template_generated_reactants_as_source_of_truth=getattr(
                default_config,
                "TEMPLATE_GENERATED_REACTANTS_AS_SOURCE_OF_TRUTH",
                True,
            ),
        )
        route.validation_report = report

        if report.normalized_route is not None:
            route.steps = report.normalized_route.steps

        route.is_successful = report.is_successful

        if report.invalid_steps and route.steps:
            invalid = report.invalid_steps[0]
            idx = min(invalid.step_index, len(route.steps) - 1)
            route.steps[idx].is_valid = False
            route.steps[idx].feedback = invalid.reason
        elif route.steps and report.non_purchasable:
            route.steps[-1].feedback = (
                "Valid reactions but terminal molecules are not in stock: "
                + " | ".join(report.non_purchasable)
            )
        elif route.steps and report.is_successful:
            route.steps[-1].feedback = (
                "Route successful: strict reaction checks passed and all leaves are in stock."
            )

        return route


evaluator_instance = ThreeLevelEvaluator(db=db_instance)
