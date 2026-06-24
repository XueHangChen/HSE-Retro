from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_llm_output(llm_response: str):
    from planner.route import ReactionStep, SynthesisRoute

    try:
        route_match = re.search(r"<ROUTE>(.*?)</ROUTE>", llm_response or "", re.S | re.I)
        exp_match = re.search(r"<EXPLANATION>(.*?)</EXPLANATION>", llm_response or "", re.S | re.I)
        chem_match = re.search(
            r"<chemical_reasoning>(.*?)</chemical_reasoning>",
            llm_response or "",
            re.S | re.I,
        )

        if not route_match:
            print("[Parser][ERROR] Missing <ROUTE> block.")
            return None

        route_text = route_match.group(1).strip()
        explanation = ""
        if exp_match:
            explanation = exp_match.group(1).strip()
        elif chem_match:
            explanation = chem_match.group(1).strip()

        step_dicts = _parse_route_payload(route_text)
        reaction_steps: List[ReactionStep] = []

        for item in step_dicts:
            try:
                step = ReactionStep(
                    molecule_set=_parse_list(
                        _get(item, "molecule_set", "Molecule set", "molecules", default=[])
                    ),
                    rational=str(_get(item, "rationale", "Rational", "rational", default="")),
                    product=_parse_list(
                        _get(item, "product_smiles", "Product", "product", default=[])
                    ),
                    reaction=_parse_scalar(
                        _get(item, "reaction_smarts", "Reaction", "reaction", default="")
                    ),
                    reactants=_parse_list(
                        _get(item, "reactant_smiles", "Reactants", "reactants", default=[])
                    ),
                    updated_molecule_set=_parse_list(
                        _get(
                            item,
                            "updated_molecule_set",
                            "Updated molecule set",
                            "updated_molecules",
                            default=[],
                        )
                    ),
                )
                reaction_steps.append(step)
            except Exception as exc:
                print(f"[Parser][WARN] Skipped malformed step: {exc}")

        if not reaction_steps:
            print("[Parser][ERROR] <ROUTE> contained no parseable steps.")
            return None

        return SynthesisRoute(steps=reaction_steps, explanation=explanation)
    except Exception as exc:
        print(f"[Parser][ERROR] Fatal parse error: {exc}")
        return None


def _parse_route_payload(route_text: str) -> List[Dict[str, Any]]:
    cleaned = _strip_code_fences(route_text)
    parsed = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(cleaned)
            break
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    # Fallback for LLMs that emit adjacent JSON dicts without a surrounding list.
    step_dicts = []
    for match in re.finditer(r"\{.*?\}", cleaned, re.S):
        snippet = match.group(0)
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(snippet)
                if isinstance(value, dict):
                    step_dicts.append(value)
                break
            except Exception:
                continue
    return step_dicts


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|python)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _get(data: Dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in data:
            return data[key]
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return default


def _parse_scalar(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "").strip()


def _parse_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, str):
                return [parsed.strip()]
        except Exception:
            pass

    return [item.strip(" '\"") for item in value.strip("[]").split(",") if item.strip(" '\"")]
