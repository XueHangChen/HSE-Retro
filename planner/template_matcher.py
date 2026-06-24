from __future__ import annotations

import gzip
import os
import pickle
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdChemReactions

from config import default_config
from chem_utils.reaction import (
    apply_retrosynthetic_template,
    canonicalize_reactant_set,
    canonicalize_smiles,
    reaction_matches_reactants,
)

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


@dataclass
class TemplateBackedMatch:
    is_matched: bool
    reason: str
    matched_template: str = ""
    source: str = ""
    similarity: float = 0.0
    checked_templates: int = 0
    query_smarts: str = ""
    generated_reactants: List[List[str]] = field(default_factory=list)
    selected_reactants: List[str] = field(default_factory=list)
    reactants_match_declared: bool = False


@dataclass
class TemplateExpansion:
    template: str
    reactants: List[str]
    checked_templates: int
    source: str = "template_direct"


@dataclass
class TemplateApplicabilityHit:
    template: str
    checked_templates: int
    source: str = "product_applicable"
    product_template_atoms: int = 0
    product_template_bonds: int = 0


@dataclass
class TemplateVerification:
    is_valid: bool
    reason: str
    selected_reactants: List[str] = field(default_factory=list)
    generated_reactants: List[List[str]] = field(default_factory=list)
    reactants_match_declared: bool = False


class TemplateMatcher:
    """Database-backed reaction validator.

    The LLM proposes product/reactants and optionally a draft SMARTS. In strict
    template-backed mode, the database template is the source of truth: once a
    template can run on the product, the generated reactants are used to
    normalize the route. Legacy exact-reactant matching is still available via
    ``require_declared_reactants=True``.
    """

    def __init__(self, db):
        self.db = db

    def match_step(
        self,
        product_smiles: str,
        reactant_smiles: Sequence[str],
        draft_reaction_smarts: str = "",
        top_k: int = 100,
        max_outcomes: int = 5000,
        require_declared_reactants: bool = True,
    ) -> TemplateBackedMatch:
        product = canonicalize_smiles(product_smiles)
        declared_reactants = canonicalize_reactant_set(reactant_smiles or [])
        if not product or declared_reactants is None:
            return TemplateBackedMatch(False, "invalid_product_or_reactants")
        if require_declared_reactants and not declared_reactants:
            return TemplateBackedMatch(False, "invalid_or_empty_reactants")

        query_smarts = self._build_queries(
            product,
            declared_reactants,
            draft_reaction_smarts,
        )
        if not query_smarts:
            return TemplateBackedMatch(False, "missing_reaction_search_hint")
        templates = self._template_list()
        if not templates:
            return TemplateBackedMatch(False, "empty_template_database")

        checked = 0
        seen_templates = set()
        last_generated: List[List[str]] = []

        for query_index, query in enumerate(query_smarts):
            exact_candidates = []
            if draft_reaction_smarts and draft_reaction_smarts in self._template_set():
                exact_candidates.append(draft_reaction_smarts)
            if query in self._template_set():
                exact_candidates.append(query)

            for template in exact_candidates:
                if template in seen_templates:
                    continue
                seen_templates.add(template)
                checked += 1
                match = self._verify_template(
                    template,
                    product,
                    declared_reactants,
                    max_outcomes,
                    require_declared_reactants=require_declared_reactants,
                )
                if match.generated_reactants:
                    last_generated = match.generated_reactants[:5]
                if match.is_valid:
                    return TemplateBackedMatch(
                        True,
                        "template_exact_match",
                        matched_template=template,
                        source="exact",
                        similarity=1.0,
                        checked_templates=checked,
                        query_smarts=query,
                        generated_reactants=match.generated_reactants[:5],
                        selected_reactants=match.selected_reactants,
                        reactants_match_declared=match.reactants_match_declared,
                    )

            for template in self._substructure_match_templates(query):
                if template in seen_templates:
                    continue
                seen_templates.add(template)
                checked += 1
                match = self._verify_template(
                    template,
                    product,
                    declared_reactants,
                    max_outcomes,
                    require_declared_reactants=require_declared_reactants,
                )
                if match.generated_reactants:
                    last_generated = match.generated_reactants[:5]
                if match.is_valid:
                    return TemplateBackedMatch(
                        True,
                        "template_substructure_match",
                        matched_template=template,
                        source="substructure",
                        similarity=1.0,
                        checked_templates=checked,
                        query_smarts=query,
                        generated_reactants=match.generated_reactants[:5],
                        selected_reactants=match.selected_reactants,
                        reactants_match_declared=match.reactants_match_declared,
                    )

            ranked = self._topk_similar_templates(query, top_k)
            for template, similarity in ranked:
                if template in seen_templates:
                    continue
                seen_templates.add(template)
                checked += 1
                match = self._verify_template(
                    template,
                    product,
                    declared_reactants,
                    max_outcomes,
                    require_declared_reactants=require_declared_reactants,
                )
                if match.generated_reactants:
                    last_generated = match.generated_reactants[:5]
                if match.is_valid:
                    return TemplateBackedMatch(
                        True,
                        "template_topk_match",
                        matched_template=template,
                        source=f"top{top_k}",
                        similarity=similarity,
                        checked_templates=checked,
                        query_smarts=query,
                        generated_reactants=match.generated_reactants[:5],
                        selected_reactants=match.selected_reactants,
                        reactants_match_declared=match.reactants_match_declared,
                    )

            if query_index == 0 and len(query_smarts) > 1:
                continue

        failure_reason = (
            f"no_database_template_generates_declared_reactants_after_top{top_k}"
            if require_declared_reactants
            else f"no_database_template_executes_on_product_after_top{top_k}"
        )
        return TemplateBackedMatch(
            False,
            failure_reason,
            checked_templates=checked,
            query_smarts=query_smarts[0],
            generated_reactants=last_generated,
        )

    def _verify_template(
        self,
        template: str,
        product: str,
        reactants: Sequence[str],
        max_outcomes: int,
        require_declared_reactants: bool,
    ) -> TemplateVerification:
        if require_declared_reactants:
            match = reaction_matches_reactants(
                template,
                product,
                reactants,
                max_outcomes=max_outcomes,
            )
            return TemplateVerification(
                is_valid=match.is_valid,
                reason=match.reason,
                selected_reactants=match.canonical_reactants if match.is_valid else [],
                generated_reactants=match.generated_reactants,
                reactants_match_declared=match.is_valid,
            )

        error, generated = apply_retrosynthetic_template(
            reaction_smarts=template,
            product_smiles=product,
            max_outcomes=max_outcomes,
        )
        if error:
            return TemplateVerification(False, error)
        if not generated:
            return TemplateVerification(False, "template_does_not_generate_reactants")

        declared = list(reactants or [])
        declared_match = bool(declared) and declared in generated
        selected = declared if declared_match else self._select_generated_reactants(
            product,
            generated,
        )
        if not selected:
            return TemplateVerification(
                False,
                "template_generated_only_invalid_reactants",
                generated_reactants=generated,
            )
        return TemplateVerification(
            True,
            "template_generates_declared_reactants"
            if declared_match
            else "template_generated_reactants_used",
            selected_reactants=selected,
            generated_reactants=generated,
            reactants_match_declared=declared_match,
        )

    def _select_generated_reactants(
        self,
        product_smiles: str,
        generated: Sequence[Sequence[str]],
    ) -> List[str]:
        candidates = [list(item) for item in generated if item]
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda item: self._expansion_score(
                product_smiles,
                TemplateExpansion(
                    template="",
                    reactants=list(item),
                    checked_templates=0,
                ),
            ),
            reverse=True,
        )
        return ranked[0]

    def _substructure_match_templates(
        self,
        query_reaction: str,
        max_matches: int = 50,
    ) -> List[str]:
        """Exact stage: match proposed reaction molecules to template SMARTS."""
        query_product_mols, query_reactant_mols = self._reaction_side_mols(
            query_reaction,
            parser="smiles",
        )
        if not query_product_mols or not query_reactant_mols:
            return []

        matches = []
        for template, template_product_mols, template_reactant_mols in (
            self._preprocessed_template_patterns()
        ):
            if len(template_product_mols) != len(query_product_mols):
                continue
            if len(template_reactant_mols) != len(query_reactant_mols):
                continue
            if not self._one_to_one_substructure_match(
                template_product_mols,
                query_product_mols,
            ):
                continue
            if not self._one_to_one_substructure_match(
                template_reactant_mols,
                query_reactant_mols,
            ):
                continue
            matches.append(template)
            if len(matches) >= max_matches:
                break
        return matches

    def _preprocessed_template_patterns(self):
        cached = getattr(self.db, "_template_matcher_preprocessed_patterns", None)
        if cached is not None:
            return cached

        patterns = []
        for template in self._template_list():
            product_mols, reactant_mols = self._reaction_side_mols(
                template,
                parser="smarts",
            )
            if not product_mols or not reactant_mols:
                continue
            patterns.append((template, product_mols, reactant_mols))
        self.db._template_matcher_preprocessed_patterns = patterns
        return patterns

    @classmethod
    def _reaction_side_mols(
        cls,
        reaction_text: str,
        parser: str,
    ) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
        if not reaction_text or ">>" not in reaction_text:
            return [], []
        left, right = reaction_text.split(">>", 1)
        left_parts = cls._split_reaction_side(left)
        right_parts = cls._split_reaction_side(right)
        if not left_parts or not right_parts:
            return [], []

        mol_fn = Chem.MolFromSmiles if parser == "smiles" else Chem.MolFromSmarts
        product_mols = [mol_fn(item) for item in left_parts]
        reactant_mols = [mol_fn(item) for item in right_parts]
        if any(mol is None for mol in product_mols + reactant_mols):
            return [], []
        return product_mols, reactant_mols

    @staticmethod
    def _split_reaction_side(text: str) -> List[str]:
        text = (text or "").strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return [item.strip() for item in text.split(".") if item.strip()]

    @staticmethod
    def _one_to_one_substructure_match(
        smarts_mols: Sequence[Chem.Mol],
        target_mols: Sequence[Chem.Mol],
    ) -> bool:
        for permuted_targets in permutations(target_mols, len(smarts_mols)):
            try:
                if all(
                    target.HasSubstructMatch(smarts)
                    for smarts, target in zip(smarts_mols, permuted_targets)
                ):
                    return True
            except Exception:
                continue
        return False

    def expand_product(
        self,
        product_smiles: str,
        max_routes: int = 5,
        max_templates: int = 50000,
        max_outcomes_per_template: int = 25,
    ) -> List[TemplateExpansion]:
        """Direct one-step template expansion for root fallback.

        This is used only when LLM proposals cannot be backed by the template
        database. It keeps the validation standard strict by generating
        reactants from database templates directly.
        """
        product = canonicalize_smiles(product_smiles)
        product_mol = Chem.MolFromSmiles(product or "")
        if product_mol is None:
            return []

        expansions: List[TemplateExpansion] = []
        seen_reactant_sets = set()
        checked = 0
        candidate_pool_limit = max(100, max_routes * 50)

        for template in self._template_list():
            if checked >= max_templates:
                break
            checked += 1

            reaction = self._parse_reaction(template)
            if reaction is None or reaction.GetNumReactantTemplates() != 1:
                continue

            product_template = reaction.GetReactantTemplate(0)
            try:
                if not product_mol.HasSubstructMatch(product_template):
                    continue
            except Exception:
                continue

            try:
                outcomes = reaction.RunReactants((product_mol,))
            except Exception:
                continue

            for outcome in outcomes[:max_outcomes_per_template]:
                reactants = self._canonicalize_outcome(outcome)
                if not reactants:
                    continue
                key = tuple(reactants)
                if key in seen_reactant_sets:
                    continue
                seen_reactant_sets.add(key)
                expansions.append(
                    TemplateExpansion(
                        template=template,
                        reactants=reactants,
                        checked_templates=checked,
                    )
                )
                if len(expansions) > candidate_pool_limit:
                    expansions = self._rank_expansions(product, expansions)[:candidate_pool_limit]

        return self._rank_expansions(product, expansions)[:max_routes]

    def product_applicable_templates(
        self,
        product_smiles: str,
    ) -> Tuple[List[TemplateApplicabilityHit], dict]:
        """Return templates whose product-side pattern can match the molecule.

        This scans the full template library, but uses cached product-side
        pattern fingerprints as a substructure screen before the stricter
        RDKit substructure check. It is the candidate-graph analogue of a
        template-based one-step retrosynthesis model's action-space filter.
        """
        product = canonicalize_smiles(product_smiles)
        product_mol = Chem.MolFromSmiles(product or "")
        if product_mol is None:
            return [], {
                "screened_templates": 0,
                "fingerprint_candidates": 0,
                "applicable_templates": 0,
            }

        product_cache = getattr(self.db, "_template_matcher_product_applicable", None)
        if product_cache is None:
            product_cache = {}
            self.db._template_matcher_product_applicable = product_cache
        cached = product_cache.get(product)
        if cached is not None:
            return cached

        product_fp = self._product_pattern_fingerprint(product_mol)
        templates, fingerprints = self._product_template_fingerprints()
        hits: List[TemplateApplicabilityHit] = []
        fingerprint_candidates = 0

        for index, (template, template_fp) in enumerate(
            zip(templates, fingerprints),
            start=1,
        ):
            if product_fp is not None and template_fp is not None:
                try:
                    if not DataStructs.AllProbeBitsMatch(template_fp, product_fp):
                        continue
                except Exception:
                    continue

            fingerprint_candidates += 1
            reaction = self._parse_reaction(template)
            if reaction is None or reaction.GetNumReactantTemplates() != 1:
                continue

            try:
                product_template = reaction.GetReactantTemplate(0)
                if not product_mol.HasSubstructMatch(product_template):
                    continue
            except Exception:
                continue

            hits.append(
                TemplateApplicabilityHit(
                    template=template,
                    checked_templates=index,
                    product_template_atoms=product_template.GetNumAtoms(),
                    product_template_bonds=product_template.GetNumBonds(),
                )
            )

        stats = {
            "screened_templates": len(templates),
            "fingerprint_candidates": fingerprint_candidates,
            "applicable_templates": len(hits),
        }
        product_cache[product] = (hits, stats)
        return hits, stats

    def _rank_expansions(
        self,
        product_smiles: str,
        expansions: List[TemplateExpansion],
    ) -> List[TemplateExpansion]:
        return sorted(
            expansions,
            key=lambda item: self._expansion_score(product_smiles, item),
            reverse=True,
        )

    def _expansion_score(self, product_smiles: str, expansion: TemplateExpansion) -> float:
        product_heavy = self._heavy_atom_count(product_smiles)
        reactant_heavies = [self._heavy_atom_count(item) for item in expansion.reactants]
        if not reactant_heavies:
            return -1e6

        max_reactant_heavy = max(reactant_heavies)
        purchasable = getattr(self.db, "is_purchasable", lambda _: False)
        purchasable_count = sum(1 for item in expansion.reactants if purchasable(item))
        same_as_product = any(item == product_smiles for item in expansion.reactants)

        score = 0.0
        if len(expansion.reactants) == 2:
            score += 45.0
        elif len(expansion.reactants) == 3:
            score += 15.0
        elif len(expansion.reactants) == 1:
            score -= 40.0
        else:
            score -= 25.0 * len(expansion.reactants)

        score += 8.0 * purchasable_count
        score += 0.5 * max(0, product_heavy - max_reactant_heavy)
        score += self._known_reagent_bonus(expansion.reactants)

        if len(expansion.reactants) == 1:
            score -= 25.0
        if same_as_product:
            score -= 50.0
        score -= self._unstable_fragment_penalty(expansion.reactants)
        return score

    def _build_queries(
        self,
        product: str,
        reactants: Sequence[str],
        draft_reaction_smarts: str,
    ) -> List[str]:
        queries = []
        draft = (draft_reaction_smarts or "").strip()
        if ">>" in draft:
            queries.append(draft)

        if reactants:
            canonical_query = f"{product}>>{'.'.join(sorted(reactants))}"
            if canonical_query not in queries:
                queries.append(canonical_query)

        return queries

    def _template_set(self) -> set[str]:
        templates = getattr(self.db, "template_rules", set())
        if not isinstance(templates, set):
            templates = set(templates or [])
            self.db.template_rules = templates
        return templates

    def _template_list(self) -> List[str]:
        template_list = getattr(self.db, "_template_matcher_template_list", None)
        if template_list is None:
            template_list = sorted(self._template_set())
            self.db._template_matcher_template_list = template_list
        return template_list

    def _template_fingerprints(self):
        cached = getattr(self.db, "_template_matcher_fingerprints", None)
        if cached is not None:
            return cached

        cached = self._load_fingerprint_cache()
        if cached is not None:
            self.db._template_matcher_fingerprints = cached
            return cached

        valid_templates = []
        fingerprints = []
        template_list = self._template_list()
        started = time.perf_counter()
        print(
            f"[Template][INFO] Building reaction-difference fingerprint cache: "
            f"templates={len(template_list)}"
        )
        for index, template in enumerate(template_list, start=1):
            fp = self._reaction_fingerprint(template)
            if fp is None:
                continue
            valid_templates.append(template)
            fingerprints.append(fp)
            if index % 50000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[Template][INFO] Fingerprinted {index}/{len(template_list)} "
                    f"templates in {elapsed:.1f}s; valid={len(valid_templates)}"
                )

        cached = (valid_templates, fingerprints)
        self.db._template_matcher_fingerprints = cached
        self._save_fingerprint_cache(cached)
        print(
            f"[Template][INFO] Reaction-difference fingerprint cache ready: "
            f"valid={len(valid_templates)}"
        )
        return cached

    def _fingerprint_cache_path(self) -> str:
        return getattr(default_config, "TEMPLATE_FINGERPRINT_CACHE_PATH", "")

    def _product_fingerprint_cache_path(self) -> str:
        return getattr(default_config, "TEMPLATE_PRODUCT_FINGERPRINT_CACHE_PATH", "")

    def _load_fingerprint_cache(self):
        path = self._fingerprint_cache_path()
        if not path or not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("schema_version") != 2:
                return None
            template_count = len(self._template_list())
            if payload.get("template_count") != template_count:
                print(
                    "[Template][WARN] Ignoring stale fingerprint cache: "
                    f"template_count={payload.get('template_count')} current={template_count}"
                )
                return None
            valid_templates = payload.get("valid_templates") or []
            fingerprints = payload.get("fingerprints") or []
            if len(valid_templates) != len(fingerprints):
                return None
            print(
                f"[Template][INFO] Loaded reaction-difference fingerprint cache: "
                f"valid={len(valid_templates)}, path={path}"
            )
            return valid_templates, fingerprints
        except Exception as exc:
            print(f"[Template][WARN] Failed to load fingerprint cache: {exc}")
            return None

    def _save_fingerprint_cache(self, cached) -> None:
        path = self._fingerprint_cache_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            valid_templates, fingerprints = cached
            payload = {
                "schema_version": 2,
                "fingerprint_type": "rdkit_reaction_difference",
                "template_count": len(self._template_list()),
                "valid_templates": valid_templates,
                "fingerprints": fingerprints,
            }
            with gzip.open(path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Template][INFO] Saved fingerprint cache: {path}")
        except Exception as exc:
            print(f"[Template][WARN] Failed to save fingerprint cache: {exc}")

    def _product_template_fingerprints(self):
        cached = getattr(self.db, "_template_matcher_product_fingerprints", None)
        if cached is not None:
            return cached

        cached = self._load_product_fingerprint_cache()
        if cached is not None:
            self.db._template_matcher_product_fingerprints = cached
            return cached

        valid_templates = []
        fingerprints = []
        template_list = self._template_list()
        started = time.perf_counter()
        print(
            f"[Template][INFO] Building product-side fingerprint cache: "
            f"templates={len(template_list)}"
        )
        for index, template in enumerate(template_list, start=1):
            reaction = self._parse_reaction(template)
            if reaction is None or reaction.GetNumReactantTemplates() != 1:
                continue
            try:
                product_template = reaction.GetReactantTemplate(0)
                fp = self._product_pattern_fingerprint(product_template)
            except Exception:
                continue
            if fp is None:
                continue
            valid_templates.append(template)
            fingerprints.append(fp)
            if index % 50000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[Template][INFO] Product-fingerprinted {index}/"
                    f"{len(template_list)} templates in {elapsed:.1f}s; "
                    f"valid={len(valid_templates)}"
                )

        cached = (valid_templates, fingerprints)
        self.db._template_matcher_product_fingerprints = cached
        self._save_product_fingerprint_cache(cached)
        print(
            f"[Template][INFO] Product-side fingerprint cache ready: "
            f"valid={len(valid_templates)}"
        )
        return cached

    def _load_product_fingerprint_cache(self):
        path = self._product_fingerprint_cache_path()
        if not path or not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("schema_version") != 1:
                return None
            template_count = len(self._template_list())
            if payload.get("template_count") != template_count:
                print(
                    "[Template][WARN] Ignoring stale product fingerprint cache: "
                    f"template_count={payload.get('template_count')} current={template_count}"
                )
                return None
            valid_templates = payload.get("valid_templates") or []
            fingerprints = payload.get("fingerprints") or []
            if len(valid_templates) != len(fingerprints):
                return None
            print(
                f"[Template][INFO] Loaded product-side fingerprint cache: "
                f"valid={len(valid_templates)}, path={path}"
            )
            return valid_templates, fingerprints
        except Exception as exc:
            print(f"[Template][WARN] Failed to load product fingerprint cache: {exc}")
            return None

    def _save_product_fingerprint_cache(self, cached) -> None:
        path = self._product_fingerprint_cache_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            valid_templates, fingerprints = cached
            payload = {
                "schema_version": 1,
                "template_count": len(self._template_list()),
                "valid_templates": valid_templates,
                "fingerprints": fingerprints,
            }
            with gzip.open(path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Template][INFO] Saved product fingerprint cache: {path}")
        except Exception as exc:
            print(f"[Template][WARN] Failed to save product fingerprint cache: {exc}")

    def _topk_similar_templates(self, query_smarts: str, top_k: int) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []

        query_fp = self._reaction_fingerprint(query_smarts)
        if query_fp is None:
            return []

        templates, fingerprints = self._template_fingerprints()
        if not fingerprints:
            return []

        k = min(top_k, len(fingerprints))
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
        if k >= len(similarities):
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -k)[-k:]
            indices = indices[np.argsort([similarities[idx] for idx in indices])[::-1]]

        return [(templates[int(idx)], float(similarities[int(idx)])) for idx in indices[:k]]

    def similar_templates(self, query_smarts: str, top_k: int) -> List[Tuple[str, float]]:
        """Return database templates ranked by reaction-fingerprint similarity."""
        return self._topk_similar_templates(query_smarts, top_k)

    @staticmethod
    def _reaction_fingerprint(reaction_smarts: str):
        if not reaction_smarts or ">>" not in reaction_smarts:
            return None
        reaction = TemplateMatcher._smiles_to_reaction(reaction_smarts)
        for use_smiles in (False, True):
            if reaction is not None:
                break
            try:
                with rdBase.BlockLogs():
                    reaction = rdChemReactions.ReactionFromSmarts(
                        reaction_smarts,
                        useSmiles=use_smiles,
                    )
                if reaction is not None:
                    break
            except Exception:
                reaction = None

        if reaction is None:
            return None

        try:
            return rdChemReactions.CreateDifferenceFingerprintForReaction(reaction)
        except Exception:
            return None

    @staticmethod
    def _smiles_to_reaction(reaction_smiles: str):
        try:
            left, right = reaction_smiles.split(">>", 1)
            left_mols = [Chem.MolFromSmiles(item) for item in left.split(".") if item]
            right_mols = [Chem.MolFromSmiles(item) for item in right.split(".") if item]
            if not left_mols or not right_mols:
                return None
            if any(mol is None for mol in left_mols + right_mols):
                return None
            left_smarts = ".".join(Chem.MolToSmarts(mol) for mol in left_mols)
            right_smarts = ".".join(Chem.MolToSmarts(mol) for mol in right_mols)
            return rdChemReactions.ReactionFromSmarts(f"{left_smarts}>>{right_smarts}")
        except Exception:
            return None

    @staticmethod
    def _product_pattern_fingerprint(mol: Chem.Mol):
        try:
            return Chem.PatternFingerprint(mol)
        except Exception:
            return None

    @staticmethod
    def _parse_reaction(reaction_smarts: str):
        try:
            with rdBase.BlockLogs():
                return rdChemReactions.ReactionFromSmarts(reaction_smarts, useSmiles=False)
        except Exception:
            return None

    @staticmethod
    def _canonicalize_outcome(outcome: Iterable[Chem.Mol]) -> Optional[List[str]]:
        reactants = []
        for mol in outcome:
            try:
                Chem.SanitizeMol(mol)
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                clean = canonicalize_smiles(smiles)
                if not clean:
                    return None
                reactants.append(clean)
            except Exception:
                return None
        return sorted(reactants)

    @staticmethod
    def _heavy_atom_count(smiles: str) -> int:
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()

    @staticmethod
    def _known_reagent_bonus(reactants: Sequence[str]) -> float:
        bonus = 0.0
        for smiles in reactants:
            if "P2(=S)SP(=S)" in smiles or "P12" in smiles:
                bonus += 20.0
            if smiles in {"O", "CO", "CCO", "CC(C)(C)O", "O=S(=O)(O)O"}:
                bonus += 5.0
        return bonus

    @staticmethod
    def _unstable_fragment_penalty(reactants: Sequence[str]) -> float:
        penalty = 0.0
        for smiles in reactants:
            if "[O-]" in smiles or "[N-]" in smiles or "[S-]" in smiles:
                penalty += 12.0
            if "[CH]" in smiles or "[C]" in smiles or "[NH]" in smiles:
                penalty += 8.0
            if "." in smiles:
                penalty += 5.0
        return penalty


def match_step_to_template(
    db,
    product_smiles: str,
    reactant_smiles: Sequence[str],
    draft_reaction_smarts: str = "",
    top_k: int = 100,
    max_outcomes: int = 5000,
    require_declared_reactants: bool = True,
) -> TemplateBackedMatch:
    return TemplateMatcher(db).match_step(
        product_smiles=product_smiles,
        reactant_smiles=reactant_smiles,
        draft_reaction_smarts=draft_reaction_smarts,
        top_k=top_k,
        max_outcomes=max_outcomes,
        require_declared_reactants=require_declared_reactants,
    )


def expand_product_with_templates(
    db,
    product_smiles: str,
    max_routes: int = 5,
    max_templates: int = 50000,
    max_outcomes_per_template: int = 25,
) -> List[TemplateExpansion]:
    return TemplateMatcher(db).expand_product(
        product_smiles=product_smiles,
        max_routes=max_routes,
        max_templates=max_templates,
        max_outcomes_per_template=max_outcomes_per_template,
    )
