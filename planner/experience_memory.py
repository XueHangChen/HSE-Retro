from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence

from chem_utils.reaction import canonicalize_smiles


SCHEMA_VERSION = "structured_experience_v1"


def _canon(smiles: str) -> str:
    clean = canonicalize_smiles(str(smiles or "").strip())
    return clean or str(smiles or "").strip()


def _split_reactants(text: str) -> List[str]:
    if not text or text.upper() == "UNKNOWN":
        return []
    return sorted(_canon(item) for item in text.split(".") if item.strip())


def _shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def parse_observation_text(text: str) -> Dict:
    """Parse the compact experience strings produced by validation reports."""
    text = str(text or "").strip()
    if not text:
        return {}

    if text.startswith("LLM-NO-VALID-PREFIX"):
        product = _extract(r"Product=([^;|]+)", text)
        return {
            "status": "NO_VALID_PREFIX",
            "product": _canon(product),
            "reaction": "",
            "reactants": [],
            "reason": "llm_no_valid_prefix",
            "source": "llm_pathway",
            "raw": text,
        }

    if text.startswith("MEMORY-GATE rejected:"):
        product = _extract(r"Product=([^|]+)", text)
        reactants_text = _extract(r"Reactants=([^|]+)", text)
        reason = _extract(r"Reason=([^|]+)", text, "memory_gate_rejected")
        source = _extract(r"Source=([^|]+)", text, "memory_gate")
        return {
            "status": "MEMORY_GATE",
            "product": _canon(product),
            "reaction": "",
            "reactants": _split_reactants(reactants_text),
            "reason": reason,
            "source": source,
            "raw": text,
        }

    if text.startswith("PRUNED route suffix:"):
        product = _extract(r"Product=([^;|]+)", text)
        reason = _extract(r"Reason=([^;|]+)", text, "route_pruned")
        source = _extract(r"Source=([^;|]+)", text, "unknown")
        return {
            "status": "PRUNED",
            "product": _canon(product),
            "reaction": "",
            "reactants": [],
            "reason": reason,
            "source": source,
            "raw": text,
        }

    status = "VALID" if text.startswith("VALID") else "INVALID"
    if text.startswith("LEGAL but"):
        status = "WEAK_VALID"

    product = _extract(r"Product=([^|]+)", text)
    reaction = _extract(r"Reaction=([^|]+)", text)
    reactants_text = _extract(r"Reactants=([^|]+)", text)
    reason = _extract(r"Reason=([^|]+)", text)
    source = _extract(r"Source=([^|]+)", text, "unknown")
    similarity_text = _extract(r"Similarity=([0-9.]+)", text)
    score_text = _extract(r"Score=([0-9.]+)", text)

    try:
        similarity = float(similarity_text) if similarity_text else 0.0
    except ValueError:
        similarity = 0.0
    try:
        score = float(score_text) if score_text else 0.0
    except ValueError:
        score = 0.0

    return {
        "status": status,
        "product": _canon(product),
        "reaction": reaction if reaction and reaction != "NONE" else "",
        "reactants": _split_reactants(reactants_text),
        "reason": reason,
        "source": source,
        "similarity": similarity,
        "score": score,
        "raw": text,
    }


def _cut_key(product: str, reactants: Sequence[str], reaction: str = "", reason: str = "") -> str:
    return json.dumps(
        {
            "product": _canon(product),
            "reactants": sorted(_canon(item) for item in reactants if item),
            "reaction": reaction or "",
            "reason": reason or "",
        },
        sort_keys=True,
    )


class StructuredExperienceMemory:
    """Compact, evidence-backed memory used to steer later route generation."""

    def __init__(
        self,
        max_valid_cuts: int = 3,
        max_failed_cuts: int = 5,
        max_taboos: int = 3,
        max_macro_patterns: int = 5,
    ) -> None:
        self.max_valid_cuts = max_valid_cuts
        self.max_failed_cuts = max_failed_cuts
        self.max_taboos = max_taboos
        self.max_macro_patterns = max_macro_patterns
        self.micro: Dict[str, Dict] = {}
        self.global_valid: Dict[str, Dict] = {}
        self.global_failures: Dict[str, Dict] = {}
        self.failure_reason_counts: Counter = Counter()
        self.events: List[Dict] = []

    def ensure_micro(self, molecule: str) -> Dict:
        molecule = _canon(molecule)
        if molecule not in self.micro:
            self.micro[molecule] = {
                "schema_version": SCHEMA_VERSION,
                "scope": "micro",
                "molecule": molecule,
                "node_stats": {},
                "viable_cuts": {},
                "failed_cuts": {},
                "avoid_reactant_sets": [],
                "local_taboos": [],
                "next_generation_constraints": [],
            }
        return self.micro[molecule]

    def record_valid_cut(
        self,
        product: str,
        reactants: Sequence[str],
        reaction: str,
        reward: float = 0.0,
        match_source: str = "",
        similarity: float = 0.0,
        source: str = "",
        directive: str = "",
    ) -> None:
        product = _canon(product)
        reactants = sorted(_canon(item) for item in reactants if item)
        memory = self.ensure_micro(product)
        key = _cut_key(product, reactants, reaction)
        entry = memory["viable_cuts"].setdefault(
            key,
            {
                "product": product,
                "reactants": reactants,
                "reaction_smarts": reaction,
                "match_source": match_source or source or "unknown",
                "template_similarity": similarity,
                "reward_sum": 0.0,
                "best_reward": 0.0,
                "count": 0,
                "directive": directive,
            },
        )
        entry["count"] += 1
        entry["reward_sum"] += float(reward or 0.0)
        entry["best_reward"] = max(float(entry.get("best_reward", 0.0)), float(reward or 0.0))
        if match_source:
            entry["match_source"] = match_source
        if similarity:
            entry["template_similarity"] = max(float(entry.get("template_similarity", 0.0)), similarity)
        if not entry.get("directive"):
            entry["directive"] = (
                "Prefer this Product -> Reactants cut when it reduces the "
                "non-stock frontier and does not regenerate an ancestor."
            )

        if key not in self.global_valid:
            self.global_valid[key] = deepcopy(entry)
        else:
            global_entry = self.global_valid[key]
            global_entry["count"] = global_entry.get("count", 0) + 1
            global_entry["reward_sum"] = global_entry.get("reward_sum", 0.0) + float(reward or 0.0)
            global_entry["best_reward"] = max(
                float(global_entry.get("best_reward", 0.0)),
                float(reward or 0.0),
            )

    def record_failed_cut(
        self,
        product: str,
        reactants: Sequence[str],
        reaction: str,
        reason: str,
        source: str = "",
        directive: str = "",
    ) -> None:
        product = _canon(product)
        reactants = sorted(_canon(item) for item in reactants if item)
        reason = reason or "unknown_failure"
        memory = self.ensure_micro(product)
        key = _cut_key(product, reactants, "", reason)
        entry = memory["failed_cuts"].setdefault(
            key,
            {
                "product": product,
                "reactants": reactants,
                "reaction_smarts": reaction,
                "failure_reason": reason,
                "source": source or "unknown",
                "count": 0,
                "directive": directive or self._failure_directive(reason),
            },
        )
        entry["count"] += 1
        self.failure_reason_counts[reason] += 1
        self._update_micro_directives(memory)

        if key not in self.global_failures:
            self.global_failures[key] = deepcopy(entry)
        else:
            global_entry = self.global_failures[key]
            global_entry["count"] = global_entry.get("count", 0) + 1

    def record_observation_text(self, text: str) -> None:
        parsed = parse_observation_text(text)
        if not parsed:
            return
        status = parsed.get("status")
        if status == "VALID":
            self.record_valid_cut(
                product=parsed.get("product", ""),
                reactants=parsed.get("reactants", []),
                reaction=parsed.get("reaction", ""),
                reward=parsed.get("score", 0.0),
                match_source=parsed.get("source", ""),
                similarity=parsed.get("similarity", 0.0),
            )
        else:
            self.record_failed_cut(
                product=parsed.get("product", ""),
                reactants=parsed.get("reactants", []),
                reaction=parsed.get("reaction", ""),
                reason=parsed.get("reason", "") or status.lower(),
                source=parsed.get("source", ""),
            )

    def record_observation_texts(self, texts: Iterable[str]) -> None:
        for text in texts or []:
            self.record_observation_text(text)

    def record_no_valid_prefix(self, product: str, attempt: int) -> None:
        reason = "llm_no_valid_prefix"
        directive = (
            "No proposed first step survived exact/top-100 template validation; "
            "switch to a different first-step disconnection family and avoid "
            "all repeated failed reactant sets."
        )
        self.record_failed_cut(
            product=product,
            reactants=[],
            reaction="",
            reason=reason,
            directive=directive,
        )
        memory = self.ensure_micro(product)
        stats = memory.setdefault("node_stats", {})
        stats["no_valid_prefix_count"] = int(stats.get("no_valid_prefix_count", 0)) + 1
        stats["last_no_valid_prefix_attempt"] = attempt

    def update_node_stats(self, molecule: str, stats: Dict) -> None:
        memory = self.ensure_micro(molecule)
        memory["node_stats"].update(stats or {})
        self._update_micro_directives(memory)

    def note_event(self, event: Dict, limit: int = 120) -> None:
        self.events.append(event)
        if len(self.events) > limit:
            del self.events[: len(self.events) - limit]

    def micro_snapshot(self, molecule: str, compact: bool = False) -> Dict:
        memory = deepcopy(self.ensure_micro(molecule))
        self._update_micro_directives(memory)
        memory["viable_cuts"] = self._top_valid(memory.get("viable_cuts", {}).values())
        memory["failed_cuts"] = self._top_failed(memory.get("failed_cuts", {}).values())
        if compact:
            return {
                "molecule": memory["molecule"],
                "node_stats": memory.get("node_stats", {}),
                "preferred_cuts": memory["viable_cuts"][: self.max_valid_cuts],
                "avoid_reactant_sets": memory.get("avoid_reactant_sets", [])[: self.max_failed_cuts],
                "local_taboos": memory.get("local_taboos", [])[: self.max_taboos],
                "next_generation_constraints": memory.get("next_generation_constraints", [])[: self.max_taboos + 2],
            }
        return memory

    def macro_snapshot(self, search_stats: Optional[Dict] = None) -> Dict:
        failure_counts = [
            {"failure_reason": reason, "count": count}
            for reason, count in self.failure_reason_counts.most_common(8)
        ]
        valid_patterns = self._top_valid(self.global_valid.values(), self.max_macro_patterns)
        global_taboos = []
        for item in failure_counts[: self.max_taboos]:
            global_taboos.append(
                {
                    "failure_type": item["failure_reason"],
                    "support_count": item["count"],
                    "directive": self._failure_directive(item["failure_reason"]),
                }
            )
        bottlenecks = []
        for molecule, memory in self.micro.items():
            stats = memory.get("node_stats", {})
            failed_count = sum(item.get("count", 0) for item in memory.get("failed_cuts", {}).values())
            if stats.get("children_count", 0) == 0 and failed_count:
                bottlenecks.append(
                    {
                        "molecule": molecule,
                        "failed_cut_count": failed_count,
                        "visit_count": stats.get("visit_count", 0),
                        "dominant_failure_reason": self._dominant_micro_reason(memory),
                        "directive": (
                            "Treat this molecule as a local bottleneck; avoid repeated "
                            "failed reactant sets and try a template-supported different cut."
                        ),
                    }
                )
        bottlenecks = sorted(
            bottlenecks,
            key=lambda item: (item.get("failed_cut_count", 0), item.get("visit_count", 0)),
            reverse=True,
        )[: self.max_taboos]
        directives = [
            "Do not regenerate molecules already present in the current ancestor path.",
            "Do not repeat failed reactant sets listed in current_node_memory.avoid_reactant_sets.",
            "Prefer first steps that are exact/top100 template-backed, close to retrieved route precedents, or present in the candidate graph.",
        ]
        for taboo in global_taboos:
            directive = taboo.get("directive")
            if directive and directive not in directives:
                directives.append(directive)
        return {
            "schema_version": SCHEMA_VERSION,
            "scope": "macro",
            "search_stats": dict(search_stats or {}),
            "validated_patterns": valid_patterns,
            "global_taboos": global_taboos,
            "bottleneck_classes": bottlenecks,
            "generation_directives": directives[:6],
        }

    def render_prompt_context(
        self,
        target_smiles: str,
        search_stats: Optional[Dict] = None,
        max_chars: int = 5200,
        include_macro: bool = True,
    ) -> str:
        return self.render_prompt_context_compact(
            target_smiles=target_smiles,
            search_stats=search_stats,
            max_chars=max_chars,
            include_macro=include_macro,
        )

    def render_prompt_context_compact(
        self,
        target_smiles: str,
        search_stats: Optional[Dict] = None,
        max_chars: int = 5200,
        include_macro: bool = True,
    ) -> str:
        """Render prioritized JSON memory for prompt injection.

        The full memory snapshot is still saved for analysis through
        `snapshot_all`; this view is intentionally compact and action-oriented.
        """
        micro = self.micro_snapshot(target_smiles, compact=True)
        macro = self.macro_snapshot(search_stats) if include_macro else {}

        hard_constraints = []
        for reactants in micro.get("avoid_reactant_sets", [])[:3]:
            if reactants:
                hard_constraints.append(
                    "FORBIDDEN_REACTANTS: " + " + ".join(reactants)
                )
        for taboo in micro.get("local_taboos", [])[:2]:
            directive = taboo.get("directive", "")
            if directive:
                hard_constraints.append("LOCAL_TABOO: " + directive)
        if include_macro:
            for taboo in macro.get("global_taboos", [])[:2]:
                directive = taboo.get("directive", "")
                if directive:
                    hard_constraints.append("GLOBAL_TABOO: " + directive)
        for directive in micro.get("next_generation_constraints", [])[:3]:
            if directive:
                hard_constraints.append(directive)

        preferred_local = [
            self._compact_cut(cut, include_product=False)
            for cut in micro.get("preferred_cuts", [])[:3]
        ]
        macro_patterns = []
        if include_macro:
            macro_patterns = [
                self._compact_cut(cut, include_product=True)
                for cut in macro.get("validated_patterns", [])[:3]
            ]

        stats = dict(micro.get("node_stats", {}) or {})
        compact_stats = {
            key: stats.get(key)
            for key in (
                "visit_count",
                "children_count",
                "is_solved",
                "is_purchasable",
                "best_value",
                "expansion_failures",
                "no_valid_prefix_count",
            )
            if key in stats
        }

        payload = {
            "must_obey": self._dedupe_strings(hard_constraints)[:8],
            "prefer_local_cuts": preferred_local,
            "prefer_global_patterns": macro_patterns,
            "macro_guidance": (
                macro.get("generation_directives", [])[:4] if include_macro else []
            ),
            "current_node": {
                "molecule": micro.get("molecule", _canon(target_smiles)),
                "stats": compact_stats,
            },
        }
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text

        # Drop lower-priority global pattern detail before final truncation.
        payload["prefer_global_patterns"] = payload["prefer_global_patterns"][:1]
        payload["macro_guidance"] = payload["macro_guidance"][:2]
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
        fallback = {
            "must_obey": payload["must_obey"][:4],
            "prefer_local_cuts": [
                {
                    "reactants": cut.get("reactants", []),
                    "best_reward": cut.get("best_reward", 0.0),
                }
                for cut in payload["prefer_local_cuts"][:2]
            ],
            "current_node": payload["current_node"],
            "truncated": True,
        }
        text = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
        return _shorten(text, max_chars)

    def snapshot_all(self, search_stats: Optional[Dict] = None) -> Dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "macro": self.macro_snapshot(search_stats),
            "micro": [
                self.micro_snapshot(molecule)
                for molecule in sorted(self.micro)
            ],
            "recent_events": list(self.events[-40:]),
        }

    def _top_valid(self, entries: Iterable[Dict], limit: Optional[int] = None) -> List[Dict]:
        limit = self.max_valid_cuts if limit is None else limit
        rows = []
        for item in entries:
            row = deepcopy(item)
            count = max(1, int(row.get("count", 0)))
            row["avg_reward"] = float(row.get("reward_sum", 0.0)) / count
            row.pop("reward_sum", None)
            rows.append(row)
        return sorted(
            rows,
            key=lambda item: (
                item.get("best_reward", 0.0),
                item.get("avg_reward", 0.0),
                item.get("count", 0),
            ),
            reverse=True,
        )[:limit]

    def _top_failed(self, entries: Iterable[Dict], limit: Optional[int] = None) -> List[Dict]:
        limit = self.max_failed_cuts if limit is None else limit
        rows = [deepcopy(item) for item in entries]
        return sorted(rows, key=lambda item: item.get("count", 0), reverse=True)[:limit]

    @staticmethod
    def _dedupe_strings(items: Iterable[str]) -> List[str]:
        seen = set()
        output = []
        for item in items:
            clean = _shorten(item, 180)
            if clean and clean not in seen:
                seen.add(clean)
                output.append(clean)
        return output

    @staticmethod
    def _compact_cut(cut: Dict, include_product: bool = False) -> Dict:
        output = {
            "reactants": cut.get("reactants", []),
            "reaction_smarts": _shorten(cut.get("reaction_smarts", ""), 160),
            "match_source": cut.get("match_source", "unknown"),
            "best_reward": round(float(cut.get("best_reward", 0.0) or 0.0), 3),
            "count": int(cut.get("count", 0) or 0),
        }
        if include_product:
            output["product"] = cut.get("product", "")
        directive = cut.get("directive", "")
        if directive:
            output["directive"] = _shorten(directive, 140)
        return output

    def _update_micro_directives(self, memory: Dict) -> None:
        failed = self._top_failed(memory.get("failed_cuts", {}).values())
        avoid = []
        for item in failed:
            reactants = item.get("reactants", [])
            if reactants and reactants not in avoid:
                avoid.append(reactants)
        memory["avoid_reactant_sets"] = avoid[: self.max_failed_cuts]

        reason_counts = Counter()
        for item in memory.get("failed_cuts", {}).values():
            reason_counts[item.get("failure_reason", "unknown_failure")] += item.get("count", 0)

        taboos = []
        for reason, count in reason_counts.most_common(self.max_taboos):
            taboos.append(
                {
                    "failure_type": reason,
                    "evidence_count": count,
                    "directive": self._failure_directive(reason),
                }
            )
        memory["local_taboos"] = taboos

        constraints = [
            "Do not output any reactant set listed in avoid_reactant_sets.",
            "Do not regenerate ancestor molecules from the current path.",
            "Make the first step exact/top100 template-backed or close to a retrieved route precedent/candidate-graph cut.",
        ]
        for taboo in taboos:
            directive = taboo.get("directive")
            if directive and directive not in constraints:
                constraints.append(directive)
        memory["next_generation_constraints"] = constraints[: self.max_taboos + 3]

    def _dominant_micro_reason(self, memory: Dict) -> str:
        counts = Counter()
        for item in memory.get("failed_cuts", {}).values():
            counts[item.get("failure_reason", "unknown_failure")] += item.get("count", 0)
        return counts.most_common(1)[0][0] if counts else ""

    @staticmethod
    def _failure_directive(reason: str) -> str:
        reason = reason or "unknown_failure"
        if reason.startswith("no_database_template"):
            return (
                "Avoid this unsupported Product -> Reactants set; switch to a "
                "retrieved/template-backed first step."
            )
        if reason.startswith("memory_avoid_reactant_set"):
            return "Do not repeat this reactant set; it is already listed in avoid_reactant_sets."
        if reason.startswith("memory_self_reactant"):
            return "Do not propose the product molecule itself as a reactant."
        if reason.startswith("memory_ancestor_reactant"):
            return "Do not regenerate an ancestor molecule from the current route path."
        if reason.startswith("cycle_to") or "cycle" in reason:
            return "Never propose a reactant already present in the current ancestor path."
        if reason.startswith("same_molecule"):
            return "Do not propose the product itself as a reactant."
        if reason.startswith("invalid_or_empty"):
            return "Do not output UNKNOWN, empty, or unparsable reactant lists."
        if reason == "llm_no_valid_prefix":
            return (
                "The recent proposal batch had no valid prefix; choose a different "
                "first-step disconnection family."
            )
        if reason.startswith("max_depth"):
            return "Avoid extending this branch with deeper steps; prefer cuts closer to stock molecules."
        return f"Avoid repeating cuts that triggered validation failure: {reason}."
