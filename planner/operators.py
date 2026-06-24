import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from llm.client import get_deepseek_client, get_llm_client
from llm.parser import parse_llm_output
from llm.prompts import (
    INITIALIZATION_PROMPT,
    ROUTE_SYSTEM_PROMPT,
    INITIAL_MACRO_PROMPT,
    INITIAL_MACRO_CRITIC_PROMPT,
    INITIAL_MACRO_DEFENDER_PROMPT,
    INITIAL_MACRO_JUDGE_PROMPT,
    MICRO_EXPERIENCE_UPDATE_PROMPT,
    MACRO_GENERATOR_PROMPT, 
    MACRO_CRITIC_PROMPT, 
    MACRO_DEFENDER_PROMPT,  
    MACRO_JUDGE_PROMPT      
)
from data.database import db_instance 
from config import default_config
from planner.route import SynthesisRoute
from rdkit import Chem 


MICRO_EXPERIENCE_REQUIRED_FIELDS = {
    "schema_version",
    "scope",
    "molecule",
    "node_stats",
    "viable_cuts",
    "failed_cuts",
    "avoid_reactant_sets",
    "local_taboos",
    "next_generation_constraints",
}
MICRO_EXPERIENCE_SCHEMA_VERSION = "structured_experience_v1"


def canonicalize_smiles(smi: str) -> str:
    """Canonicalize a SMILES string."""
    if not smi:
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return smi
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smi

def generate_pathways(
    target_molecule: str,
    n_routes: int,
    experience: str = "",
    context: str = "",
    frontier_context: str = "",
    candidate_graph_context: str = "",
    enable_rag: bool = True,
) -> List[SynthesisRoute]:
    """Generate LLM route proposals with RAG, experience, and frontier context."""
    clean_target = canonicalize_smiles(target_molecule)

    if enable_rag:
        try:
            retrieved_routes = db_instance.retrieve_similar_routes(
                clean_target, k=default_config.RETRIEVAL_SIZE_NO
            )
            retrieved_routes_str = "\n".join(retrieved_routes)
        except Exception as exc:
            print(f"[RAG][WARN] Route retrieval failed: {exc}")
            retrieved_routes_str = "None"
    else:
        retrieved_routes_str = (
            "No external templates retrieved. Please rely on your chemical "
            "knowledge and the provided experience."
        )

    exp_text = experience if experience.strip() else "No specific experience yet."
    if frontier_context.strip():
        exp_text = (
            f"[Current Route Frontier]\n{frontier_context}\n\n"
            f"[Experience & Guidelines]\n{exp_text}"
        )
    if context.strip():
        exp_text = (
            f"[Current Synthesis Context (Path from root)]\n{context}\n\n"
            f"[Experience & Guidelines]\n{exp_text}"
        )

    prompt = INITIALIZATION_PROMPT.format(
        target_molecule=clean_target,
        retrieved_routes_str=retrieved_routes_str,
        experience_str=exp_text,
        candidate_graph_context=(
            candidate_graph_context.strip()
            or "No template-grounded candidate graph was provided."
        ),
    )

    generated_routes = []
    print(f"[LLM][INFO] Requesting {n_routes} route proposal(s).")

    llm_client = get_llm_client()
    worker_count = min(
        max(1, int(getattr(default_config, "LLM_ROUTE_PROPOSAL_WORKERS", 1))),
        max(1, int(n_routes)),
    )

    def query_one(index: int) -> Tuple[int, str]:
        response_text = llm_client.query(
            prompt,
            temperature=default_config.LLM_TEMPERATURE,
            system_prompt=ROUTE_SYSTEM_PROMPT,
        )
        return index, response_text

    if worker_count > 1 and n_routes > 1:
        responses: List[Tuple[int, str]] = []
        print(f"[LLM][INFO] Proposal calls running with {worker_count} worker(s).")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(query_one, index) for index in range(n_routes)]
            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as exc:
                    print(f"[LLM][WARN] Proposal worker failed: {exc}")
        responses.sort(key=lambda item: item[0])
    else:
        responses = [query_one(index) for index in range(n_routes)]

    for _, response in responses:
        if "LLM API Call Failed" in response:
            if getattr(default_config, "STOP_ROUTE_BATCH_ON_LLM_FAILURE", True):
                print("[LLM][WARN] Proposal batch had API failure; keeping successful proposals.")
            continue
        route = parse_llm_output(response)
        if route and route.steps:
            generated_routes.append(route)

    return generated_routes


def generate_initial_macro_experience(
    target_molecule: str,
    candidate_graph_context: str = "",
    enable_rag: bool = True,
    use_debate: bool = True,
) -> str:
    """Create target-level macro strategy before root route proposals."""
    clean_target = canonicalize_smiles(target_molecule)

    if enable_rag:
        try:
            retrieved_routes = db_instance.retrieve_similar_routes(
                clean_target, k=default_config.RETRIEVAL_SIZE_NO
            )
            retrieved_routes_str = "\n".join(retrieved_routes)
        except Exception as exc:
            print(f"[RAG][WARN] Initial macro route retrieval failed: {exc}")
            retrieved_routes_str = "None"
    else:
        retrieved_routes_str = "RAG disabled."

    prompt = INITIAL_MACRO_PROMPT.format(
        target_molecule=clean_target,
        retrieved_routes_str=retrieved_routes_str,
        candidate_graph_context=(
            candidate_graph_context.strip()
            or "No template-grounded candidate graph was provided."
        ),
    )

    main_client = get_llm_client()

    print("[Experience][INFO] Initial macro agent 1 drafting strategy.")
    draft = main_client.query(prompt, temperature=0.3).strip()
    if not use_debate:
        max_chars = int(
            getattr(default_config, "EXPERIENCE_POLICY", {}).get("max_macro_chars", 1800)
        )
        if len(draft) > max_chars:
            draft = draft[: max_chars - 40].rstrip() + "\n... [truncated]"
        return draft

    critic_client = get_deepseek_client()

    print("[Experience][INFO] Initial macro agent 2 critiquing strategy.")
    critique_prompt = INITIAL_MACRO_CRITIC_PROMPT.format(
        target_molecule=clean_target,
        retrieved_routes_str=retrieved_routes_str,
        candidate_graph_context=(
            candidate_graph_context.strip()
            or "No template-grounded candidate graph was provided."
        ),
        draft_macro_experience=draft,
    )
    critique = critic_client.query(critique_prompt, temperature=0.1).strip()

    print("[Experience][INFO] Initial macro agent 1 revising strategy.")
    defender_prompt = INITIAL_MACRO_DEFENDER_PROMPT.format(
        target_molecule=clean_target,
        draft_macro_experience=draft,
        critique=critique,
    )
    revised = main_client.query(defender_prompt, temperature=0.2).strip()

    print("[Experience][INFO] Initial macro agent 3 judging final strategy.")
    judge_prompt = INITIAL_MACRO_JUDGE_PROMPT.format(
        target_molecule=clean_target,
        draft_macro_experience=draft,
        critique=critique,
        revised_macro_experience=revised,
    )
    response = main_client.query(judge_prompt, temperature=0.1).strip()
    max_chars = int(
        getattr(default_config, "EXPERIENCE_POLICY", {}).get("max_macro_chars", 1800)
    )
    if len(response) > max_chars:
        response = response[: max_chars - 40].rstrip() + "\n... [truncated]"
    return response


def _extract_json_object(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _coerce_list(value) -> list:
    return value if isinstance(value, list) else []


def _validate_micro_experience_json(response: str, current_exp: str) -> str:
    """Validate optional LLM-produced micro memory and fall back safely."""
    try:
        parsed = json.loads(_extract_json_object(response))
    except json.JSONDecodeError:
        print("[Experience][WARN] Invalid micro JSON; keeping previous memory.")
        return current_exp

    if not isinstance(parsed, dict):
        print("[Experience][WARN] Micro memory is not a JSON object; keeping previous memory.")
        return current_exp

    missing = MICRO_EXPERIENCE_REQUIRED_FIELDS.difference(parsed)
    if missing:
        print(
            "[Experience][WARN] Micro JSON missing fields "
            f"{sorted(missing)}; keeping previous memory."
        )
        return current_exp

    parsed["schema_version"] = MICRO_EXPERIENCE_SCHEMA_VERSION
    parsed["scope"] = "micro"
    parsed["node_stats"] = parsed.get("node_stats") if isinstance(parsed.get("node_stats"), dict) else {}
    parsed["viable_cuts"] = _coerce_list(parsed.get("viable_cuts"))[:3]
    parsed["failed_cuts"] = _coerce_list(parsed.get("failed_cuts"))[:5]
    parsed["avoid_reactant_sets"] = _coerce_list(parsed.get("avoid_reactant_sets"))[:5]
    parsed["local_taboos"] = _coerce_list(parsed.get("local_taboos"))[:3]
    parsed["next_generation_constraints"] = _coerce_list(
        parsed.get("next_generation_constraints")
    )[:5]
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _is_micro_experience_prompt(prompt_template: str) -> bool:
    return (
        prompt_template is MICRO_EXPERIENCE_UPDATE_PROMPT
        or '"scope": "micro"' in str(prompt_template)
        or "structured_experience_v1" in str(prompt_template)
    )

def evolve_experience(
    current_exp: str, 
    good_examples: List[str], 
    bad_examples: List[str], 
    current_batch: List[str] = None,
    prompt_template: str = None,
    use_debate: bool = False
) -> str:
    
    good_str = "\n".join(good_examples) if good_examples else "None"
    bad_str = "\n".join(bad_examples) if bad_examples else "None"
    batch_str = "\n".join(current_batch) if current_batch else "None"

    try:
        from llm.client import get_llm_client, get_deepseek_client
        
        main_client = get_llm_client()
        critic_client = get_deepseek_client()

        if not use_debate:
            if prompt_template is None:
                prompt_template = MICRO_EXPERIENCE_UPDATE_PROMPT
                
            prompt = prompt_template.format(
                old_experience=current_exp if current_exp else "None",
                current_batch_str=batch_str,
                good_examples=good_str,
                bad_examples=bad_str
            )
            response = main_client.query(prompt, temperature=0.2).strip()
            if _is_micro_experience_prompt(prompt_template):
                return _validate_micro_experience_json(response, current_exp)
            return response

        print("[Experience][INFO] Macro agent 1 drafting update.")
        draft_prompt = MACRO_GENERATOR_PROMPT.format(
            old_experience=current_exp if current_exp else "None",
            good_examples=good_str,
            bad_examples=bad_str
        )
        draft_experience = main_client.query(draft_prompt, temperature=0.4).strip()
        
        print("[Experience][INFO] Macro agent 2 reviewing chemical logic.")
        critic_prompt = MACRO_CRITIC_PROMPT.format(
            draft_experience=draft_experience,
            good_examples=good_str,
            bad_examples=bad_str
        )
        critique = critic_client.query(critic_prompt, temperature=0.1).strip()
        
        print("[Experience][INFO] Macro agent 1 revising after critique.")
        defender_prompt = MACRO_DEFENDER_PROMPT.format(
            draft_experience=draft_experience,
            critique=critique
        )
        rebuttal = main_client.query(defender_prompt, temperature=0.3).strip()
        
        print("[Experience][INFO] Macro agent 3 integrating final update.")
        judge_prompt = MACRO_JUDGE_PROMPT.format(
            draft_experience=draft_experience,
            critique=critique,
            rebuttal=rebuttal
        )
        final_experience = main_client.query(judge_prompt, temperature=0.1).strip()
        
        return final_experience

    except Exception as e:
        print(f"[Experience][WARN] Experience evolution failed: {e}")
        return current_exp
