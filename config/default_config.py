from __future__ import annotations
import os
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PATH_PURCHASABLE_SET = os.path.join(DATA_DIR, "purchasable_set.pkl.gz")
PATH_ROUTES_DB = os.path.join(DATA_DIR, "routes_database.parquet")
TEMPLATE_FINGERPRINT_CACHE_PATH = os.path.join(
    DATA_DIR,
    "template_fingerprint_cache.pkl.gz",
)
TEMPLATE_PRODUCT_FINGERPRINT_CACHE_PATH = os.path.join(
    DATA_DIR,
    "template_product_fingerprint_cache.pkl.gz",
)

# ---------------------------------------------------------------------------
# LLM runtime
# ---------------------------------------------------------------------------

USE_MOCK = False
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
CRITIC_LLM_API_KEY = os.getenv("CRITIC_LLM_API_KEY", LLM_API_KEY)
CRITIC_LLM_BASE_URL = os.getenv("CRITIC_LLM_BASE_URL", LLM_BASE_URL)
CRITIC_LLM_MODEL_NAME = os.getenv("CRITIC_LLM_MODEL_NAME", LLM_MODEL_NAME)

LLM_RUNTIME: Dict[str, Any] = {
    "timeout_seconds": 90,
    "max_retries": 2,
    "retry_backoff_seconds": 8,
    "overload_backoff_seconds": 20,
    "stop_batch_on_failure": True,
    "circuit_breaker_failures": 2,
    "circuit_breaker_cooldown_seconds": 120,
    "service_recovery_attempts": 12,
    "route_proposal_workers": int(os.getenv("LLM_ROUTE_PROPOSAL_WORKERS", "3")),
}

def resolve_model_name(model: str | None = None) -> str:
    """Return the configured model id."""
    return (model or LLM_MODEL_NAME).strip()

# ---------------------------------------------------------------------------
# Paper-level experimental knobs
# ---------------------------------------------------------------------------
#
# These are the values that should normally appear in ablations/tables. The
# many old engineering flags are now fixed by the method policy below.

BUDGET = 100
ROUTE_WIDTH = 5
RAG_TOP_K = 5
MAX_SEARCH_DEPTH = 16
LLM_TEMPERATURE = 0.5
TEMPLATE_TOP_K = 100

CANDIDATE_WIDTH = 5
CANDIDATE_TEMPLATE_TOP_K = 1000

# ---------------------------------------------------------------------------
# Fixed method policy
# ---------------------------------------------------------------------------

SEARCH_POLICY: Dict[str, Any] = {
    "availability_weight": 0.4,
    "macro_update_interval": 10,
    "root_retry_attempts": 3,
    "max_accepted_prefix_steps": 3,
    "dead_end_max_expansion_failures": 999,
}

VALIDATION_POLICY: Dict[str, Any] = {
    "max_outcomes": 5000,
}

CANDIDATE_GRAPH_POLICY: Dict[str, Any] = {
    "depth": 0,
    "max_nodes": 6,
    "rag_top_k": 8,
    "direct_max_templates": 0,
    "applicable_template_limit": 2000,
    "max_outcomes": 20,
    "max_chars": 5200,
    "max_lateral_risk": 0.88,
}

EXPERIENCE_POLICY: Dict[str, Any] = {
    "retrieval_max_bad": 8,
    "max_macro_chars": 1800,
    "max_micro_chars": 1400,
    "max_context_chars": 5200,
    "top_valid_cuts": 3,
    "top_failed_cuts": 5,
    "top_taboos": 3,
    "top_macro_patterns": 5,
}

ABLATION_POLICY: Dict[str, Any] = {
    "disable_macro_experience": False,
    "disable_micro_experience": False,
    "disable_candidate_cuts": False,
    "disable_initial_macro_debate": False,
}


def configure_experiment(
    *,
    model: str | None = None,
    critic_model: str | None = None,
    budget: int | None = None,
    route_width: int | None = None,
    rag_top_k: int | None = None,
    max_search_depth: int | None = None,
    temperature: float | None = None,
    template_top_k: int | None = None,
    candidate_width: int | None = None,
    candidate_template_top_k: int | None = None,
    ablation_policy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Set the small set of reportable knobs for a paper-style experiment."""
    global LLM_MODEL_NAME, CRITIC_LLM_MODEL_NAME
    global BUDGET, ROUTE_WIDTH, RAG_TOP_K, MAX_SEARCH_DEPTH
    global LLM_TEMPERATURE, TEMPLATE_TOP_K
    global CANDIDATE_WIDTH, CANDIDATE_TEMPLATE_TOP_K
    global ABLATION_POLICY

    if model is not None:
        LLM_MODEL_NAME = resolve_model_name(model)
    if critic_model is not None:
        CRITIC_LLM_MODEL_NAME = resolve_model_name(critic_model)
    if budget is not None:
        BUDGET = int(budget)
    if route_width is not None:
        ROUTE_WIDTH = int(route_width)
    if rag_top_k is not None:
        RAG_TOP_K = int(rag_top_k)
    if max_search_depth is not None:
        MAX_SEARCH_DEPTH = int(max_search_depth)
    if temperature is not None:
        LLM_TEMPERATURE = float(temperature)
    if template_top_k is not None:
        TEMPLATE_TOP_K = int(template_top_k)
    if candidate_width is not None:
        CANDIDATE_WIDTH = int(candidate_width)
    if candidate_template_top_k is not None:
        CANDIDATE_TEMPLATE_TOP_K = int(candidate_template_top_k)
    if ablation_policy:
        ABLATION_POLICY.update(
            {key: bool(value) for key, value in ablation_policy.items()}
        )

    _sync_runtime_aliases()
    return experiment_snapshot()


def experiment_snapshot() -> Dict[str, Any]:
    """Return only the method-defining settings worth saving with results."""
    return {
        "budget": BUDGET,
        "llm_model": resolve_model_name(),
        "critic_llm_model": resolve_model_name(CRITIC_LLM_MODEL_NAME),
        "route_width": ROUTE_WIDTH,
        "rag_top_k": RAG_TOP_K,
        "max_search_depth": MAX_SEARCH_DEPTH,
        "llm_temperature": LLM_TEMPERATURE,
        "template_top_k": TEMPLATE_TOP_K,
        "candidate_width": CANDIDATE_WIDTH,
        "candidate_template_top_k": CANDIDATE_TEMPLATE_TOP_K,
        "method": {
            "search": "experience-guided tree search with strict legal-prefix insertion",
            "validation": "rdchiral database-template execution",
            "guidance": "RAG + candidate graph + structured experience",
        },
        "ablation_policy": dict(ABLATION_POLICY),
    }


def _sync_runtime_aliases() -> None:
    """Compatibility names used by the LLM client and route generator."""
    global RETRIEVAL_SIZE_NO, EXPANSION_WIDTH, MACRO_UPDATE_INTERVAL
    global LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES, LLM_RETRY_BACKOFF_SECONDS
    global LLM_OVERLOAD_BACKOFF_SECONDS, STOP_ROUTE_BATCH_ON_LLM_FAILURE
    global LLM_CIRCUIT_BREAKER_FAILURES, LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS
    global LLM_SERVICE_RECOVERY_ATTEMPTS, LLM_ROUTE_PROPOSAL_WORKERS

    RETRIEVAL_SIZE_NO = RAG_TOP_K
    EXPANSION_WIDTH = ROUTE_WIDTH
    MACRO_UPDATE_INTERVAL = SEARCH_POLICY["macro_update_interval"]

    LLM_TIMEOUT_SECONDS = LLM_RUNTIME["timeout_seconds"]
    LLM_MAX_RETRIES = LLM_RUNTIME["max_retries"]
    LLM_RETRY_BACKOFF_SECONDS = LLM_RUNTIME["retry_backoff_seconds"]
    LLM_OVERLOAD_BACKOFF_SECONDS = LLM_RUNTIME["overload_backoff_seconds"]
    STOP_ROUTE_BATCH_ON_LLM_FAILURE = LLM_RUNTIME["stop_batch_on_failure"]
    LLM_CIRCUIT_BREAKER_FAILURES = LLM_RUNTIME["circuit_breaker_failures"]
    LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS = LLM_RUNTIME[
        "circuit_breaker_cooldown_seconds"
    ]
    LLM_SERVICE_RECOVERY_ATTEMPTS = int(
        LLM_RUNTIME.get("service_recovery_attempts", 0)
    )
    LLM_ROUTE_PROPOSAL_WORKERS = max(
        1,
        int(LLM_RUNTIME.get("route_proposal_workers", 1)),
    )


_sync_runtime_aliases()
