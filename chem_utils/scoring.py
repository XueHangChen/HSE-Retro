import os
import sys

from rdkit import Chem


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from scscore.standalone_model_numpy import SCScorer
except ImportError as exc:
    print(f"[Scoring][ERROR] Failed to import SCScorer: {exc}")
    SCScorer = None
except Exception as exc:
    print(f"[Scoring][ERROR] Unexpected SCScorer import error: {exc}")
    SCScorer = None


SC_SCORE_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "full_reaxys_model_1024bool",
)
SC_SCORE_MODEL_PATH = os.path.join(
    SC_SCORE_MODEL_DIR,
    "model.ckpt-10654.as_numpy.json.gz",
)


global_scorer = None
if SCScorer is not None:
    try:
        if not os.path.exists(SC_SCORE_MODEL_PATH):
            raise FileNotFoundError(f"SC-Score model not found: {SC_SCORE_MODEL_PATH}")

        global_scorer = SCScorer()
        global_scorer.restore(weight_path=SC_SCORE_MODEL_PATH)
        print(f"[Scoring][INFO] Loaded SC-Score model: {SC_SCORE_MODEL_PATH}")
    except Exception as exc:
        print(f"[Scoring][ERROR] Failed to load SC-Score model: {exc}")
        global_scorer = None
else:
    print("[Scoring][WARN] SCScorer unavailable; using fallback score.")


def get_sc_score(smiles: str) -> float:
    """Return the SC-Score for a valid SMILES string."""
    if global_scorer is None:
        print("[Scoring][WARN] SC-Score model is not loaded; returning fallback 5.0.")
        return 5.0

    if not smiles:
        return 10.0

    try:
        if not Chem.MolFromSmiles(smiles):
            return 10.0

        _, score = global_scorer.get_score_from_smi(smiles)
        if float(score) < 0.1:
            return 10.0
        return float(score)
    except Exception as exc:
        print(f"[Scoring][WARN] SC-Score failed for {smiles}: {exc}")
        return 5.0
