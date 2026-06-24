import gzip
import os
import pickle

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from config import default_config


RETRO_DATA_DIR = os.path.join(default_config.DATA_DIR, "retro_data")
ONE_STEP_MODEL_DIR = os.path.join(RETRO_DATA_DIR, "one_step_model")
TEMPLATE_FILE_PATH = os.path.join(ONE_STEP_MODEL_DIR, "template_rules_1.dat")

PURCHASABLE_DB_PATH = default_config.PATH_PURCHASABLE_SET
ROUTES_DB_PATH = default_config.PATH_ROUTES_DB


class SynthesisDatabase:
    """Database interface for templates, purchasable molecules, and route RAG."""

    def __init__(self):
        print("[Data][INFO] Initializing synthesis database.")
        self.template_rules = self._load_template_rules(TEMPLATE_FILE_PATH)
        self.purchasable_set = self._load_purchasable_set(PURCHASABLE_DB_PATH)
        self.route_db, self.route_fingerprints = self._load_routes_db(ROUTES_DB_PATH)

    def _load_template_rules(self, file_path):
        print(f"[Data][INFO] Loading reaction templates: {file_path}")
        try:
            template_set = set()
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    template = line.strip()
                    if template:
                        template_set.add(template)
            print(f"[Data][INFO] Loaded {len(template_set)} reaction templates.")
            return template_set
        except FileNotFoundError:
            print(f"[Data][ERROR] Reaction template file not found: {file_path}")
            return set()
        except Exception as exc:
            print(f"[Data][ERROR] Failed to load reaction templates: {exc}")
            return set()

    def _load_purchasable_set(self, file_path):
        if not os.path.exists(file_path):
            print(f"[Data][ERROR] Purchasable molecule file not found: {file_path}")
            print("[Data][WARN] Purchasable checks will return False.")
            return set()

        print(f"[Data][INFO] Loading purchasable molecules: {file_path}")
        try:
            with gzip.open(file_path, "rb") as handle:
                purchasable_set = pickle.load(handle)
            print(f"[Data][INFO] Loaded {len(purchasable_set)} purchasable molecules.")
            return purchasable_set
        except Exception as exc:
            print(f"[Data][ERROR] Failed to load purchasable molecules: {exc}")
            return set()

    def _load_routes_db(self, file_path):
        if not os.path.exists(file_path):
            print(f"[Data][ERROR] Route RAG database not found: {file_path}")
            print("[Data][WARN] Route retrieval will run without external examples.")
            return None, []

        print(f"[Data][INFO] Loading route RAG database: {file_path}")
        try:
            route_db = pd.read_parquet(file_path)
            print(f"[Data][INFO] Converting {len(route_db)} route fingerprints.")
            fingerprints = [
                DataStructs.ExplicitBitVect(fp_bytes)
                for fp_bytes in route_db["fingerprint"]
            ]
            print(f"[Data][INFO] Loaded {len(route_db)} route examples for RAG.")
            return route_db, fingerprints
        except Exception as exc:
            print(f"[Data][ERROR] Failed to load route RAG database: {exc}")
            return None, []

    def match_reaction(self, llm_reaction_template: str) -> str | None:
        if llm_reaction_template in self.template_rules:
            return llm_reaction_template
        return None

    def is_purchasable(self, smiles: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            return canonical_smi in self.purchasable_set
        except Exception:
            return False

    def retrieve_similar_routes(self, target_smiles: str, k: int) -> list:
        if self.route_db is None:
            return []

        try:
            mol = Chem.MolFromSmiles(target_smiles)
            if mol is None:
                return []
            target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            similarities = DataStructs.BulkTanimotoSimilarity(
                target_fp,
                self.route_fingerprints,
            )

            import numpy as np

            k = min(k, len(similarities))
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            return [self.route_db.iloc[idx]["route_text"] for idx in top_k_indices]
        except Exception as exc:
            print(f"[Data][WARN] Route retrieval failed: {exc}")
            return []


print("[Data][INFO] Creating global synthesis database instance.")
db_instance = SynthesisDatabase()
