from rdkit import Chem 
from rdkit.Chem import AllChem, DataStructs

def is_valid(smiles: str) -> bool:
    if not smiles: return False
    try: mol = Chem.MolFromSmiles(smiles); return mol is not None
    except Exception: return False

def get_morgan_fingerprint(smiles: str):
    try: mol = Chem.MolFromSmiles(smiles); return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception: return None

def get_tanimoto_similarity(fp1, fp2) -> float:
    if fp1 is None or fp2 is None: return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def canonicalize_smiles(smi):
    try: return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except Exception: return None

def get_reaction_fingerprint(reaction_smarts: str):
    # TODO: 替换为真实的反应指纹 (如 DRFP)
    return hash(reaction_smarts)