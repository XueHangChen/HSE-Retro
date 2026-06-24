from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class ReactionStep:
    molecule_set: List[str]; rational: str; product: List[str]
    reaction: str; reactants: List[str]; updated_molecule_set: List[str]
    is_valid: bool = False; feedback: str = "未评估"

@dataclass
class SynthesisRoute:
    steps: List[ReactionStep]; explanation: str = ""
    reward: float = -float('inf'); is_successful: bool = False
    
    def get_first_invalid_step_index(self) -> int:
        for i, step in enumerate(self.steps):
            if not step.is_valid: return i
        return -1