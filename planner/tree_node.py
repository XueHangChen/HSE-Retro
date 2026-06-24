import math
from typing import List, Optional

class ORNode:
    def __init__(self, smiles: str, is_purchasable: bool):
        self.smiles = smiles
        self.is_purchasable = is_purchasable
        self.is_solved = is_purchasable
        
        self.children: List['ANDNode'] = []  
        self.parents: List['ANDNode'] = []   
        
        self.visit_count = 0
        self.value = 0.0  
        self.micro_experience = ""
        self.structured_experience = {}
        self.expansion_failures = 0
        self.backoff_until_iteration = 0
        self.backoff_count = 0
        self.is_dead_end = False
        self.dead_end_reason = ""

    def update_solved_status(self):
        if self.is_purchasable:
            self.is_solved = True
            return
        original_status = self.is_solved
        self.is_solved = any(child.is_solved for child in self.children)
        if self.is_solved != original_status:
            for parent in self.parents:
                parent.update_solved_status()

class ANDNode:
    def __init__(self, reaction_smarts: str, product: 'ORNode', reactants: List['ORNode'], depth: int):
        self.reaction_smarts = reaction_smarts
        self.product = product      
        self.children = reactants   
        self.depth = depth
        self.is_valid = True
        self.validation_feedback = ""
        self.is_solved = False
        
        self.visit_count = 0
        self.value = 0.0    
        self.reward = 0.0   
        self.generation_source = ""
        self.candidate_id = ""
        self.frontier_metrics = {}
        self.reward_components = {}

    def update_solved_status(self):
        original_status = self.is_solved
        if not self.children:
            self.is_solved = False 
        else:
            self.is_solved = self.is_valid and all(child.is_solved for child in self.children)
        
        if self.is_solved != original_status:
            self.product.update_solved_status()

    def ucb_score(self, c_param: float = 1.4) -> float:
        n_parent = self.product.visit_count
        if n_parent <= 1: n_parent = 2 
        
        exploitation = self.value
        n_self = max(1, self.visit_count)
        exploration = c_param * math.sqrt(math.log(n_parent) / n_self)
        
        return exploitation + exploration
