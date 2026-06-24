from planner.route import SynthesisRoute
from planner.evaluation import evaluator_instance
from chem_utils import scoring
from data.database import db_instance


def compute_reward(route: SynthesisRoute, target_molecule: str) -> SynthesisRoute:
    # 1. 评估路线 (这将更新 route.is_successful)
    route = evaluator_instance.evaluate_route(route, target_molecule)
    
    # 2. 计算奖励
    if route.is_successful:
        route.reward = 0.0 
        return route

    invalid_step_index = route.get_first_invalid_step_index()
    
    # 获取需要计算 SC Score 的分子集合
    if invalid_step_index == -1:
        # 路径没有显式错误（Valid），但可能未完成（Open ends）
        if not route.steps: 
            route.reward = -100.0 # 空路径惩罚
            return route
        molecules_at_invalid_step = route.steps[-1].updated_molecule_set
    else:
        # 路径在某一步断了
        invalid_step = route.steps[invalid_step_index]
        molecules_at_invalid_step = invalid_step.molecule_set
    
    total_sc_score = 0.0
    has_non_purchasable = False
    
    if molecules_at_invalid_step:
        for mol_smi in molecules_at_invalid_step:
            if not db_instance.is_purchasable(mol_smi):
                score = scoring.get_sc_score(mol_smi)
                total_sc_score += score
                has_non_purchasable = True
    
    # 3. 最终奖励计算逻辑
    if total_sc_score == 0.0 and not has_non_purchasable:
        route.reward = -10.0 
    else:
        route.reward = -total_sc_score
        
    return route
