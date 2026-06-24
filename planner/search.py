from planner.strategic_search import StrategicRetrosynthesisPlanner


class RetrosynthesisPlanner(StrategicRetrosynthesisPlanner):
    """Compatibility entry point for the current experience-guided planner."""


__all__ = ["RetrosynthesisPlanner", "StrategicRetrosynthesisPlanner"]
