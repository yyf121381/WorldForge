"""RuleAgent: heuristic policy for emotion_depletion environment.

Strategy:
  - If energy is critically low (<20): empathize
  - If energy is low (<35) and frustration is high (>50): empathize
  - If energy is adequate (>40) and frustration is high (>60): guide
  - If willingness is very low (<20) and energy > 25: encourage
  - If energy is strong (>50) and willingness > 40: challenge
  - Otherwise: empathize (safe default)
"""
from agents import Agent
from envs.base_env import WorldState, Action


class RuleAgent(Agent):
    def __init__(self, name: str = "RuleAgent"):
        self.name = name

    def act(self, state: WorldState, available: list) -> Action:
        s = state.to_dict()
        energy = s.get("energy", 50)
        frustration = s.get("frustration", 50)
        willingness = s.get("willingness", 50)

        if energy < 20:
            return Action("empathize", {"reason": "energy critically low"})
        elif energy < 35 and frustration > 50:
            return Action("empathize", {"reason": "low energy, high frustration"})
        elif energy > 40 and frustration > 60:
            return Action("guide", {"reason": "stable enough for guidance"})
        elif willingness < 20 and energy > 25:
            return Action("encourage", {"reason": "needs motivation boost"})
        elif energy > 50 and willingness > 40:
            return Action("challenge", {"reason": "strong enough to be challenged"})
        elif energy > 35:
            return Action("guide", {"reason": "default guidance"})
        else:
            return Action("empathize", {"reason": "safe default"})
