"""WorldForge · Agent interfaces

Two agents provided:
  1. RuleAgent: simple rule-based policy (no external dependencies)
  2. LingYaoAgent: wraps the real LingYao decision engine via API

The LingYaoAgent makes this a closed-loop:
  World Simulator → State → LingYao Planner → Action → World Simulator
"""
from abc import ABC, abstractmethod
from typing import Optional, List
import json
from env import WorldState, Action


class Agent(ABC):
    """Base agent that observes state and returns actions."""

    @abstractmethod
    def act(self, state: WorldState, available_actions: List[str]) -> Action:
        """Observe state, return an action."""
        pass

    def reset(self):
        """Called at start of each episode."""
        pass


class RandomAgent(Agent):
    """Baseline: randomly selects from available actions."""
    def __init__(self, name: str = "RandomAgent"):
        self.name = name
    def act(self, state: WorldState, available: List[str]) -> Action:
        import random
        return Action(random.choice(available), {"reason": "random"})


class RuleAgent(Agent):
    """Rule-based agent for local testing without LingYao.
    
    Simple heuristics for the emotion_depletion environment:
    - If energy is very low (<20), always empathize
    - If energy is low (<35), prefer empathize over guide
    - If energy is good (>40) and frustration is high, use guide
    - If willingness is very low, use encourage
    - Never challenge when energy < 30
    """

    def __init__(self, name: str = "RuleAgent"):
        self.name = name
        self.state_history = []

    def act(self, state: WorldState, available: List[str]) -> Action:
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


def build_state_prompt(state: WorldState, env_name: str) -> str:
    """Convert simulation state to a prompt for LingYao's decision engine."""
    s = state.to_dict()
    lines = [f"你正在模拟环境「{env_name}」中。当前状态："]
    for k, v in s.items():
        if k in ("episode", "step", "done", "_consecutive_empathy", "_consecutive_challenge"):
            continue
        lines.append(f"  {k}: {v:.1f}")
    lines.append("\n你可以选择以下行动之一：")
    lines.append("  1. empathize - 共情、倾听、理解对方的感受")
    lines.append("  2. guide - 给出结构化建议和行动方案")
    lines.append("  3. encourage - 给予鼓励和积极肯定")
    lines.append("  4. challenge - 提出挑战，推动对方行动")
    lines.append("  5. inform - 提供信息和知识")
    lines.append("\n根据当前状态，你应该选择哪个行动？只回复行动名称。")
    return "\n".join(lines)


class LingYaoAgent(Agent):
    """Uses the real LingYao planner to make decisions.
    
    This connects to a running LingYao server via SSH/API.
    For standalone testing, use RuleAgent instead.
    """

    def __init__(self, planner_fn=None):
        """
        Args:
            planner_fn: a callable(state_text) -> action_type
                        If None, uses RuleAgent as fallback.
        """
        self.planner_fn = planner_fn
        self.fallback = RuleAgent("LingYaoFallback")
        self.name = "LingYao"

    def act(self, state: WorldState, available: List[str]) -> Action:
        if self.planner_fn:
            try:
                prompt = build_state_prompt(state, "emotion_depletion")
                action_type = self.planner_fn(prompt)
                if action_type in available:
                    return Action(action_type, {"source": "lingyao_planner"})
            except Exception:
                pass
        # Fallback to rules
        return self.fallback.act(state, available)
