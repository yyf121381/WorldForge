"""LingYaoAgent: bridge to the LingYao Decision Engine.

Requires a running LingYao instance with the planner module available.
Falls back to RuleAgent if the bridge is unavailable.
"""
from agents import Agent
from agents.rule_agent import RuleAgent
from envs.base_env import WorldState, Action


def build_state_prompt(state: WorldState, env_name: str) -> str:
    """Convert simulation state to a prompt for LingYao."""
    s = state.to_dict()
    lines = [f"你正在模拟环境「{env_name}」中。当前状态："]
    for k, v in s.items():
        if k.startswith("_") or k in ("episode", "step", "done"):
            continue
        lines.append(f"  {k}: {v:.1f}")
    lines.append("\n可选行动：empathize / guide / encourage / challenge / inform")
    lines.append("根据当前状态选一个并回复行动名称。")
    return "\n".join(lines)


class LingYaoAgent(Agent):
    """Uses the LingYao planner. Falls back to RuleAgent."""

    def __init__(self, planner_fn=None):
        self.name = "LingYaoAgent"
        self.planner_fn = planner_fn
        self.fallback = RuleAgent("LingYaoFallback")

    def act(self, state: WorldState, available: list) -> Action:
        if self.planner_fn:
            try:
                prompt = build_state_prompt(state, "emotion_depletion")
                action_type = self.planner_fn(prompt)
                if action_type in available:
                    return Action(action_type, {"source": "lingyao"})
            except Exception:
                pass
        return self.fallback.act(state, available)
