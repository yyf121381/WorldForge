"""WorldForge · Evaluation engine.

Computes standardized metrics: reward, velocity, success rates.
"""
import json, statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from metrics.velocity import compute_velocity


@dataclass
class EvalReport:
    env_name: str = ""
    agent_name: str = ""
    num_episodes: int = 0
    avg_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    std_reward: float = 0.0
    avg_steps: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    avg_velocity: float = 0.0
    sustained_velocity: float = 0.0
    velocity_trend: float = 0.0

    def to_dict(self) -> dict:
        return {
            "env": self.env_name,
            "agent": self.agent_name,
            "episodes": self.num_episodes,
            "metrics": {
                "avg_reward": self.avg_reward,
                "min_reward": self.min_reward,
                "max_reward": self.max_reward,
                "std_reward": self.std_reward,
                "avg_steps": self.avg_steps,
                "success_rate": self.success_rate,
                "failure_rate": self.failure_rate,
                "timeout_rate": self.timeout_rate,
            },
            "velocity": {
                "avg_velocity": self.avg_velocity,
                "sustained_velocity": self.sustained_velocity,
                "velocity_trend": self.velocity_trend,
            },
            "action_profile": self.action_counts,
        }

    def summary(self) -> str:
        trend = "↑" if self.velocity_trend > 0.3 else ("↓" if self.velocity_trend < -0.3 else "→")
        return (
            f"[{self.env_name}] {self.agent_name} × {self.num_episodes} episodes\n"
            f"  Reward:    avg={self.avg_reward:.2f} σ={self.std_reward:.2f} [{self.min_reward:.1f}, {self.max_reward:.1f}]\n"
            f"  Steps:     avg={self.avg_steps:.1f}\n"
            f"  Rates:     success={self.success_rate:.0%} failure={self.failure_rate:.0%} timeout={self.timeout_rate:.0%}\n"
            f"  Velocity:  avg={self.avg_velocity:+.2f} sustained={self.sustained_velocity:+.2f} trend={trend}\n"
            f"  Actions:   {self.action_counts}"
        )


def evaluate(env_name: str, agent_name: str, episode_data_list: List[dict]) -> EvalReport:
    """Tabulate episode results into a standardized EvalReport."""
    n = len(episode_data_list)
    rewards = [d["total_reward"] for d in episode_data_list]
    lengths = [d["steps_taken"] for d in episode_data_list]

    successes = sum(1 for d in episode_data_list
                    if d.get("final_state", {}).get("willingness", 0) > 70
                    and d.get("final_state", {}).get("energy", 0) > 50
                    and d.get("final_state", {}).get("frustration", 0) < 30)
    failures = sum(1 for d in episode_data_list
                   if d.get("final_state", {}).get("energy", 0) <= 0)
    timeouts = n - successes - failures

    action_counts = {}
    for d in episode_data_list:
        for step in d.get("steps", []):
            at = step.get("action", "?")
            action_counts[at] = action_counts.get(at, 0) + 1

    all_v = [compute_velocity(d.get("steps", [])) for d in episode_data_list]

    return EvalReport(
        env_name=env_name, agent_name=agent_name, num_episodes=n,
        avg_reward=round(statistics.mean(rewards), 2),
        min_reward=round(min(rewards), 2),
        max_reward=round(max(rewards), 2),
        std_reward=round(statistics.stdev(rewards), 2) if n > 1 else 0.0,
        avg_steps=round(statistics.mean(lengths), 1),
        success_rate=successes / n, failure_rate=failures / n, timeout_rate=timeouts / n,
        action_counts=action_counts,
        avg_velocity=round(statistics.mean([v["avg"] for v in all_v]), 2) if all_v else 0.0,
        sustained_velocity=round(statistics.mean([v["sustained"] for v in all_v]), 2) if all_v else 0.0,
        velocity_trend=round(statistics.mean([v["trend"] for v in all_v]), 2) if all_v else 0.0,
    )
