"""WorldForge · Evaluation module

Computes standardized metrics including velocity-based trajectory analysis.
"""
from typing import Dict, List, Optional
import json, statistics
from dataclasses import dataclass, field


@dataclass
class EvalReport:
    """Standardized evaluation report for one (env, agent) combination."""
    env_name: str
    agent_name: str
    num_episodes: int
    
    # Core metrics
    avg_reward: float
    min_reward: float
    max_reward: float
    std_reward: float
    
    avg_steps: float
    success_rate: float
    failure_rate: float
    timeout_rate: float
    
    # Action profile
    action_counts: Dict[str, int] = field(default_factory=dict)
    
    # Velocity metrics (trajectory quality)
    avg_velocity: float = 0.0           # mean per-step velocity across all episodes
    sustained_velocity: float = 0.0     # mean velocity over last 5 steps (late-stage push)
    velocity_trend: float = 0.0         # +1 = improving, -1 = decaying, 0 = flat
    
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
        return (
            f"[{self.env_name}] {self.agent_name} × {self.num_episodes} episodes\n"
            f"  Reward:    avg={self.avg_reward:.2f} σ={self.std_reward:.2f} [{self.min_reward:.1f}, {self.max_reward:.1f}]\n"
            f"  Steps:     avg={self.avg_steps:.1f}\n"
            f"  Rates:     success={self.success_rate:.0%} failure={self.failure_rate:.0%} timeout={self.timeout_rate:.0%}\n"
            f"  Velocity:  avg={self.avg_velocity:+.2f} sustained={self.sustained_velocity:+.2f} trend={'↑' if self.velocity_trend>0.3 else '↓' if self.velocity_trend<-0.3 else '→'}\n"
            f"  Actions:   {self.action_counts}"
        )


def _velocity_from_state(prev: dict, curr: dict) -> float:
    """Compute velocity between two states. Works with any state schema.
    
    For each environment type, uses appropriate key mappings:
      emotion_depletion:  velocity = ΔE - ΔF + ΔW
      meaning_crisis:     velocity = ΔM - ΔC + ΔO
    
    Falls back to mean of all numeric changes if no known pattern matches.
    """
    # Try known environment patterns
    if all(k in prev or k in curr for k in ["energy", "frustration", "willingness"]):
        de = curr.get("energy", 0) - prev.get("energy", 0)
        df = curr.get("frustration", 0) - prev.get("frustration", 0)
        dw = curr.get("willingness", 0) - prev.get("willingness", 0)
        return de - df + dw
    
    if all(k in prev or k in curr for k in ["meaning", "confusion", "openness"]):
        dm = curr.get("meaning", 0) - prev.get("meaning", 0)
        dc = curr.get("confusion", 0) - prev.get("confusion", 0)
        do = curr.get("openness", 0) - prev.get("openness", 0)
        return dm - dc + do
    
    # Fallback: mean delta of all common numeric keys
    deltas = []
    for k in set(prev.keys()) & set(curr.keys()):
        if isinstance(prev[k], (int, float)) and isinstance(curr[k], (int, float)):
            deltas.append(curr[k] - prev[k])
    return sum(deltas) / len(deltas) if deltas else 0.0


def _compute_velocity(steps: List[dict]) -> dict:
    if len(steps) < 2:
        return {"avg": 0.0, "sustained": 0.0, "trend": 0.0}
    
    velocities = []
    for i in range(1, len(steps)):
        prev = steps[i-1].get("state_before", {})
        curr = steps[i].get("state_before", {})
        if not prev or not curr:
            continue
        velocities.append(_velocity_from_state(prev, curr))
    
    if not velocities:
        return {"avg": 0.0, "sustained": 0.0, "trend": 0.0}
    
    avg_v = statistics.mean(velocities)
    
    # Sustained velocity: average over last 5 steps (or all if < 5)
    last_n = min(5, len(velocities))
    sustained_v = statistics.mean(velocities[-last_n:]) if last_n > 0 else 0.0
    
    # Trend: slope of velocity over time (linear approximation)
    if len(velocities) >= 3:
        # Simple trend: first half vs second half
        mid = len(velocities) // 2
        first_half = statistics.mean(velocities[:mid]) if mid > 0 else 0
        second_half = statistics.mean(velocities[mid:]) if mid < len(velocities) else 0
        trend = second_half - first_half
    else:
        trend = 0.0
    
    return {"avg": round(avg_v, 2), "sustained": round(sustained_v, 2), "trend": round(trend, 2)}


def compute_report(env_name: str, agent_name: str, episode_data_list: List[dict]) -> EvalReport:
    """Tabulate episode results into a standardized EvalReport with velocity metrics."""
    n = len(episode_data_list)
    rewards = [d["total_reward"] for d in episode_data_list]
    lengths = [d["steps_taken"] for d in episode_data_list]
    
    # Classify outcomes
    successes = sum(1 for d in episode_data_list
                    if d.get("final_state", {}).get("willingness", 0) > 70
                    and d.get("final_state", {}).get("energy", 0) > 50
                    and d.get("final_state", {}).get("frustration", 0) < 30)
    failures = sum(1 for d in episode_data_list
                   if d.get("final_state", {}).get("energy", 0) <= 0)
    timeouts = n - successes - failures
    
    # Action profile
    action_counts = {}
    for d in episode_data_list:
        for step in d.get("steps", []):
            at = step.get("action", "?")
            action_counts[at] = action_counts.get(at, 0) + 1
    
    # Velocity: aggregate across all episodes
    all_vel = []
    all_sus = []
    all_tr = []
    for d in episode_data_list:
        v = _compute_velocity(d.get("steps", []))
        all_vel.append(v["avg"])
        all_sus.append(v["sustained"])
        all_tr.append(v["trend"])
    
    avg_v = round(statistics.mean(all_vel), 2) if all_vel else 0.0
    avg_sus = round(statistics.mean(all_sus), 2) if all_sus else 0.0
    avg_tr = round(statistics.mean(all_tr), 2) if all_tr else 0.0
    
    return EvalReport(
        env_name=env_name,
        agent_name=agent_name,
        num_episodes=n,
        avg_reward=round(statistics.mean(rewards), 2),
        min_reward=round(min(rewards), 2),
        max_reward=round(max(rewards), 2),
        std_reward=round(statistics.stdev(rewards), 2) if n > 1 else 0.0,
        avg_steps=round(statistics.mean(lengths), 1),
        success_rate=successes / n,
        failure_rate=failures / n,
        timeout_rate=timeouts / n,
        action_counts=action_counts,
        avg_velocity=avg_v,
        sustained_velocity=avg_sus,
        velocity_trend=avg_tr,
    )
