"""WorldForge · Evaluation engine.

Three-layer success hierarchy:
  1. Survival success — state never collapsed (energy > 0)
  2. Stability success — sustained positive velocity (late-stage improvement)
  3. Recovery success — reached full recovery threshold
"""
import statistics
from typing import Dict, List
from dataclasses import dataclass, field
from metrics.velocity import compute_velocity


# Success thresholds (for emotion_depletion schema)
SURVIVAL_OK = {"energy_min": 20, "frustration_max": 80}
RECOVERY_OK = {"energy_min": 50, "frustration_max": 30, "willingness_min": 70}


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

    # Three-layer success
    survival_rate: float = 0.0    # kept state stable
    stability_rate: float = 0.0   # sustained positive velocity
    recovery_rate: float = 0.0    # reached recovery threshold

    failure_rate: float = 0.0
    timeout_rate: float = 0.0

    action_counts: Dict[str, int] = field(default_factory=dict)
    avg_velocity: float = 0.0
    sustained_velocity: float = 0.0
    velocity_trend: float = 0.0

    def to_dict(self) -> dict:
        return {
            "env": self.env_name, "agent": self.agent_name, "episodes": self.num_episodes,
            "metrics": {
                "avg_reward": self.avg_reward, "min_reward": self.min_reward,
                "max_reward": self.max_reward, "std_reward": self.std_reward,
                "avg_steps": self.avg_steps,
            },
            "success": {
                "survival_rate": self.survival_rate,
                "stability_rate": self.stability_rate,
                "recovery_rate": self.recovery_rate,
            },
            "failure": {
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
        tr = "↑" if self.velocity_trend > 0.3 else ("↓" if self.velocity_trend < -0.3 else "→")
        return (
            f"[{self.env_name}] {self.agent_name} × {self.num_episodes} episodes\n"
            f"  Reward:    avg={self.avg_reward:.2f} σ={self.std_reward:.2f} [{self.min_reward:.1f}, {self.max_reward:.1f}]\n"
            f"  Steps:     avg={self.avg_steps:.1f}\n"
            f"  Success:   survival={self.survival_rate:.0%}  stability={self.stability_rate:.0%}  recovery={self.recovery_rate:.0%}\n"
            f"  Failure:   collapse={self.failure_rate:.0%}  timeout={self.timeout_rate:.0%}\n"
            f"  Velocity:  avg={self.avg_velocity:+.2f}  sustained={self.sustained_velocity:+.2f}  trend={tr}\n"
            f"  Actions:   {self.action_counts}"
        )


def _check_survival(fs: dict) -> bool:
    """Survival success: state didn't collapse."""
    if fs.get("energy", 0) <= 0:
        return False
    if fs.get("energy", 50) >= SURVIVAL_OK["energy_min"]:
        return True
    return fs.get("frustration", 0) <= SURVIVAL_OK["frustration_max"]


def _check_recovery(fs: dict) -> bool:
    """Recovery success: reached full recovery threshold."""
    return (fs.get("energy", 0) >= RECOVERY_OK["energy_min"]
            and fs.get("frustration", 0) <= RECOVERY_OK["frustration_max"]
            and fs.get("willingness", 0) >= RECOVERY_OK["willingness_min"])


def _check_stability(velocity_result: dict) -> bool:
    """Stability success: sustained positive velocity (last 5 steps)."""
    return velocity_result.get("sustained", 0) > 3.0


def evaluate(env_name: str, agent_name: str, episode_data_list: List[dict]) -> EvalReport:
    """Tabulate episode results into a standardized EvalReport."""
    n = len(episode_data_list)
    rewards = [d["total_reward"] for d in episode_data_list]
    lengths = [d["steps_taken"] for d in episode_data_list]

    # Three-layer success
    survivors = 0
    stable = 0
    recovered = 0
    collapsed = 0

    for d in episode_data_list:
        fs = d.get("final_state", {})
        v = compute_velocity(d.get("steps", []))

        if _check_recovery(fs):
            recovered += 1
            survivors += 1
            stable += 1
        elif _check_survival(fs):
            survivors += 1
            if _check_stability(v):
                stable += 1
        else:
            collapsed += 1

    timeouts = n - survivors - collapsed
    # Ensure no negative timeouts
    if timeouts < 0:
        timeouts = 0

    # Action profile
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
        survival_rate=survivors / n,
        stability_rate=stable / n,
        recovery_rate=recovered / n,
        failure_rate=collapsed / n,
        timeout_rate=timeouts / n,
        action_counts=action_counts,
        avg_velocity=round(statistics.mean([v["avg"] for v in all_v]), 2) if all_v else 0.0,
        sustained_velocity=round(statistics.mean([v["sustained"] for v in all_v]), 2) if all_v else 0.0,
        velocity_trend=round(statistics.mean([v["trend"] for v in all_v]), 2) if all_v else 0.0,
    )
