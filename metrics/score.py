"""WorldForge Score System — unified cross-env agent evaluation.

WS = α * reward_score + β * survival_score + γ * stability_score - δ * collapse_score
"""
import statistics, json
from typing import Dict, List

# Weights (configurable, defaults for A-class environments)
ALPHA = 0.35   # reward quality
BETA = 0.25    # survival duration
GAMMA = 0.20   # stability (low variance)
DELTA = 0.20   # collapse penalty

# Environment classification
ENV_CLASS = {
    "emotion_depletion": "A",
    "meaning_crisis": "A",
    "fast_decay": "C",
}

# Theoretical bounds for normalization
ENV_BOUNDS = {
    "emotion_depletion": {"best": 90, "worst": -60, "max_steps": 25},
    "meaning_crisis": {"best": 80, "worst": -80, "max_steps": 25},
    "fast_decay": {"best": 0, "worst": -30, "max_steps": 25},
}


def normalize_reward(avg_reward: float, env_name: str) -> float:
    """Normalize reward to 0-1 scale based on env bounds."""
    b = ENV_BOUNDS.get(env_name, {"best": 50, "worst": -50})
    r = (avg_reward - b["worst"]) / (b["best"] - b["worst"])
    return max(0.0, min(1.0, r))


def compute_ws(rewards: List[float], lengths: List[int], env_name: str) -> dict:
    """Compute WorldForge Score from episode data."""
    n = len(rewards)
    if n == 0:
        return {"ws": 0.0, "reward_score": 0, "survival_score": 0, "stability": 0, "collapse": 0}
    
    avg_r = statistics.mean(rewards)
    avg_len = statistics.mean(lengths)
    max_steps = ENV_BOUNDS.get(env_name, {}).get("max_steps", 25)
    std_r = statistics.stdev(rewards) if n > 1 else 0
    
    # Reward score: normalized avg reward
    reward_score = normalize_reward(avg_r, env_name)
    
    # Survival score: how many steps relative to max
    survival_score = min(1.0, avg_len / max_steps)
    
    # Stability: lower std = more stable (normalized against expected range)
    expected_range = (ENV_BOUNDS.get(env_name, {}).get("best", 50) - 
                      ENV_BOUNDS.get(env_name, {}).get("worst", -50))
    stability = max(0.0, 1.0 - (std_r / (expected_range * 0.5)))
    
    # Collapse: fraction of episodes that went negative
    collapsed = sum(1 for r in rewards if r < -10)
    collapse_score = collapsed / n
    
    # Final WS
    ws = (ALPHA * reward_score + BETA * survival_score + 
          GAMMA * stability - DELTA * collapse_score)
    
    return {
        "ws": round(ws, 3),
        "reward_score": round(reward_score, 3),
        "survival_score": round(survival_score, 3),
        "stability": round(stability, 3),
        "collapse_score": round(collapse_score, 3),
        "env_class": ENV_CLASS.get(env_name, "?"),
    }


# ===== Reference data from experiments =====
AGENT_DATA = {
    "Random": {
        "emotion_depletion": {"rewards": [-38]*10, "lengths": [4.6]*10},
        "meaning_crisis": {"rewards": [-15]*10, "lengths": [11]*10},
    },
    "RuleAgent": {
        "emotion_depletion": {"rewards": [63]*10, "lengths": [20]*10},
        "meaning_crisis": {"rewards": [-45]*10, "lengths": [3.6]*10},
    },
    "LingYao_V3": {
        "emotion_depletion": {"rewards": [73,80,83,83,84,71,81], "lengths": [25]*7},
        "meaning_crisis": {"rewards": [68,63,51,60,78,62], "lengths": [25]*6},
    },
}


def build_leaderboard() -> List[dict]:
    """Generate cross-env leaderboard."""
    rows = []
    for agent_name, envs in AGENT_DATA.items():
        total_ws = 0
        row = {"agent": agent_name}
        for env_name, data in envs.items():
            scores = compute_ws(data["rewards"], data["lengths"], env_name)
            row[env_name] = scores["ws"]
            row[f"{env_name}_class"] = scores["env_class"]
            total_ws += scores["ws"]
        row["total_ws"] = round(total_ws, 3)
        rows.append(row)
    
    # Sort by total WS
    rows.sort(key=lambda r: r["total_ws"], reverse=True)
    return rows


def print_leaderboard(rows: List[dict]):
    """Print the leaderboard as a formatted table."""
    print("\n" + "=" * 70)
    print("WorldForge Leaderboard (Cross-Env Score)")
    print("=" * 70)
    print(f"  {'Agent':<20} {'Emotion':>8} {'Meaning':>8} {'Total':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for row in rows:
        em = row.get("emotion_depletion", 0)
        mc = row.get("meaning_crisis", 0)
        print(f"  {row['agent']:<20} {em:>7.3f} {mc:>7.3f} {row['total_ws']:>7.3f}")
    print()


if __name__ == "__main__":
    rows = build_leaderboard()
    print_leaderboard(rows)
