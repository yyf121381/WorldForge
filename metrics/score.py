"""WorldForge Score System — unified cross-env agent evaluation.

WS = α * reward_score + β * survival_score + γ * stability_score - δ * collapse_score
"""
import statistics, json
from typing import Dict, List, Tuple

ENV_CLASS = {"emotion_depletion": "A", "meaning_crisis": "A", "fast_decay": "C"}
ENV_BOUNDS = {
    "emotion_depletion": {"best": 90, "worst": -60, "max_steps": 25},
    "meaning_crisis": {"best": 80, "worst": -80, "max_steps": 25},
    "fast_decay": {"best": 0, "worst": -30, "max_steps": 25},
}

DEFAULT_WEIGHTS = (0.35, 0.25, 0.20, 0.20)  # α, β, γ, δ

AGENT_DATA = {
    "LingYao_V3": {
        "emotion_depletion": {"rewards": [73,80,83,83,84,71,81], "lengths": [25]*7},
        "meaning_crisis": {"rewards": [68,63,51,60,78,62], "lengths": [25]*6},
    },
    "RuleAgent": {
        "emotion_depletion": {"rewards": [63]*10, "lengths": [20]*10},
        "meaning_crisis": {"rewards": [-45]*10, "lengths": [3.6]*10},
    },
    "Qwen_ZeroShot": {
        "emotion_depletion": {"rewards": [46,44,38,37,40], "lengths": [16,16,13,13,13]},
        "meaning_crisis": {"rewards": [-47,76,-95,68,-42], "lengths": [9,19,24,16,6]},
    },
    "Random": {
        "emotion_depletion": {"rewards": [-38]*10, "lengths": [4.6]*10},
        "meaning_crisis": {"rewards": [-15]*10, "lengths": [11]*10},
    },
}


def compute_ws(rewards, lengths, env_name, weights=None):
    w = weights or DEFAULT_WEIGHTS
    a, b, g, d = w
    n = len(rewards)
    if n == 0:
        return {"ws": 0.0}
    avg_r = statistics.mean(rewards)
    avg_len = statistics.mean(lengths)
    std_r = statistics.stdev(rewards) if n > 1 else 0
    bounds = ENV_BOUNDS.get(env_name, {"best": 50, "worst": -50, "max_steps": 25})

    reward_score = max(0, min(1, (avg_r - bounds["worst"]) / (bounds["best"] - bounds["worst"])))
    survival_score = min(1.0, avg_len / bounds["max_steps"])
    stability = max(0.0, 1.0 - (std_r / ((bounds["best"] - bounds["worst"]) * 0.5)))
    collapse = sum(1 for r in rewards if r < -10) / n

    ws = a * reward_score + b * survival_score + g * stability - d * collapse
    return {"ws": round(ws, 3), "reward_score": round(reward_score, 3),
            "survival_score": round(survival_score, 3), "stability": round(stability, 3),
            "collapse_score": round(collapse, 3)}


def build_leaderboard(weights=None):
    w = weights or DEFAULT_WEIGHTS
    rows = []
    for agent_name, envs in AGENT_DATA.items():
        row = {"agent": agent_name}
        total = 0
        for env_name, data in envs.items():
            s = compute_ws(data["rewards"], data["lengths"], env_name, weights)
            row[env_name] = s["ws"]
            total += s["ws"]
        row["total_ws"] = round(total, 3)
        rows.append(row)
    rows.sort(key=lambda r: r["total_ws"], reverse=True)
    return rows


def print_leaderboard(rows):
    print(f"\n{'='*60}")
    print("WorldForge Leaderboard")
    print('='*60)
    print(f"  {'Agent':<20} {'Emotion':>7} {'Meaning':>7} {'Total':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        print(f"  {r['agent']:<20} {r.get('emotion_depletion',0):>7.3f} {r.get('meaning_crisis',0):>7.3f} {r['total_ws']:>7.3f}")
    print()


def sensitivity_test():
    tests = [
        ("Default",      0.35, 0.25, 0.20, 0.20),
        ("Reward+10%",   0.385, 0.25, 0.20, 0.20),
        ("Reward-10%",   0.315, 0.25, 0.20, 0.20),
        ("Survival+10%", 0.35, 0.275, 0.20, 0.20),
        ("Survival-10%", 0.35, 0.225, 0.20, 0.20),
        ("Stability+10%", 0.35, 0.25, 0.22, 0.20),
        ("Collapse+10%", 0.35, 0.25, 0.20, 0.22),
        ("Reward-heavy",  0.60, 0.15, 0.15, 0.10),
        ("Survival-heavy", 0.20, 0.50, 0.15, 0.15),
        ("Equal",         0.25, 0.25, 0.25, 0.25),
    ]
    print(f"\n{'='*80}")
    print("Weight Sensitivity Test")
    print('='*80)
    print(f"  {'Variation':<18} {'αβγδ':<20} {'LY_V3':>7} {'Rule':>7} {'Random':>7} {'1st':>6}")
    print(f"  {'-'*18} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for name, a, b, g, d in tests:
        rows = build_leaderboard((a, b, g, d))
        w = {r['agent']: r['total_ws'] for r in rows}
        print(f"  {name:<18} {a:.2f}/{b:.2f}/{g:.2f}/{d:.2f}  {w.get('LingYao_V3',0):>7.3f} {w.get('RuleAgent',0):>7.3f} {w.get('Random',0):>7.3f}  {rows[0]['agent'][:6]:>6}")


if __name__ == "__main__":
    rows = build_leaderboard()
    print_leaderboard(rows)
    sensitivity_test()
