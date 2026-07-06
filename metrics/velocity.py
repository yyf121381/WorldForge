"""Velocity metric computation.

Computes directional improvement velocity from step state data.
Environment-agnostic: detects known state schemas automatically.
"""
from typing import List, Dict


def velocity_from_state(prev: Dict, curr: Dict) -> float:
    """Compute velocity between two consecutive states.
    
    emotion_depletion: velocity = ΔE - ΔF + ΔW
    meaning_crisis:    velocity = ΔM - ΔC + ΔO
    fallback:         mean of all numeric deltas
    """
    if all(k in prev for k in ["energy", "frustration", "willingness"]):
        return (curr.get("energy", 0) - prev.get("energy", 0)
                - (curr.get("frustration", 0) - prev.get("frustration", 0))
                + (curr.get("willingness", 0) - prev.get("willingness", 0)))

    if all(k in prev for k in ["meaning", "confusion", "openness"]):
        return (curr.get("meaning", 0) - prev.get("meaning", 0)
                - (curr.get("confusion", 0) - prev.get("confusion", 0))
                + (curr.get("openness", 0) - prev.get("openness", 0)))

    deltas = []
    for k in set(prev.keys()) & set(curr.keys()):
        if isinstance(prev[k], (int, float)):
            deltas.append(curr[k] - prev[k])
    return sum(deltas) / len(deltas) if deltas else 0.0


def compute_velocity(steps: List[dict]) -> dict:
    """Compute avg, sustained, and trend velocity from step data."""
    if len(steps) < 2:
        return {"avg": 0.0, "sustained": 0.0, "trend": 0.0}

    velocities = []
    for i in range(1, len(steps)):
        prev = steps[i - 1].get("state_before", {})
        curr = steps[i].get("state_before", {})
        if prev and curr:
            velocities.append(velocity_from_state(prev, curr))

    if not velocities:
        return {"avg": 0.0, "sustained": 0.0, "trend": 0.0}

    import statistics
    avg = statistics.mean(velocities)
    last_n = min(5, len(velocities))
    sustained = statistics.mean(velocities[-last_n:]) if last_n > 0 else 0.0

    if len(velocities) >= 3:
        mid = len(velocities) // 2
        trend = (statistics.mean(velocities[mid:]) - statistics.mean(velocities[:mid])) if mid > 0 else 0.0
    else:
        trend = 0.0

    return {"avg": round(avg, 2), "sustained": round(sustained, 2), "trend": round(trend, 2)}
