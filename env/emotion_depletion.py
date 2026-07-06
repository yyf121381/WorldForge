"""WorldForge · Environment 01: Emotion Depletion

Simulates a person experiencing emotional exhaustion/burnout.
The agent must choose how to respond to help them recover.

Reward: triple-objective (survival + recovery + smoothness)
  R = surv + rec - inst

  surv = -α*max(0, F-80) - β*max(0, 20-E)    # don't crash
  rec  = λ1*ΔE - λ2*ΔF + λ3*ΔS + λ4*ΔW         # directional improvement  
  inst = δ*(|ΔE|+|ΔF|+|ΔS|+|ΔW|)               # no wild swings
"""
from dataclasses import dataclass
from typing import List, Optional
import random
from . import WorldState, Action, StepResult, Environment


@dataclass
class EmotionState(WorldState):
    energy: float = 50.0
    frustration: float = 40.0
    social_support: float = 30.0
    willingness: float = 40.0
    _consecutive_empathy: int = 0
    _consecutive_challenge: int = 0


class EmotionDepletionEnv(Environment):
    """A person is emotionally exhausted. Agent decides how to respond."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__("emotion_depletion", config)
        self.state: Optional[EmotionState] = None

    def reset(self) -> EmotionState:
        self.state = EmotionState(
            episode=len(self.history) + 1,
            step=0,
            energy=random.uniform(10, 30),
            frustration=random.uniform(40, 70),
            social_support=random.uniform(10, 40),
            willingness=random.uniform(10, 35),
        )
        self.history = []
        return self.state.clone()

    def step(self, action: Action) -> StepResult:
        s = self.state
        s.step += 1

        # === Natural decay ===
        s.energy -= random.uniform(2, 5)
        s.frustration += random.uniform(0, 3)
        s.willingness -= random.uniform(0, 2)

        # === Compute action deltas ===
        de = df = ds = dw = 0.0

        if action.type == "empathize":
            de = random.uniform(3, 8)
            df = -random.uniform(3, 7)
            ds = random.uniform(5, 12)
            dw = random.uniform(0, 3)
            s._consecutive_challenge = 0
            s._consecutive_empathy += 1

        elif action.type == "guide":
            s._consecutive_empathy = 0
            s._consecutive_challenge = 0
            if s.energy > 30:
                de = -random.uniform(3, 8)
                df = -random.uniform(5, 10)
                dw = random.uniform(5, 12)
            else:
                df = random.uniform(5, 12)
                dw = -random.uniform(3, 8)
                de = -random.uniform(2, 5)

        elif action.type == "encourage":
            de = random.uniform(1, 4)
            df = -random.uniform(0, 3)
            dw = random.uniform(3, 8)
            s._consecutive_empathy = 0
            s._consecutive_challenge = 0

        elif action.type == "challenge":
            s._consecutive_challenge += 1
            s._consecutive_empathy = 0
            if s.energy > 30 and s.willingness > 25:
                de = -random.uniform(5, 10)
                dw = random.uniform(8, 15)
                df = -random.uniform(2, 6)
            else:
                df = random.uniform(8, 15)
                dw = -random.uniform(5, 10)
                de = -random.uniform(3, 8)
            if s._consecutive_challenge >= 3:
                dw -= 3; de -= 2

        elif action.type == "inform":
            de = -random.uniform(0, 3)
            s._consecutive_empathy = 0
            s._consecutive_challenge = 0

        # Apply effects
        s.energy += de
        s.frustration += df
        s.social_support += ds
        s.willingness += dw

        # Clamp
        s.energy = max(0, min(100, s.energy))
        s.frustration = max(0, min(100, s.frustration))
        s.social_support = max(0, min(100, s.social_support))
        s.willingness = max(0, min(100, s.willingness))

        # === Triple-objective reward ===
        max_steps = self.config.get("max_steps", 20)
        alpha, beta = 0.2, 0.2
        surv = -alpha * max(0, s.frustration - 80) - beta * max(0, 20 - s.energy)
        l1, l2, l3, l4 = 0.3, 0.3, 0.2, 0.4
        rec = l1 * de - l2 * df + l3 * ds + l4 * dw
        delta_coef = 0.1
        inst = delta_coef * (abs(de) + abs(df) + abs(ds) + abs(dw))
        reward = round(surv + rec - inst, 2)

        # Terminal conditions
        done = False
        if s.energy <= 0:
            reward -= 10.0
            done = True
        elif s.willingness > 70 and s.energy > 50 and s.frustration < 30:
            reward += 8.0
            done = True
        elif s.step >= max_steps:
            done = True
        s.done = done

        result = StepResult(
            state=s.clone(),
            action=action,
            reward=reward,
            done=done,
            info={"action_type": action.type, "surv": surv, "rec": rec, "inst": inst}
        )
        self.history.append(result)
        return result

    def available_actions(self, state: WorldState) -> List[str]:
        return ["empathize", "guide", "encourage", "challenge", "inform"]

    def render(self, state: WorldState) -> str:
        s = state
        bar = lambda v, n=20: "█" * int(v / 100 * n) + "░" * (n - int(v / 100 * n))
        return (
            f"Energy:      {s.energy:5.1f} {bar(s.energy)}\n"
            f"Frustration: {s.frustration:5.1f} {bar(s.frustration)}\n"
            f"Support:     {s.social_support:5.1f} {bar(s.social_support)}\n"
            f"Willingness: {s.willingness:5.1f} {bar(s.willingness)}\n"
        )
