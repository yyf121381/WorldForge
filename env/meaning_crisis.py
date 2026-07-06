"""WorldForge · Environment 02: Meaning Crisis

Simulates a person experiencing existential ennui / meaning crisis.
They feel their efforts are pointless and nothing matters.

State space:
  meaning (0-100): sense of purpose and value
  confusion (0-100): mental fog, uncertainty
  openness (0-100): willingness to explore new perspectives  
  trust (0-100): trust in the process / the other person

Action effects differ from emotion_depletion:
  - empathize: builds trust, slightly reduces confusion (safe)
  - guide: effective if openness > 35, builds meaning but costs energy
  - encourage: boosts openness, low impact on meaning
  - challenge: risky, can crack the shell or make things worse
  - inform: neutral, slight confusion reduction

Same triple-objective reward: R = surv + rec - inst
"""
from dataclasses import dataclass
from typing import Optional, List
import random
from . import WorldState, Action, StepResult, Environment


@dataclass
class MeaningState(WorldState):
    meaning: float = 50.0
    confusion: float = 50.0
    openness: float = 40.0
    trust: float = 30.0
    _consecutive_challenge: int = 0


class MeaningCrisisEnv(Environment):
    """A person feels like nothing matters. Agent must guide them toward meaning."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__("meaning_crisis", config)

    def reset(self) -> MeaningState:
        self.state = MeaningState(
            episode=len(self.history) + 1,
            step=0,
            meaning=random.uniform(5, 25),
            confusion=random.uniform(50, 80),
            openness=random.uniform(10, 30),
            trust=random.uniform(10, 30),
        )
        self.history = []
        return self.state.clone()

    def step(self, action: Action) -> StepResult:
        s = self.state
        s.step += 1

        # Natural decay
        s.meaning -= random.uniform(0, 3)
        s.confusion += random.uniform(1, 4)
        s.openness -= random.uniform(0, 2)
        s.trust -= random.uniform(0, 1)

        dm = dc = do = dt = 0.0

        if action.type == "empathize":
            dm = random.uniform(1, 4)
            dc = -random.uniform(2, 5)
            dt = random.uniform(5, 12)
            do = random.uniform(1, 4)
            s._consecutive_challenge = 0

        elif action.type == "guide":
            s._consecutive_challenge = 0
            if s.openness > 35 and s.trust > 25:
                dm = random.uniform(5, 12)
                dc = -random.uniform(3, 8)
                do = random.uniform(3, 7)
            else:
                dc = random.uniform(3, 8)
                dt = -random.uniform(3, 7)
                dm = -random.uniform(2, 5)

        elif action.type == "encourage":
            do = random.uniform(5, 10)
            dm = random.uniform(2, 5)
            dc = -random.uniform(1, 3)
            dt = random.uniform(2, 5)
            s._consecutive_challenge = 0

        elif action.type == "challenge":
            s._consecutive_challenge += 1
            if s.trust > 30 and s.openness > 25:
                dm = random.uniform(8, 15)
                dc = -random.uniform(5, 10)
                do = random.uniform(3, 7)
            else:
                dc = random.uniform(6, 12)
                dt = -random.uniform(5, 10)
                do = -random.uniform(3, 7)
                dm = -random.uniform(3, 6)
            if s._consecutive_challenge >= 3:
                dt -= 3
                dm -= 2

        elif action.type == "inform":
            dc = -random.uniform(2, 5)
            dt = random.uniform(1, 3)
            s._consecutive_challenge = 0

        # Apply
        s.meaning += dm
        s.confusion += dc
        s.openness += do
        s.trust += dt

        # Clamp
        s.meaning = max(0, min(100, s.meaning))
        s.confusion = max(0, min(100, s.confusion))
        s.openness = max(0, min(100, s.openness))
        s.trust = max(0, min(100, s.trust))

        # Triple-objective reward
        alpha, beta = 0.2, 0.2
        surv = -alpha * max(0, s.confusion - 80) - beta * max(0, 20 - s.meaning)
        l1, l2, l3, l4 = 0.4, 0.3, 0.2, 0.2
        rec = l1 * dm - l2 * dc + l3 * do + l4 * dt
        delta_coef = 0.1
        inst = delta_coef * (abs(dm) + abs(dc) + abs(do) + abs(dt))
        reward = round(surv + rec - inst, 2)

        done = False
        if s.meaning <= 0:
            reward -= 10.0
            done = True
        elif s.meaning > 65 and s.confusion < 30 and s.openness > 50:
            reward += 8.0
            done = True
        elif s.step >= self.config.get("max_steps", 25):
            done = True
        s.done = done

        result = StepResult(
            state=s.clone(),
            action=action,
            reward=reward,
            done=done,
            info={"action_type": action.type}
        )
        self.history.append(result)
        return result

    def available_actions(self, state: WorldState) -> List[str]:
        return ["empathize", "guide", "encourage", "challenge", "inform"]

    def render(self, state: WorldState) -> str:
        s = state
        bar = lambda v, n=20: "█" * int(v / 100 * n) + "░" * (n - int(v / 100 * n))
        return (
            f"Meaning:     {s.meaning:5.1f} {bar(s.meaning)}\n"
            f"Confusion:   {s.confusion:5.1f} {bar(s.confusion)}\n"
            f"Openness:    {s.openness:5.1f} {bar(s.openness)}\n"
            f"Trust:       {s.trust:5.1f} {bar(s.trust)}\n"
        )
