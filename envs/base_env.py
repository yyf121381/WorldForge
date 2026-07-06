"""WorldForge · Base environment classes.

WorldState, Action, StepResult, Environment — shared across all envs.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json, copy
from abc import ABC, abstractmethod


@dataclass
class WorldState:
    episode: int = 0
    step: int = 0
    done: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    def clone(self) -> "WorldState":
        return copy.deepcopy(self)

    def summary(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Action:
    type: str
    params: Dict = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict:
        return {"type": self.type, "params": self.params}


@dataclass
class StepResult:
    state: WorldState
    action: Action
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


class Environment(ABC):
    """Closed-world simulation environment base."""

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.history: List[StepResult] = []

    @abstractmethod
    def reset(self) -> WorldState:
        pass

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        pass

    @abstractmethod
    def available_actions(self, state: WorldState) -> List[str]:
        pass

    def render(self, state: WorldState) -> str:
        return state.summary()
