"""WorldForge · Agent interfaces"""
from abc import ABC, abstractmethod
from typing import List
from envs.base_env import WorldState, Action


class Agent(ABC):
    @abstractmethod
    def act(self, state: WorldState, available: List[str]) -> Action:
        pass

    def reset(self):
        pass
