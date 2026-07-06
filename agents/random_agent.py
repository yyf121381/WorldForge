"""RandomAgent: selects actions uniformly at random."""
import random
from typing import List
from agents import Agent
from envs.base_env import WorldState, Action


class RandomAgent(Agent):
    def __init__(self, name: str = "RandomAgent"):
        self.name = name

    def act(self, state: WorldState, available: List[str]) -> Action:
        return Action(random.choice(available), {"reason": "random"})
