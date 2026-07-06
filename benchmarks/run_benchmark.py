"""WorldForge · Benchmark runner

Run episodes, collect trajectories, generate evaluation reports.
"""
import json, time, os
from typing import Optional
from envs import ENV_REGISTRY
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from metrics.evaluation import evaluate


def run_episode(env, agent, max_steps: int = 25) -> dict:
    """Run one episode. Returns trajectory data."""
    state = env.reset()
    agent.reset()
    steps = []
    total_reward = 0.0

    while not state.done:
        state_before = state.clone()
        action = agent.act(state, env.available_actions(state))
        result = env.step(action)
        state = result.state
        steps.append({
            "step": state.step,
            "action": action.type,
            "reward": result.reward,
            "state_before": state_before.to_dict(),
        })
        total_reward += result.reward

    return {
        "total_reward": round(total_reward, 2),
        "steps_taken": state.step,
        "final_state": state.to_dict(),
        "steps": steps,
    }


def benchmark(env_name: str, agent, episodes: int = 10, max_steps: int = 25) -> dict:
    """Run multiple episodes and produce an EvalReport."""
    env_class = ENV_REGISTRY[env_name]
    all_data = [run_episode(env_class({"max_steps": max_steps}), agent, max_steps)
                for _ in range(episodes)]
    report = evaluate(env_name, agent.name, all_data)
    return report


def compare(env_name: str, episodes: int = 10, max_steps: int = 25, save: bool = True) -> dict:
    """Run random vs rule comparison for an environment."""
    agents = {
        "RandomAgent": RandomAgent(),
        "RuleAgent": RuleAgent(),
    }
    results = {}
    for name, agent in agents.items():
        results[name.lower()] = benchmark(env_name, agent, episodes, max_steps)

    if save:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(out_dir, exist_ok=True)
        for key, report in results.items():
            out = os.path.join(out_dir, f"{env_name}_{key}_{int(time.time())}.json")
            with open(out, "w") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    return results
