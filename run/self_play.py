"""WorldForge · Self-play runner + Evaluation

Runs the closed-loop: State → Agent → Action → Environment → Reward → Next State
Generates standardized evaluation reports with metrics across agents.

Usage:
    python run/self_play.py                                          # RuleAgent demo
    python run/self_play.py --agent rule --episodes 20               # Benchmark
    python run/self_play.py --agent random --episodes 20             # Random baseline
    python run/self_play.py --agent rule --random --compare          # Side-by-side
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from env.emotion_depletion import EmotionDepletionEnv
from env.meaning_crisis import MeaningCrisisEnv
from agent import RandomAgent, RuleAgent, LingYaoAgent
from eval import compute_report


# Registry
ENVIRONMENTS = {
    "emotion_depletion": EmotionDepletionEnv,
    "meaning_crisis": MeaningCrisisEnv,
}
AGENTS = {
    "random": RandomAgent,
    "rule": RuleAgent,
}


def run_episode(env: Environment, agent, max_steps: int = 30) -> dict:
    """Run one complete episode, return trajectory data."""
    state = env.reset()
    agent.reset()
    steps = []
    total_reward = 0.0

    while not state.done:
        state_before = state.clone()
        available = env.available_actions(state)
        action = agent.act(state, available)
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


def run_benchmark(env_name: str, agent, episodes: int, max_steps: int) -> dict:
    """Run multiple episodes and tabulate results."""
    env_class = ENVIRONMENTS[env_name]
    all_data = []

    for ep in range(episodes):
        env = env_class({"max_steps": max_steps})
        data = run_episode(env, agent, max_steps)
        all_data.append(data)

    report = compute_report(env_name, agent.name, all_data)
    return report


def main():
    parser = argparse.ArgumentParser(description="WorldForge Self-Play")
    parser.add_argument("--env", default="emotion_depletion", choices=list(ENVIRONMENTS.keys()))
    parser.add_argument("--agent", default="rule", choices=list(AGENTS.keys()))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="Compare random vs rule")
    args = parser.parse_args()

    if args.compare:
        # Side-by-side comparison
        ref_results = {}
        for aname, aclass in AGENTS.items():
            agent = aclass()
            report = run_benchmark(args.env, agent, args.episodes, args.max_steps)
            ref_results[aname] = report

            # Save individual
            out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "results", f"{args.env}_{aname}_{int(time.time())}.json")
            with open(out, "w") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Print comparison table
        print(f"\n{'='*60}")
        print(f"Comparison: {args.env} × {args.episodes} episodes")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Random':>12} {'Rule':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        for metric in ["avg_reward", "std_reward", "min_reward", "max_reward", "avg_steps"]:
            r_val = ref_results["random"].to_dict()["metrics"][metric]
            ru_val = ref_results["rule"].to_dict()["metrics"][metric]
            print(f"  {metric:<20} {r_val:>11.2f} {ru_val:>11.2f}")
        for metric in ["success_rate", "failure_rate", "timeout_rate"]:
            r_val = ref_results["random"].to_dict()["metrics"][metric]
            ru_val = ref_results["rule"].to_dict()["metrics"][metric]
            print(f"  {metric:<20} {r_val:>11.0%} {ru_val:>11.0%}")
        # Velocity
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        for metric in ["avg_velocity", "sustained_velocity"]:
            r_val = ref_results["random"].to_dict()["velocity"][metric]
            ru_val = ref_results["rule"].to_dict()["velocity"][metric]
            print(f"  {metric:<20} {r_val:>+11.2f} {ru_val:>+11.2f}")
        print(f"\n  Action Profile:")
        all_actions = set(list(ref_results["random"].action_counts.keys()) + list(ref_results["rule"].action_counts.keys()))
        for a in sorted(all_actions):
            rc = ref_results["random"].action_counts.get(a, 0)
            ruc = ref_results["rule"].action_counts.get(a, 0)
            print(f"    {a:<15} random={rc}  rule={ruc}")
    else:
        # Single agent benchmark
        agent = AGENTS[args.agent]()
        print(f"\nWorldForge · Benchmark")
        print(f"  Environment: {args.env}")
        print(f"  Agent:       {agent.name}")
        print(f"  Episodes:    {args.episodes}")

        report = run_benchmark(args.env, agent, args.episodes, args.max_steps)
        print(f"\n{report.summary()}")

        # Save
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", f"{args.env}_{agent.name}_{int(time.time())}.json")
        with open(out, "w") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
