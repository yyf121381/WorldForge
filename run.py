#!/usr/bin/env python3
"""WorldForge v0.1 — Synthetic Decision Environment Benchmark

Usage:
    python run.py --compare                                # Compare all envs
    python run.py --env emotion_depletion --agent rule     # Single config
    python run.py --env meaning_crisis --compare           # Compare on one env
"""
import argparse
from envs import ENV_REGISTRY
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from benchmarks.run_benchmark import benchmark, compare


def main():
    parser = argparse.ArgumentParser(description="WorldForge Benchmark")
    parser.add_argument("--env", choices=list(ENV_REGISTRY.keys()) + ["all"], default="all")
    parser.add_argument("--agent", choices=["random", "rule"], default="rule")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--compare", action="store_true", help="Compare random vs rule")
    args = parser.parse_args()

    envs = list(ENV_REGISTRY.keys()) if args.env == "all" else [args.env]

    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        if args.compare:
            results = compare(env_name, args.episodes, args.max_steps)
            print()
            for aname, report in results.items():
                print(report.summary())
                print()
        else:
            agent = RandomAgent() if args.agent == "random" else RuleAgent()
            report = benchmark(env_name, agent, args.episodes, args.max_steps)
            print()
            print(report.summary())


if __name__ == "__main__":
    main()
