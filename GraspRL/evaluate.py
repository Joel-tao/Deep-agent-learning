"""Evaluate a saved SAC policy with a frozen reward schedule and log success."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from grasp_env import AdaptiveGraspEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--render", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    model = SAC.load(str(args.model), device="cpu")
    # Common sparse reward for evaluation; success definition identical across models.
    env = AdaptiveGraspEnv(reward_mode="sparse", render_mode="human" if args.render else None)
    results = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            for step in range(env.horizon):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            results.append({"seed": args.seed + episode, "steps": step + 1, "success": info["is_success"]})
    finally:
        env.close()
    payload = {"model": str(args.model), "success_rate": float(np.mean([x["success"] for x in results])),
               "episodes": results, "note": "Single-policy evaluation; no cross-condition conclusion."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
