"""Optional SAC entry point. Short smoke runs do not establish policy quality."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from grasp_env import AdaptiveGraspEnv
from replay_buffer import ComponentReplayBuffer


class EpisodeLogger(BaseCallback):
    def __init__(self, env, path):
        super().__init__()
        self.task_env, self.path = env, path

    def _on_step(self):
        if isinstance(self.model.replay_buffer, ComponentReplayBuffer):
            # Current episode's weights; next reset activates newly scheduled weights.
            self.model.replay_buffer.reward_weights = self.task_env._episode_weights.copy()
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([self.num_timesteps, info["is_success"],
                        *info["stage_ever"].tolist(), *info["weights"].tolist()])
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sparse", "fixed", "adaptive"], default="adaptive")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--historical-rewards", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/training"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    env = AdaptiveGraspEnv(reward_mode=args.mode)
    wrapped = Monitor(env, str(args.output / "monitor.csv"))
    relabel = args.mode == "adaptive" and not args.historical_rewards
    try:
        model = SAC("MlpPolicy", wrapped, seed=args.seed, device="cpu", verbose=1,
            buffer_size=args.buffer_size, learning_starts=args.learning_starts,
            batch_size=128, policy_kwargs={"net_arch": [128, 128]},
            replay_buffer_class=ComponentReplayBuffer if relabel else None)
        log = args.output / "episodes.csv"
        log.write_text("step,success,reach,grasp,lift,w_reach,w_grasp,w_lift\n", encoding="utf-8")
        model.learn(total_timesteps=args.steps, callback=EpisodeLogger(env, log))
        model.save(str(args.output / "sac_model"))
        import psutil
        memory = psutil.Process().memory_info()
        metadata = {"steps": model.num_timesteps, "gradient_updates": model._n_updates,
                    "seed": args.seed, "mode": args.mode, "relabel": relabel,
                    "observation_dim": int(env.observation_space.shape[0]),
                    "action_dim": int(env.action_space.shape[0]),
                    "scheduler": env.scheduler.snapshot(),
                    "peak_process_ram_mib": round(getattr(memory, "peak_wset", memory.rss) / 2**20, 1),
                    "note": "Training execution only; no claim of learned grasping performance."}
        (args.output / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
