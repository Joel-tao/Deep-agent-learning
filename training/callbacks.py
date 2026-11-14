"""Training-time metrics for reproducible reward-shaping experiments."""
import csv
from pathlib import Path

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class LiftMetricsCallback(BaseCallback):
    """Record terminal skill outcomes, scheduler rates, and weights to CSV/TensorBoard."""

    def __init__(self, csv_path: str | Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.file = None
        self.writer = None

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        fields = ["timesteps", "episode_return", "episode_length", "reached", "grasped", "success", "reach_rate", "grasp_rate", "lift_rate", "weight_reach", "weight_grasp", "weight_lift", "scheduler_stage", "next_scheduler_stage"]
        self.diagnostic_fields = ['stable_grasp', 'stable_lift', 'max_contact_steps',
                                  'max_held_lift_steps', 'max_height_after_contact',
                                  'upward_action_fraction_after_contact']
        fields += self.diagnostic_fields
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Monitor supplies episode data only at episode end, not at every control step.
            if "episode" not in info:
                continue
            rates = info.get("skill_rates", {})
            weights = info.get("reward_weights", {})
            episode = info["episode"]
            # Log episode skills, the weights used for this reward, and the next stage.
            row = {
                "timesteps": self.num_timesteps,
                "episode_return": episode["r"], "episode_length": episode["l"],
                "reached": float(info.get("episode_reached", False)),
                "grasped": float(info.get("episode_grasped", False)),
                "success": float(info.get("episode_success", False)),
                "reach_rate": rates.get("reach_rate", ""), "grasp_rate": rates.get("grasp_rate", ""), "lift_rate": rates.get("lift_rate", ""),
                "weight_reach": weights.get("reach", ""), "weight_grasp": weights.get("grasp", ""), "weight_lift": weights.get("lift", ""),
                "scheduler_stage": info.get("scheduler_stage", ""),
                "next_scheduler_stage": info.get("next_scheduler_stage", ""),
            }
            row.update({k:info.get('diagnostics', {}).get(k, '') for k in self.diagnostic_fields})
            self.writer.writerow(row)
            self.file.flush()
            for name in ("reached", "grasped", "success"):
                self.logger.record(f"skills/{name}", row[name])
            for name, value in rates.items(): self.logger.record(f"scheduler/{name}", value)
            for name, value in weights.items(): self.logger.record(f"scheduler/weight_{name}", value)
        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.close()


class SuccessRateEvalCallback(BaseCallback):
    """Periodically evaluate/save models, with optional stopping after repeated passes.

    The shared runner sets consecutive_passes=0 to disable early stopping
    and keep interaction budgets equal across reward conditions.
    """

    def __init__(
        self,
        eval_env,
        model_path: str | Path,
        *,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 20,
        target_success_rate: float = 0.90,
        consecutive_passes: int = 3,
        seed: int = 10_000,
        csv_path: str | Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.model_path = Path(model_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.target_success_rate = target_success_rate
        self.consecutive_passes = consecutive_passes
        self.seed = seed
        self.csv_path = Path(csv_path) if csv_path else None
        self.passes = 0
        self.best_success_rate = -np.inf
        self.best_score = (-1.0, -1.0, -1.0)
        self.last_metrics = {}

    def _on_training_start(self) -> None:
        if self.eval_freq < 1 or self.n_eval_episodes < 1:
            raise ValueError("Evaluation interval and episode count must be positive")
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_path.write_text("timesteps,success_rate,reach_rate,grasp_rate,episodes\n", encoding="utf-8")
        # Record the actual policy before this run, not a fabricated
        # initial point (a resumed run already has training history). All compared reward conditions use the same procedure.
        self._evaluate_and_save()

    def _evaluate(self) -> float:
        """Run a deterministic policy on common evaluation seeds and count episode skills."""
        successes, reaches, grasps = [], [], []
        for episode in range(self.n_eval_episodes):
            # Reuse the same held-out reset seeds at each checkpoint to control placement variation.
            obs, _ = self.eval_env.reset(seed=self.seed + episode)
            terminated = truncated = False
            success = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = self.eval_env.step(action)
                success |= bool(info.get("is_success", False))
            successes.append(success)
            reaches.append(bool(info.get("episode_reached", False)))
            grasps.append(bool(info.get("episode_grasped", False)))
        self.last_metrics = dict(success_rate=float(np.mean(successes)),
                                 reach_rate=float(np.mean(reaches)),
                                 grasp_rate=float(np.mean(grasps)))
        return self.last_metrics["success_rate"]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq:
            return True
        return self._evaluate_and_save()

    def _evaluate_and_save(self) -> bool:
        """Log evaluation and save models; return whether SB3 should continue training."""
        success_rate = self._evaluate()
        if self.csv_path:
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([self.num_timesteps, success_rate,
                                          self.last_metrics["reach_rate"], self.last_metrics["grasp_rate"],
                                          self.n_eval_episodes])
        for name, value in self.last_metrics.items():
            self.logger.record(f"eval/{name}", value)
        if self.verbose:
            print(f"\nEvaluation at {self.num_timesteps:,}: {success_rate:.1%} Lift, "
                  f"{self.last_metrics['reach_rate']:.1%} Reach, {self.last_metrics['grasp_rate']:.1%} Grasp",
                  flush=True)
        # Lift remains primary. Skill progress breaks a tie when every candidate
        # has the same Lift rate, rather than always retaining an untrained model.
        score = (success_rate, self.last_metrics["grasp_rate"], self.last_metrics["reach_rate"])
        if score > self.best_score:
            self.best_score = score
            self.best_success_rate = success_rate
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
        self.model.save(self.model_path.parent / "latest")
        self.passes = self.passes + 1 if success_rate >= self.target_success_rate else 0
        self.logger.record("eval/consecutive_target_passes", self.passes)
        if self.consecutive_passes > 0 and self.passes >= self.consecutive_passes:
            print(
                f"Target met: >= {self.target_success_rate:.0%} for "
                f"{self.consecutive_passes} consecutive evaluations."
            )
            return False
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()
