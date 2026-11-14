"""Random and hand-scripted demos; neither is a trained RL policy."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from grasp_env import AdaptiveGraspEnv


class ScriptedPolicy:
    """Position servo: approach, descend, close, lift. Cube pose is simulator state."""
    def __init__(self):
        self.stage = 0
        self.ticks = 0

    def action(self, env):
        base = env.sim_env
        eef = base.sim.data.site_xpos[base.robots[0].eef_site_id["right"]].copy()
        cube = base.sim.data.body_xpos[base.cube_body_id].copy()
        target = cube.copy()
        gripper = -1.0
        if self.stage == 0:
            target[2] += 0.12
        elif self.stage == 1:
            target[2] += 0.005
        elif self.stage == 2:
            target[2] += 0.005
            gripper = 1.0
        else:
            target = self.lift_target
            gripper = 1.0
        delta = target - eef
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[:3] = np.clip(delta * 12, -0.7, 0.7)
        action[-1] = gripper
        self.ticks += 1
        if self.stage in (0, 1) and np.linalg.norm(delta) < 0.012:
            self.stage += 1
            self.ticks = 0
        elif self.stage == 2 and self.ticks >= 20:
            self.stage = 3
            self.lift_target = eef + np.array([0, 0, 0.18])
        return action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["random", "scripted"], default="random")
    p.add_argument("--mode", choices=["sparse", "fixed", "adaptive"], default="fixed")
    p.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/demo_summary.json"))
    p.add_argument("--realtime", action="store_true")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    if args.video and args.render != "rgb_array":
        p.error("--video requires --render rgb_array")
    env = AdaptiveGraspEnv(reward_mode=args.mode,
        render_mode=None if args.render == "none" else args.render, horizon=args.horizon)
    writer = None
    summaries = []
    import psutil
    process = psutil.Process()
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=20, macro_block_size=1)
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            env.action_space.seed(args.seed + episode)
            policy = ScriptedPolicy()
            total = 0.0
            episode_started = time.perf_counter()
            peak_rss = process.memory_info().rss
            saved_grasp = False
            trace = []
            if writer:
                writer.append_data(env.render())
            print(f"Episode {episode}: observation={obs.shape}, action={env.action_space.shape}, reset_info={info}")
            for step in range(args.horizon):
                started = time.perf_counter()
                action = env.action_space.sample() if args.policy == "random" else policy.action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                peak_rss = max(peak_rss, process.memory_info().rss)
                trace.append({"step": step + 1, "reward": reward,
                    "distance": info["reach_distance"], "grasp": info["valid_grasp"],
                    "cube_z": info["cube_height"], "success": info["is_success"]})
                if writer:
                    frame = env.render()
                    writer.append_data(frame)
                    if episode == 0 and step in (0, 60, 120):
                        import imageio.v2 as imageio
                        imageio.imwrite(str(args.video.with_name(f"{args.policy}_step_{step:03}.png")), frame)
                    if info["valid_grasp"] and not saved_grasp:
                        imageio.imwrite(str(args.video.with_name(f"{args.policy}_grasp.png")), frame)
                        saved_grasp = True
                    if terminated or truncated:
                        imageio.imwrite(str(args.video.with_name(f"{args.policy}_final.png")), frame)
                if step % 20 == 0 or terminated or truncated:
                    print(f"step={step+1} reward={reward:.4f} terminated={terminated} truncated={truncated} "
                          f"distance={info['reach_distance']:.4f} grasp={info['valid_grasp']} success={info['is_success']}")
                if args.realtime:
                    time.sleep(max(0, 0.05 - (time.perf_counter() - started)))
                if terminated or truncated:
                    break
            summaries.append({"episode": episode, "seed": args.seed + episode,
                "policy": args.policy, "mode": args.mode, "steps": step + 1,
                "return": total, "terminated": terminated, "truncated": truncated,
                "success": info["is_success"], "stage_ever": info["stage_ever"].tolist(),
                "peak_process_ram_mib": round(peak_rss / 2**20, 1),
                "elapsed_seconds": round(time.perf_counter() - episode_started, 3),
                "scheduler": env.scheduler.snapshot()})
            args.output.with_name(f"{args.output.stem}_episode_{episode}_trace.json").write_text(
                json.dumps(trace, indent=2), encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
