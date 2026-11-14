"""Write actual observation layout, success threshold, versions and hardware RAM."""
import json
import platform
import importlib.metadata as metadata
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
from grasp_env import AdaptiveGraspEnv

env = AdaptiveGraspEnv()
try:
    obs, _ = env.reset(seed=42)
    raw = env.sim_env._get_observations(force_update=True)
    packages = ["robosuite", "mujoco", "gymnasium", "stable-baselines3", "numpy", "torch"]
    manifest = {"platform": platform.platform(), "python": platform.python_version(),
        "versions": {p: metadata.version(p) for p in packages},
        "observation_dim": len(obs), "selected_keys": list(env.observation_keys),
        "raw_observation_shapes": {k: list(v.shape) for k, v in raw.items()},
        "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
        "reach_threshold_m": env.reach_threshold, "table_z_m": float(env.sim_env.table_offset[2]),
        "lift_success_center_z_m": float(env.sim_env.table_offset[2] + 0.04),
        "control_hz": 20, "simulation_timestep_s": env.sim_env.model_timestep,
        "model_construction_seed": 0, "cube_half_sizes_m": env.sim_env.cube.size.tolist(),
        "total_ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "available_ram_gib_at_check": round(psutil.virtual_memory().available / 2**30, 2)}
    Path("results/environment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
finally:
    env.close()
