# ME5418 Deep Reinforcement Learning for Robotic Grasping

Group 36 project on adaptive reward shaping for a simulated Panda arm. The agent uses Soft Actor-Critic (SAC) to reach, grasp, and lift a tabletop cube in MuJoCo and robosuite. Fixed and adaptive training share the robot, policy architecture, and task criteria. Adaptive training changes the relative weights of Reach, Grasp, and Lift and recomputes rewards from stored components when sampling replay.

![Same-scene comparison of the scripted controller and trained policies](media/comparison.gif)

The animation projects actual simulation positions onto the XZ plane. The scripted controller is a feasibility reference. The other panels show learned policies. See the [distance, height, gripper and contact traces](media/signals.png) for the same scene.

## Run locally

```bash
conda env create -f environment.yml
conda activate learning_agent_g36
python demo.py
python -m pytest -q
```

The environment lock records the Windows CPU validation setup. The default demo saves a scripted grasp animation and trajectory locally.

```bash
python -m training.train_fixed --seed 42 --timesteps 10000 --output-root runs/fixed_example
python -m training.train_adaptive --seed 42 --timesteps 10000 --output-root runs/adaptive_example
python -m evaluation.evaluate runs/adaptive_example/adaptive/seed_42/policy.zip --episodes 20 --seed 30000
```

These small example budgets exercise the training pipeline. For the sustained-contact experiment use `--grasp-hold-steps 5`. The optional preparation reward also uses `--reward-profile prepare_v1`.

## Results

- [Original Fixed and Adaptive evaluation](results/original_evaluation.csv). In the single-seed 200,000-step evaluation, Fixed lifted in 0 of 20 episodes and Adaptive lifted in 1 of 20. This is preliminary evidence with a small sample.
- [Trajectory diagnostics](results/failure_diagnostics.csv). Learned policies often failed to approach or maintain bilateral contact. One raw height event occurred without held contact.
- [Preparation reward pilot](results/preparation_pilot.csv). Both 10,000-step variants achieved zero grasps and lifts in ten held-out scenes.

`envs/` wraps the simulator, `rewards/` defines reward components and the curriculum, `training/` runs SAC, and `evaluation/` records policy behavior. MuJoCo and robosuite supply physics and robot control. Stable-Baselines3 and PyTorch supply SAC and neural-network optimization.
