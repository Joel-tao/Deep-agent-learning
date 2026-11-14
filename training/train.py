"""Shared SAC runner: reward condition is the only default experimental variable."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from envs.factory import make_env
from envs.observation_normalization import ObservationNormalization
from training.callbacks import LiftMetricsCallback, SuccessRateEvalCallback
from training.replay_buffer import ComponentReplayBuffer
from rewards.adaptive_reward import AdaptiveRewardScheduler
from rewards.adaptive_reward_wrapper import FIXED_WEIGHT_RATIOS, FIXED_WEIGHTS
from rewards.scaling import REWARD_SCALE_VERSION, SHAPING_BUDGET, SUCCESS_REWARD


def main(condition=None):
    """Run shared SAC training; the three entry points select a reward condition.

    CLI options set the interaction budget, seed, and output directory; save
    the full configuration with each run. Resume restores a checkpoint while
    collecting a fresh replay buffer.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    if condition is None:
        parser.add_argument('--condition', choices=['sparse', 'fixed', 'adaptive'], required=True)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=500_000)
    parser.add_argument('--learning-starts', type=int, default=10_000)
    parser.add_argument('--eval-freq', type=int, default=50_000, help='0 disables evaluation')
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--eval-seed', type=int, default=10_000)
    parser.add_argument('--output-root', type=Path, default=Path('runs'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--ent-coef', default='auto_0.1',
                        help='initial entropy coefficient; shared across all reward conditions')
    parser.add_argument('--normalize-observations', action='store_true',
                        help='standardize policy inputs; save stats with the model, keep rewards unchanged')
    parser.add_argument('--historical-rewards', action='store_true', help='adaptive replay ablation')
    parser.add_argument('--reward-profile', choices=['legacy','prepare_v1'], default='legacy')
    parser.add_argument('--grasp-hold-steps', type=int, default=1,
                        help='contact reward ramp length; >1 appends contact progress to observations')
    parser.add_argument('--scheduler-window', type=int, default=100)
    parser.add_argument('--scheduler-interval', type=int, default=10)
    parser.add_argument('--scheduler-confirmations', type=int, default=2)
    parser.add_argument('--transition-episodes', type=int, default=0)
    parser.add_argument('--grasp-stage-lift-ratio', type=float, default=0.3)
    parser.add_argument('--resume', type=Path,
                        help='continue a current-scale checkpoint in a new output root; replay starts empty')
    args = parser.parse_args()
    if args.grasp_hold_steps < 1:
        parser.error('--grasp-hold-steps must be positive')
    scheduler_options = dict(window_size=args.scheduler_window, update_interval=args.scheduler_interval,
                             confirmations=args.scheduler_confirmations,
                             transition_episodes=args.transition_episodes,
                             grasp_stage_lift_ratio=args.grasp_stage_lift_ratio)
    # Validate before creating run directories or starting physics.
    AdaptiveRewardScheduler(**scheduler_options)
    condition = condition or args.condition
    if min(args.timesteps, args.horizon, args.batch_size, args.buffer_size,
           args.eval_episodes, args.torch_threads) < 1 or args.eval_freq < 0 or args.learning_starts < 0:
        parser.error('Budgets and counts must be positive; eval-freq and learning-starts may be zero')
    if args.historical_rewards and condition != 'adaptive':
        parser.error('--historical-rewards applies only to adaptive runs')
    # Check reward and replay compatibility before resuming an existing experiment.
    source_config = None
    if args.resume:
        source_path = args.resume.parent / 'config.json'
        if not args.resume.is_file() or not source_path.is_file():
            parser.error('--resume requires a model and its adjacent config.json')
        source_config = json.loads(source_path.read_text())
        if args.reward_profile != source_config.get('reward_profile','legacy'):
            parser.error('Resume requires the same reward profile')
        if (source_config.get('condition') != condition or
                source_config.get('reward_config', {}).get('scale_version') != REWARD_SCALE_VERSION or
                source_config.get('historical_rewards', False) != args.historical_rewards):
            parser.error('Resume requires the same reward condition, scale version, and replay treatment')
        args.normalize_observations = source_config.get('normalize_observations', False)
        if args.grasp_hold_steps != source_config.get('grasp_hold_steps', 1):
            parser.error('Resume requires the same grasp-hold-steps and observation layout')
        scheduler_options = source_config.get('scheduler_options', dict(window_size=100, update_interval=10, confirmations=2))
    # Never mix old and new runs or overwrite previous results silently.
    suffix = '_historical' if args.historical_rewards else ''
    run_dir = args.output_root / (condition + suffix) / f'seed_{args.seed}'
    if run_dir.exists():
        parser.error(f'{run_dir} exists; choose a new --output-root')
    run_dir.mkdir(parents=True)
    torch.set_num_threads(args.torch_threads)
    task = make_env(condition, horizon=args.horizon, seed=args.seed, scheduler_config=scheduler_options, grasp_hold_steps=args.grasp_hold_steps, reward_profile=args.reward_profile)
    # Keep references to the task wrapper for curriculum and skill metrics.
    if args.grasp_hold_steps > 1:
        task_for_scheduler = task.env
    else:
        task_for_scheduler = task
    normalized_env = ObservationNormalization(task) if args.normalize_observations else None
    # Monitor supplies episode returns/lengths; task retains the physical skill interface.
    env = Monitor(normalized_env if normalized_env is not None else task)
    eval_env = None
    try:
        obs, _ = env.reset(seed=args.seed)
        # Record measured dimensions, reward version, and parameters for reproducibility.
        config = dict(vars(args), scheduler_options=scheduler_options, condition=condition, output_root=str(args.output_root),
                      resume=str(args.resume) if args.resume else None,
                      observation_dim=int(obs.size), action_dim=int(env.action_space.shape[0]),
                      controller='OSC_POSE', control_freq=20, gamma=0.99, tau=0.005,
                      ent_coef=args.ent_coef, net_arch=[256, 256],
                      run_kind='pipeline_check' if args.horizon < 100 or args.timesteps < 1000 else
                               ('pilot' if args.timesteps < 1_000_000 else 'experiment'),
                      reward_config=dict(reach_distance=0.06, reach_scale=10.0,
                                         success_height_above_table=0.04,
                                         scale_version=REWARD_SCALE_VERSION,
                                         per_step_range=[0.0, 1.0], shaping_budget=SHAPING_BUDGET,
                                         success_reward=SUCCESS_REWARD, success_overrides_shaping=True,
                                         fixed_weight_ratios=FIXED_WEIGHT_RATIOS,
                                         fixed_weights=FIXED_WEIGHTS),
                      scheduler_config=dict(window_size=scheduler_options['window_size'], update_interval=scheduler_options['update_interval'],
                                            confirmations=scheduler_options['confirmations'], thresholds=[0.7, 0.7],
                                            grasp_stage_lift_ratio=scheduler_options.get('grasp_stage_lift_ratio', 0.3),
                                            weights=AdaptiveRewardScheduler(**scheduler_options).stage_weights),
                      reward_relabelling=condition == 'adaptive' and not args.historical_rewards)
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        replay_kwargs = {}
        if config['reward_relabelling']:
            # Relabel adaptive replay by default; other conditions use standard replay.
            replay_kwargs = dict(replay_buffer_class=ComponentReplayBuffer,
                                 replay_buffer_kwargs={'scheduler': task_for_scheduler.scheduler})
        if args.resume:
            # SB3 restores policy/optimizer; restore curriculum and input statistics below.
            model = SAC.load(args.resume, env=env, device=args.device,
                             tensorboard_log=str(run_dir / 'tensorboard'),
                             learning_rate=args.learning_rate, batch_size=args.batch_size,
                             buffer_size=args.buffer_size)
            if normalized_env is not None:
                if not hasattr(model, 'observation_stats'):
                    raise ValueError('Normalized checkpoint has no saved observation statistics')
                normalized_env.stats = model.observation_stats
            if condition == 'adaptive':
                restored = getattr(model, 'reward_scheduler', None)
                if restored is None and isinstance(model.replay_buffer, ComponentReplayBuffer):
                    restored = model.replay_buffer.scheduler
                if restored is None:
                    raise ValueError('Checkpoint has no curriculum history for adaptive resume')
                task_for_scheduler.scheduler = restored
                model.reward_scheduler = restored
                if isinstance(model.replay_buffer, ComponentReplayBuffer):
                    model.replay_buffer.scheduler = restored
                    model.replay_buffer_kwargs['scheduler'] = restored
            # Refill the new buffer before training; no hidden repeated updates
            # from one fresh transition after loading an already trained model.
            model.learning_starts = model.num_timesteps + args.learning_starts
            config.update(initial_timesteps=model.num_timesteps, replay_resumed=False,
                          additional_timesteps=args.timesteps,
                          source_condition=source_config['condition'])
        else:
            # All reward conditions share this network and SAC parameter configuration.
            model = SAC('MlpPolicy', env, learning_rate=args.learning_rate,
                        buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                        batch_size=args.batch_size, gamma=0.99, tau=0.005, ent_coef=args.ent_coef,
                        policy_kwargs={'net_arch': [256, 256]}, seed=args.seed,
                        device=args.device, verbose=1, tensorboard_log=str(run_dir / 'tensorboard'),
                        **replay_kwargs)
            if getattr(task_for_scheduler, 'scheduler', None) is not None:
                model.reward_scheduler = task_for_scheduler.scheduler
            config.update(initial_timesteps=0, replay_resumed=False)
        if normalized_env is not None:
            # SAC.save serializes this shared object alongside policy weights;
            # a selected checkpoint thus carries its matching input statistics.
            model.observation_stats = normalized_env.stats
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        model.grasp_hold_steps = args.grasp_hold_steps
        model.reward_profile = args.reward_profile
        from rewards.preparation import PARAMETERS
        config['preparation_parameters'] = PARAMETERS if args.reward_profile=='prepare_v1' else None
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        callbacks = [LiftMetricsCallback(run_dir / 'episode_metrics.csv')]
        if args.eval_freq:
            # Common sparse evaluation environment: adaptive weights never update here.
            eval_task = make_env('sparse', horizon=args.horizon, seed=args.eval_seed, grasp_hold_steps=args.grasp_hold_steps)
            if normalized_env is not None:
                eval_task = ObservationNormalization(eval_task, stats=normalized_env.stats, update=False)
            eval_env = Monitor(eval_task)
            callbacks.append(SuccessRateEvalCallback(
                eval_env, run_dir / 'best', eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes, seed=args.eval_seed,
                consecutive_passes=0, csv_path=run_dir / 'evaluations.csv'))
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not bool(args.resume))
        # Save the final policy; evaluation callbacks separately save best/latest checkpoints.
        model.save(run_dir / 'policy')
        if getattr(task_for_scheduler, 'scheduler', None) is not None:
            state = dict(stage=task_for_scheduler.scheduler.stage, episodes=task_for_scheduler.scheduler.episodes,
                         rates=task_for_scheduler.scheduler.get_rates(), weights=task_for_scheduler.scheduler.get_weights())
            (run_dir / 'scheduler.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f'Saved run to {run_dir}')
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == '__main__':
    main()
