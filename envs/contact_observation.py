"""Expose contact-duration memory when rewards depend on sustained contact."""
import gymnasium as gym
import numpy as np


class ContactObservation(gym.ObservationWrapper):
    def __init__(self, env, target_steps):
        super().__init__(env)
        self.target_steps=target_steps
        self.observation_space=gym.spaces.Box(
            np.append(env.observation_space.low,0).astype(np.float32),
            np.append(env.observation_space.high,1).astype(np.float32),dtype=np.float32)

    def observation(self, observation):
        duration=min(self.env.diagnostics.contact_run/self.target_steps,1.)
        return np.append(observation,duration).astype(np.float32)

    def task_state(self):
        return self.env.task_state()

    def task_geometry(self):
        return self.env.task_geometry()

    @property
    def base_env(self):
        return self.env.base_env

    @property
    def episode_reached(self):
        return self.env.episode_reached

    @property
    def episode_grasped(self):
        return self.env.episode_grasped

    @property
    def episode_lifted(self):
        return self.env.episode_lifted
