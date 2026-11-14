"""Episode diagnostics that do not change rewards or native success criteria."""
import numpy as np


class SkillDiagnostics:
    def __init__(self, contact_steps=5, lift_steps=10):
        if min(contact_steps, lift_steps) < 1:
            raise ValueError('Hold lengths must be positive')
        self.contact_steps, self.lift_steps = contact_steps, lift_steps
        self.reset()

    def reset(self):
        self.contact_run = self.held_run = 0
        self.max_contact_run = self.max_held_run = 0
        self.ever_contact = self.lift_after_contact = False
        self.first_contact_step = None
        self.steps = self.post_contact_steps = self.upward_steps = 0
        self.max_height = 0.0

    def update(self, grasped, lifted, height, action):
        self.steps += 1
        if grasped and not self.ever_contact:
            self.first_contact_step = self.steps
        self.ever_contact |= bool(grasped)
        self.contact_run = self.contact_run + 1 if grasped else 0
        self.held_run = self.held_run + 1 if grasped and lifted else 0
        self.max_contact_run = max(self.max_contact_run, self.contact_run)
        self.max_held_run = max(self.max_held_run, self.held_run)
        if self.ever_contact:
            self.post_contact_steps += 1
            self.upward_steps += int(float(action[2]) > 0.05)
            self.max_height = max(self.max_height, float(height))
            # Requires current contact too: a later collision bounce is not held Lift.
            self.lift_after_contact |= bool(grasped and lifted)

    def summary(self):
        return dict(stable_grasp=self.max_contact_run >= self.contact_steps,
                    stable_lift=self.max_held_run >= self.lift_steps,
                    max_contact_steps=self.max_contact_run,
                    max_held_lift_steps=self.max_held_run,
                    first_contact_step=self.first_contact_step,
                    grasped_lift_after_contact=self.lift_after_contact,
                    max_height_after_contact=self.max_height if self.ever_contact else None,
                    upward_action_fraction_after_contact=(self.upward_steps / self.post_contact_steps
                                                           if self.post_contact_steps else None))
