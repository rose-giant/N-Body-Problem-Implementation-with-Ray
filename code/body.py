# body.py
import ray
import numpy as np
from physics import compute_net_force, integrate_euler


@ray.remote
class Body:
    def __init__(self, body_id, mass, position, velocity):
        self.body_id = body_id
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)

    def get_state(self):
        return self.body_id, self.mass, self.position, self.velocity

    def step(self, masses, positions, dt):
        force = compute_net_force(self.body_id, masses, positions)
        self.position, self.velocity = integrate_euler(
            self.position,
            self.velocity,
            force,
            self.mass,
            dt,
        )

    def get_position(self):
        return self.position
