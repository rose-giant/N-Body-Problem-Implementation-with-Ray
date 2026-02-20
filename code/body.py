# body.py
import ray
import numpy as np
import time
from physics import compute_gravitational_force, integrate_euler


@ray.remote(num_cpus=1)
class Body:
    def __init__(self, body_id, mass, position, velocity):
        self.body_id = body_id
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.compute_time = 0.0

    def get_state(self):
        return self.body_id, self.mass, self.position, self.velocity

    def step(self, masses, positions, dt):
        """
        Compute total force on self from all bodies.
        This is done in a single call to avoid task overhead.
        """
        start = time.perf_counter()
        total_force = np.zeros(3)
        for other_id, other_mass in masses.items():
            if other_id == self.body_id:
                continue
            total_force += compute_gravitational_force(
                self.mass,
                self.position,
                other_mass,
                positions[other_id]
            )
        # Update position and velocity
        self.position, self.velocity = integrate_euler(
            self.position, self.velocity, total_force, self.mass, dt
        )
        end = time.perf_counter()
        self.compute_time += (end - start)

    def get_compute_time(self):
        return self.compute_time
    def get_position(self):
        return self.position
