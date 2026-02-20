# body.py
import ray
import numpy as np
import time
from physics import compute_net_force, integrate_euler

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

    def step_with_force(self, total_force, dt):
        start = time.perf_counter()
        self.position, self.velocity = integrate_euler(
            self.position, self.velocity, total_force, self.mass, dt
        )
        end = time.perf_counter()
        self.compute_time += end - start

    def get_compute_time(self):
        return self.compute_time

    def get_position(self):
        return self.position
