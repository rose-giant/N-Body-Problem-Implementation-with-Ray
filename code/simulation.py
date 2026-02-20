# simulation.py
import ray
import time
import numpy as np
import math
from physics import compute_force_task  # your existing physics function

class Simulation:
    def __init__(self, bodies, dt, monitor=None):
        self.bodies = bodies
        self.dt = dt
        self.monitor = monitor
        self.communication_time = 0.0

    def run(self, steps):
        if self.monitor:
            self.monitor.start_total()

        total_start = time.perf_counter()

        for step in range(steps):
            step_start = time.perf_counter()

            # 1️⃣ Gather states
            states_start = time.perf_counter()
            states = ray.get([b.get_state.remote() for b in self.bodies])
            masses = {sid: mass for sid, mass, _, _ in states}
            positions = {sid: pos for sid, _, pos, _ in states}
            states_end = time.perf_counter()
            self.communication_time += (states_end - states_start)

            if self.monitor:
                self.monitor.analyze_step(masses, positions)

            # 2️⃣ Step actors (parallel execution)
            step_start_comm = time.perf_counter()
            futures = [b.step.remote(masses, positions, self.dt) for b in self.bodies]
            ray.get(futures)
            step_end_comm = time.perf_counter()
            self.communication_time += (step_end_comm - step_start_comm)

            step_duration = time.perf_counter() - step_start
            if self.monitor:
                self.monitor.record_step_time(step_duration)

            print(f"Step {step} completed in {step_duration:.6f} sec")

        total_end = time.perf_counter()
        total_runtime = total_end - total_start
        self.print_stats(total_runtime)

        if self.monitor:
            self.monitor.end_total()
            self.monitor.report()

    def print_stats(self, total_runtime):
        compute_times = ray.get([b.get_compute_time.remote() for b in self.bodies])
        total_compute = sum(compute_times)

        print("\n===== PERFORMANCE REPORT =====")
        print(f"Total runtime:        {total_runtime:.4f} sec")
        print(f"Total compute time:   {total_compute:.4f} sec")
        print(f"Total communication:  {self.communication_time:.4f} sec")
        other = total_runtime - total_compute - self.communication_time
        print(f"Other/overhead:       {other:.4f} sec")
        print("==============================\n")