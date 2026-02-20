# simulation.py
import ray
import time
import numpy as np
import math
from physics import compute_force_task  # your existing physics function

class Simulation:
    def __init__(self, bodies, dt, monitor=None, chunk_size=None):
        self.bodies = bodies
        self.dt = dt
        self.monitor = monitor
        self.communication_time = 0.0
        self.chunk_size = chunk_size  # optional chunking for task parallelism

    def run(self, steps):
        if self.monitor:
            self.monitor.start_total()

        total_start = time.perf_counter()

        for step in range(steps):
            step_start = time.perf_counter()

            # 1️⃣ Get current states from all actors
            states = ray.get([b.get_state.remote() for b in self.bodies])
            masses = {sid: mass for sid, mass, _, _ in states}
            positions = {sid: pos for sid, _, pos, _ in states}

            if self.monitor:
                self.monitor.analyze_step(masses, positions)

            # 2️⃣ Compute forces using tasks
            comm_start = time.perf_counter()

            all_force_futures = []

            for i, body in enumerate(self.bodies):
                body_state = states[i]
                others = [s for j, s in enumerate(states) if j != i]

                # Optional chunking
                if self.chunk_size:
                    chunks = [others[k:k+self.chunk_size] for k in range(0, len(others), self.chunk_size)]
                else:
                    chunks = [others]

                # Submit tasks
                body_force_futures = [
                    compute_force_task.remote(body_state, other)
                    for chunk in chunks
                    for other in chunk
                ]
                all_force_futures.append((body, body_force_futures))

            # Wait and sum
            for body, futures in all_force_futures:
                forces = ray.get(futures)
                total_force = np.sum(forces, axis=0)
                body.step_with_force.remote(total_force, self.dt)

            comm_end = time.perf_counter()
            self.communication_time += (comm_end - comm_start)

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
        # Get compute time from each actor
        compute_times = ray.get(
            [body.get_compute_time.remote() for body in self.bodies]
        )

        total_compute = sum(compute_times)

        print("\n===== PERFORMANCE REPORT =====")
        print(f"Total runtime:        {total_runtime:.4f} sec")
        print(f"Total compute time:   {total_compute:.4f} sec")
        print(f"Total communication:  {self.communication_time:.4f} sec")

        other = total_runtime - total_compute - self.communication_time
        print(f"Other/overhead:       {other:.4f} sec")
        print("==============================\n")