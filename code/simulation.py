# simulation.py
import ray
import time
import numpy as np
from physics import compute_gravitational_force

@ray.remote
def compute_chunk_forces(target_state, chunk_states):
    """
    Compute total force on target_state from a chunk of other bodies.
    """
    _, m1, p1, _ = target_state
    total_force = np.zeros(3)
    for _, m2, p2, _ in chunk_states:
        total_force += compute_gravitational_force(m1, p1, m2, p2)
    return total_force

class Simulation:
    def __init__(self, bodies, dt, monitor=None, chunk_size=2):
        self.bodies = bodies
        self.dt = dt
        self.monitor = monitor
        self.chunk_size = chunk_size
        self.communication_time = 0.0

    def run(self, steps):
        if self.monitor:
            self.monitor.start_total()

        total_start = time.perf_counter()

        for step in range(steps):
            step_start = time.perf_counter()

            # 1️⃣ Gather all states
            comm_start = time.perf_counter()
            states = ray.get([b.get_state.remote() for b in self.bodies])
            comm_end = time.perf_counter()
            self.communication_time += comm_end - comm_start

            id_to_state = {sid: state for sid, *state in states}

            if self.monitor:
                masses = {sid: state[0] for sid, state in id_to_state.items()}
                positions = {sid: state[1] for sid, state in id_to_state.items()}
                self.monitor.analyze_step(masses, positions)

            # 2️⃣ Submit chunked tasks for each body
            comm_start = time.perf_counter()
            all_futures = []
            for i, body in enumerate(self.bodies):
                target_state = states[i]
                others = [s for j, s in enumerate(states) if j != i]

                # split others into chunks
                chunks = [others[k:k+self.chunk_size] for k in range(0, len(others), self.chunk_size)]

                # submit one task per chunk
                futures = [compute_chunk_forces.remote(target_state, chunk) for chunk in chunks]
                all_futures.append((body, futures))
            comm_end = time.perf_counter()
            self.communication_time += comm_end - comm_start

            # 3️⃣ Aggregate forces and update actors
            comm_start = time.perf_counter()
            for body, futures in all_futures:
                forces = ray.get(futures)
                total_force = np.sum(forces, axis=0)
                body.step_with_force.remote(total_force, self.dt)
            comm_end = time.perf_counter()
            self.communication_time += comm_end - comm_start

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