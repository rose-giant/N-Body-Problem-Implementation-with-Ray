# simulation.py
import ray
import time

class Simulation:
    def __init__(self, bodies, dt, monitor=None):
        self.bodies = bodies
        self.dt = dt
        self.monitor = monitor

    def run(self, steps):
        if self.monitor:
            self.monitor.start_total()

        for step in range(steps):
            step_start = time.perf_counter()

            states = ray.get([b.get_state.remote() for b in self.bodies])
            masses = {sid: mass for sid, mass, _, _ in states}
            positions = {sid: pos for sid, _, pos, _ in states}

            if self.monitor:
                self.monitor.analyze_step(masses, positions)

            futures = [
                b.step.remote(masses, positions, self.dt)
                for b in self.bodies
            ]
            ray.get(futures)

            step_duration = time.perf_counter() - step_start

            if self.monitor:
                self.monitor.record_step_time(step_duration)

            print(f"Step {step} completed in {step_duration:.6f} sec")

        if self.monitor:
            self.monitor.end_total()
            self.monitor.report()
