# monitor.py
import numpy as np
import time
from physics import compute_gravitational_force


class Monitor:
    def __init__(self, sun_id="sun"):
        self.sun_id = sun_id
        self.history = []

        # Timing
        self.step_times = []
        self.total_start = None
        self.total_end = None

    # ---- Timing control ----
    def start_total(self):
        self.total_start = time.perf_counter()

    def end_total(self):
        self.total_end = time.perf_counter()

    def record_step_time(self, duration):
        self.step_times.append(duration)

    # ---- Physics analysis ----
    def analyze_step(self, masses, positions):
        sun_pos = positions[self.sun_id]
        sun_mass = masses[self.sun_id]

        step_metrics = {}

        for body_id in masses:
            if body_id == self.sun_id:
                continue

            body_pos = positions[body_id]
            body_mass = masses[body_id]

            F_sun = compute_gravitational_force(
                body_mass, body_pos,
                sun_mass, sun_pos
            )

            F_total = np.zeros(3)
            for other_id in masses:
                if other_id == body_id:
                    continue
                F_total += compute_gravitational_force(
                    body_mass, body_pos,
                    masses[other_id], positions[other_id]
                )

            dominance_ratio = np.linalg.norm(F_sun) / (
                np.linalg.norm(F_total) + 1e-12
            )

            distance_to_sun = np.linalg.norm(body_pos - sun_pos)

            step_metrics[body_id] = {
                "dominance_ratio": dominance_ratio,
                "distance_to_sun": distance_to_sun,
            }

        self.history.append(step_metrics)

    # ---- Reporting ----
    def report(self):
        print("\n=== Monitoring Report ===")

        # Timing summary
        total_time = self.total_end - self.total_start
        avg_step = np.mean(self.step_times)
        print(f"Total execution time: {total_time:.4f} sec")
        print(f"Average step time: {avg_step:.6f} sec")
        print(f"Max step time: {np.max(self.step_times):.6f} sec")

        # Physics warnings
        for step_data in self.history:
            for body_id, metrics in step_data.items():
                if metrics["dominance_ratio"] < 0.9:
                    print(
                        f"Warning: {body_id} sun dominance dropped to "
                        f"{metrics['dominance_ratio']:.3f}"
                    )
                # else:
                #     print(f"body#{body_id}: {metrics}")
