# main.py
import ray
import numpy as np
from body import Body
from simulation import Simulation
from monitor import Monitor


def create_bodies():
    bodies = []

    # 🌞 Sun
    sun = Body.remote(
        "sun",
        mass=1.989e30,
        position=[0, 0, 0],
        velocity=[0, 0, 0],
    )
    bodies.append(sun)

    # 🪐 5 planets
    for i in range(5):
        distance = 1.5e11 + i * 2e10
        velocity = np.sqrt(6.67430e-11 * 1.989e30 / distance)

        planet = Body.remote(
            f"planet_{i}",
            mass=5.972e24,
            position=[distance, 0, 0],
            velocity=[0, velocity, 0],
        )
        bodies.append(planet)

    # ☄️ Comet
    comet = Body.remote(
        "comet",
        mass=1e14,
        position=[3e11, 0, 0],
        velocity=[0, 15000, 10000],
    )
    bodies.append(comet)

    return bodies


def main():
    ray.init(num_cpus=8)

    dt = 60 * 60
    monitor = Monitor(sun_id="sun")

    bodies = create_bodies()

    sim = Simulation(bodies, dt, monitor=monitor)
    sim.run(steps=100)


if __name__ == "__main__":
    main()