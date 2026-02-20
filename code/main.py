# main.py
import ray
import numpy as np
from body import Body
from simulation import Simulation

if __name__ == "__main__":
    ray.init(num_cpus=8)

    dt = 60*60  # 1 hour
    bodies = []

    # Sun
    sun = Body.remote("sun", 1.989e30, [0,0,0], [0,0,0])
    bodies.append(sun)

    # 5 planets
    for i in range(5):
        distance = 1.5e11 + i*2e10
        velocity = np.sqrt(6.67430e-11*1.989e30 / distance)
        planet = Body.remote(f"planet_{i}", 5.972e24, [distance,0,0], [0, velocity,0])
        bodies.append(planet)

    # 1 comet
    comet = Body.remote("comet", 1e14, [3e11,0,0], [0,15000,10000])
    bodies.append(comet)

    sim = Simulation(bodies, dt, chunk_size=2)  # adjust chunk_size for more bodies
    sim.run(steps=500)