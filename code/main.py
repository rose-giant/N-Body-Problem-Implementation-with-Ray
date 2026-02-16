# main.py
import ray
import numpy as np
from body import Body
from simulation import Simulation
from monitor import Monitor

ray.init()

dt = 60 * 60  # 1 hour timestep

bodies = []

# 🌞 Sun
sun = Body.remote(
    "sun",
    mass=1.989e30,
    position=[0, 0, 0],
    velocity=[0, 0, 0],
)
bodies.append(sun)

# 🪐 5 planets (circular-ish orbits)
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

# ☄️ Comet (eccentric)
comet = Body.remote(
    "comet",
    mass=1e14,
    position=[3e11, 0, 0],
    velocity=[0, 15000, 10000],
)
bodies.append(comet)

monitor = Monitor(sun_id="sun")

sim = Simulation(bodies, dt, monitor=monitor)
sim.run(steps=1000)
