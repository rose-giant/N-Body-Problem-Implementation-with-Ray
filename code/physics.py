# physics.py
import numpy as np
import ray

G = 6.67430e-11  # gravitational constant

def compute_gravitational_force(m1, p1, m2, p2, softening=1e-5):
    """
    Returns force vector exerted on body1 by body2.
    """
    r_vec = p2 - p1
    distance = np.linalg.norm(r_vec) + softening
    force_magnitude = G * m1 * m2 / (distance ** 2)
    force_direction = r_vec / distance
    return force_magnitude * force_direction


def compute_net_force(body_id, masses, positions):
    """
    Compute total force on body_id from all other bodies.
    """
    net_force = np.zeros(3)
    for other_id in masses:
        if other_id == body_id:
            continue
        net_force += compute_gravitational_force(
            masses[body_id],
            positions[body_id],
            masses[other_id],
            positions[other_id],
        )
    return net_force


def integrate_euler(position, velocity, force, mass, dt):
    """
    Simple Euler integration step.
    """
    acceleration = force / mass
    new_velocity = velocity + acceleration * dt
    new_position = position + new_velocity * dt
    return new_position, new_velocity


@ray.remote
def compute_force_task(body_state, other_state):
    """
    Compute force exerted on body_state by other_state.
    body_state and other_state = (id, mass, position, velocity)
    """
    _, m1, p1, _ = body_state
    _, m2, p2, _ = other_state
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return compute_gravitational_force(m1, p1, m2, p2)