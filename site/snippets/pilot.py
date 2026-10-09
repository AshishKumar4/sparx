import flax.linen as nn
import jax
import jax.numpy as jnp

from sparx.nn import LI, LIF

# 7 readings in: the way to the target, velocity, attitude, spin.
# 2 membranes out: the rotors' thrust.
pilot = nn.Sequential([
    nn.Dense(64), LIF(tau=3.0, reset="zero"),
    nn.Dense(64), LIF(tau=3.0, reset="zero"),
    nn.Dense(2), LI(tau=5.0),
])


def step(params, carried, readings):
    """10 ms of the network, its membranes carried in `state`."""
    out, mutated = pilot.apply(
        {"params": params, "state": carried}, readings[None],
        mutable=["state"])
    return out[0], mutated["state"]


params = pilot.init(jax.random.key(0), jnp.zeros((1, 1, 7)))["params"]
# 256 drones, every neuron at rest
membranes, carried = step(params, {}, jnp.zeros((256, 7)))
# Training scans step() and the drone's physics over 2 s of flight
# and takes jax.grad of a cost led by the distance to the target:
# through the spikes by their surrogate, and through the physics.
