import flax.linen as nn
import jax
import jax.numpy as jnp

from sparx.nn import LI, LIF

pilot = nn.Sequential([
    nn.Dense(64), LIF(tau=3.0, reset="zero"),     # 7 readings in: the way to the target,
    nn.Dense(64), LIF(tau=3.0, reset="zero"),     # velocity, attitude and spin
    nn.Dense(2), LI(tau=5.0),                     # 2 membranes out: the rotors' thrust
])


def step(params, carried, readings):
    """10 ms of the network, its membranes carried in the `state` collection."""
    out, mutated = pilot.apply({"params": params, "state": carried}, readings[None], mutable=["state"])
    return out[0], mutated["state"]


params = pilot.init(jax.random.key(0), jnp.zeros((1, 1, 7)))["params"]
membranes, carried = step(params, {}, jnp.zeros((256, 7)))   # 256 drones, every neuron at rest
# Training scans step() and the drone's physics over 2 s of flight and takes jax.grad of the
# distance to the target: through the spikes by their surrogate, and through the physics.
