import flax.linen as nn
import jax
import jax.numpy as jnp
import nir

from sparx.nn import LI, LIF
from sparx.nir import from_nir, to_nir

pilot = nn.Sequential([nn.Dense(64), LIF(tau=3.0, reset="zero"),
                       nn.Dense(64), LIF(tau=3.0, reset="zero"),
                       nn.Dense(2), LI(tau=5.0)])
variables = pilot.init(jax.random.key(0), jnp.zeros((1, 1, 7)))
graph = to_nir(pilot, variables, dt=0.01)          # a step is 10 ms; NIR counts seconds
nir.write("pilot.nir", graph)

model, read = from_nir(nir.read("pilot.nir"), dt=0.01)
x = jax.random.normal(jax.random.key(1), (200, 8, 7))
print(jnp.abs(pilot.apply(variables, x) - model.apply(read, x)).max())   # 0.0
