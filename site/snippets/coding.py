import jax
import jax.numpy as jnp

from sparx.encode import DeltaEncoder, LatencyEncoder, RateEncoder

values = jnp.array([[0.95, 0.62, 0.3, 0.05]])     # one record, [B, 4]
key = jax.random.key(0)

rate = RateEncoder(steps=16)(key, values)         # [16, 1, 4]
latency = LatencyEncoder(steps=16)(key, values)   # [16, 1, 4]
print("rate:", rate.sum(0)[0], "spikes; reads", rate.mean(0)[0])
when = jnp.argmax(latency, axis=0)[0]
print("latency fires at", when, "; reads", 1 - when / 15)

# A signal over time, [B, T, 1]: a slow ramp, then a jump
signal = jnp.concatenate([jnp.linspace(0.0, 0.5, 40),
                          jnp.full(20, 0.9)])[None, :, None]
events = DeltaEncoder(threshold=0.1, off_spikes=True)(key, signal)
print("delta events at steps", jnp.flatnonzero(events[:, 0, 0]))
