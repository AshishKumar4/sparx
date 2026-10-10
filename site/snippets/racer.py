import jax
import jax.numpy as jnp

import sparx
from sparx.surrogate import ATan

threshold, event = 0.15, ATan(alpha=8.0)  # the racer's pixels


def brightness(x):
    """A road's edge: dark road, then a light verge past 0.8 m."""
    return 0.12 + 0.7 * jax.nn.sigmoid((x - 0.8) / 0.12)


def on_events(shift):
    """ON events from 9 pixels across the edge when the car moves
    `shift` metres toward the verge."""
    pixels = jnp.linspace(0.0, 1.6, 9)        # metres from the middle
    before = jnp.log(brightness(pixels))
    after = jnp.log(brightness(pixels + shift))
    return sparx.spike(after - before - threshold, event).sum()


for shift in (0.05, 0.1, 0.2, 0.4):
    count, slope = jax.value_and_grad(on_events)(shift)
    print(f"move {shift:.2f} m: {int(count)} ON events,"
          f" gradient {float(slope):.1f} per m")
