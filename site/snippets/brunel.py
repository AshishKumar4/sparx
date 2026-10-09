import jax

from sparx.graph import PopulationRate, SpikeRaster, simulate
from sparx.graph.models import brunel

network = brunel(250, g=5.0, eta=2.0)        # 1,250 LIF neurons
monitors = {"spikes": SpikeRaster("e"), "rate": PopulationRate("e")}
result = simulate(network, network.init(jax.random.key(0)),
                  duration=400.0, key=jax.random.key(1),
                  monitors=monitors)

# [4000, 1000]: one row of booleans per 0.1 ms step
spikes = result.records["spikes"]
print(float(result.records["rate"][1000:].mean()), "Hz")
