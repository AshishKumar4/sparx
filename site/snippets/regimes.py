import jax
import numpy as np

from sparx.graph import SpikeRaster, simulate
from sparx.graph.models import brunel
from sparx.spiketrains import cv_isi, population_fano, rates_hz


def measure(g, eta, j, order=250):            # 5 * order neurons
    network = brunel(order, g=g, eta=eta, j=j)
    result = simulate(network, network.init(jax.random.key(0)),
                      duration=600.0, key=jax.random.key(1),
                      monitors={"e": SpikeRaster("e")})
    spikes = np.asarray(result.records["e"])[2000:]  # after 200 ms
    return (f"{rates_hz(spikes, 0.1).mean():5.1f} Hz, "
            f"CV {np.median(cv_isi(spikes)):.2f}, "
            f"Fano {population_fano(spikes, 0.1):5.1f}")


# Brunel's synapses, 0.1 mV, but each neuron has 100 inputs, not 1,000
print("g 5, eta 2, j 0.1:", measure(5.0, 2.0, 0.1))
# Synapses ten times stronger: 100 inputs of 1 mV, as his 1,000 of 0.1
for g, eta in [(3.0, 2.0), (5.0, 2.0), (6.0, 4.0), (4.5, 0.9)]:
    print(f"g {g}, eta {eta}, j 1.0:", measure(g, eta, 1.0))
