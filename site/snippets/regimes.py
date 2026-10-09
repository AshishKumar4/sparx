import jax
import numpy as np

from sparx.graph import SpikeRaster, simulate
from sparx.graph.models import brunel
from sparx.spiketrains import cv_isi, population_fano, rates_hz

regimes = {"synchronous regular": (3.0, 2.0),
           "asynchronous irregular": (5.0, 2.0),
           "synchronous, fast": (6.0, 4.0),
           "synchronous, slow": (4.5, 0.9)}
for name, (g, eta) in regimes.items():
    network = brunel(250, g=g, eta=eta)          # 1,250 LIF neurons
    result = simulate(network, network.init(jax.random.key(0)),
                      duration=600.0, key=jax.random.key(1),
                      monitors={"e": SpikeRaster("e")})
    spikes = np.asarray(result.records["e"])[2000:]  # after 200 ms
    print(f"{name}: {rates_hz(spikes, 0.1).mean():.1f} Hz, "
          f"CV {np.median(cv_isi(spikes)):.2f}, "
          f"Fano {population_fano(spikes, 0.1):.1f}")
