"""Simulation over several devices gives one device's results: trials spread, or neurons partitioned."""

from pathlib import Path

import numpy as np
from devices import run_on

PROGRAM = '''
import json, sys
import jax, jax.numpy as jnp, numpy as np
from dew.training import MeshSpec
sys.path.insert(0, TESTS)
jax.config.update("jax_enable_x64", True)
from test_graph import nest_network
from sparx.dynamics import LeakyIntegrateAndFire, Exponential, Receptor
from sparx.graph import FixedProbability, Network, PoissonInput, Population, Projection, SpikeRaster, simulate

mesh = MeshSpec()  # every device on the data axis
out = {"devices": jax.device_count()}

# Neurons partitioned: NEST's 60-neuron recurrent network, 300 ms.
network, case = nest_network("iaf_psc_exp")
steps = 3000
drive = {"dc": np.broadcast_to(case["current"], (steps, len(case["current"])))}
result = simulate(network, network.init(jax.random.key(0)), duration=steps * 0.1, drive=drive,
                  monitors={"n": SpikeRaster("n")}, mesh=mesh)
out["neurons"] = np.flatnonzero(result.records["n"].ravel()).tolist()
v = result.variables["state"]["network"]["populations"]["n"]["point_neuron"].neuron.v
out["neuron_sharding"] = list(v.sharding.spec)
out["shards"] = len(v.addressable_shards)

# Trials spread: a Poisson-driven network, 8 trials of 50 ms.
net = Network((Population("a", 64, LeakyIntegrateAndFire(), {"ex": Receptor(Exponential(5.0))}),),
              (Projection("a", "a", FixedProbability(0.1), weight=20.0, delay=1.0, receptor="ex"),),
              inputs=(PoissonInput("a", rate=1000.0, weight=60.0, receptor="ex", count=5),), dt=0.1,
              dtype=jnp.float64)
result = simulate(net, net.init(jax.random.key(0)), duration=50.0, key=jax.random.key(1), trials=8,
                  monitors={"a": SpikeRaster("a")}, mesh=mesh)
out["trials"] = np.flatnonzero(result.records["a"].ravel()).tolist()
out["trial_shape"] = list(result.records["a"].shape)
v = result.variables["state"]["network"]["populations"]["a"]["point_neuron"].neuron.v
out["trial_sharding"] = list(v.sharding.spec)

# Both: trials over the data axis and each trial's neurons over fsdp.
if jax.device_count() % 2 == 0:
    result = simulate(net, net.init(jax.random.key(0)), duration=50.0, key=jax.random.key(1), trials=8,
                      monitors={"a": SpikeRaster("a")}, mesh=MeshSpec(fsdp=2))
    out["mixed"] = np.flatnonzero(result.records["a"].ravel()).tolist()
    v = result.variables["state"]["network"]["populations"]["a"]["point_neuron"].neuron.v
    out["mixed_sharding"] = list(v.sharding.spec)
print(json.dumps(out))
'''


def _run(devices: int) -> dict:
    return run_on(devices, PROGRAM.replace("TESTS", repr(str(Path(__file__).parent))), timeout=600)


def test_simulation_over_devices_equals_one_device():
    one, four = _run(1), _run(4)
    assert (one["devices"], four["devices"]) == (1, 4)
    assert len(one["neurons"]) > 150 and len(one["trials"]) > 1000
    assert four["neurons"] == one["neurons"]
    assert four["shards"] == 4 and four["neuron_sharding"] == ["data"]  # 15 neurons each
    assert four["trial_sharding"] == ["data"]  # two trials each, every neuron
    assert four["mixed"] == one["trials"]
    assert four["mixed_sharding"] == ["data", "fsdp"]  # four trials and 32 neurons each
    assert four["trials"] == one["trials"]
    assert one["trial_shape"] == [8, 500, 64]
    # Trials differ from one another: each has its own noise.
    spikes = np.zeros(8 * 500 * 64)
    spikes[one["trials"]] = 1
    per_trial = spikes.reshape(8, -1)
    assert not np.array_equal(per_trial[0], per_trial[1])
