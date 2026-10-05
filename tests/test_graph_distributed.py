"""Simulation over several devices gives one device's results: trials spread, or neurons partitioned."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROGRAM = '''
import json, sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh
sys.path.insert(0, TESTS)
jax.config.update("jax_enable_x64", True)
from test_graph import nest_network
from sparx.dynamics import LIF, Exponential, Receptor
from sparx.graph import FixedProbability, Network, PoissonInput, Population, Projection, Spikes, simulate

mesh = Mesh(np.array(jax.devices()), ("x",)) if jax.device_count() > 1 else None
out = {"devices": jax.device_count()}

# Neurons partitioned: NEST's 60-neuron recurrent network, 300 ms.
network, case = nest_network("iaf_psc_exp")
steps = 3000
drive = {"dc": np.broadcast_to(case["current"], (steps, len(case["current"])))}
result = simulate(network, network.init(jax.random.key(0)), duration=steps * 0.1, drive=drive,
                  monitors=(Spikes("n"),), mesh=mesh)
out["neurons"] = np.flatnonzero(result.records[0].ravel()).tolist()
v = result.variables["state"]["network"]["populations"]["n"]["cell"].neuron.v
out["neuron_sharding"] = str(v.sharding.spec) if mesh is not None else None
out["shards"] = len(v.addressable_shards)

# Trials spread: a Poisson-driven network, 8 trials of 50 ms.
net = Network((Population("a", 64, LIF(), {"ex": Receptor(Exponential(5.0))}),),
              (Projection("a", "a", FixedProbability(0.1), weight=20.0, delay=1.0),),
              inputs=(PoissonInput("a", rate=1000.0, weight=60.0, count=5),), dt=0.1, dtype=jnp.float64)
result = simulate(net, net.init(jax.random.key(0)), duration=50.0, key=jax.random.key(1), trials=8,
                  monitors=(Spikes("a"),), mesh=mesh)
out["trials"] = np.flatnonzero(result.records[0].ravel()).tolist()
out["trial_shape"] = list(result.records[0].shape)
print(json.dumps(out))
'''


def _run(devices: int) -> dict:
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    program = PROGRAM.replace("TESTS", repr(str(Path(__file__).parent)))
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_simulation_over_devices_equals_one_device():
    one, four = _run(1), _run(4)
    assert (one["devices"], four["devices"]) == (1, 4)
    assert len(one["neurons"]) > 150 and len(one["trials"]) > 1000
    assert four["neurons"] == one["neurons"]
    assert four["shards"] == 4 and "'x'" in four["neuron_sharding"]  # each device holds 15 neurons
    assert four["trials"] == one["trials"]
    assert one["trial_shape"] == [8, 500, 64]
    # Trials differ from one another: each has its own noise.
    spikes = np.zeros(8 * 500 * 64)
    spikes[one["trials"]] = 1
    per_trial = spikes.reshape(8, -1)
    assert not np.array_equal(per_trial[0], per_trial[1])
