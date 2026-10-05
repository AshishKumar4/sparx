"""Training through dew's mesh gives the result one device gives."""

import json
import os
import subprocess
import sys

import numpy as np

PROGRAM = '''
import json, os
import jax, numpy as np, optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset, Loading
from dew.training import MeshSpec
import sparx
from sparx.dew import RateBand, SpikingClassifier
from sparx.encode import Direct
from sparx.models import SpikingMLP

rng = np.random.default_rng(0)
labels = rng.integers(0, 3, 64).astype(np.int32)
images = (rng.uniform(size=(64, 6, 6, 1)) * 255).astype(np.uint8)
images[labels == 1, :3] = 255
net = SpikingMLP(hidden=(32, 16), classes=3, neuron=sparx.nn.ALIF(tau=3.0, tau_adapt=10.0, learn_tau=True),
                 recurrent=True)
objective = SpikingClassifier(net, Field("image", (6, 6, 1)), Direct(steps=5), readout="max",
                              rates=RateBand(0.02, 0.4))
data = Dataset.from_records({"image": images, "label": labels}, batch=32,
                            loading=Loading(workers=0, threads=1, read_buffer=1))
trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0), mesh=MeshSpec(fsdp=FSDP))
state = trainer.fit(data, steps=6, log_every=6)
leaves = {jax.tree_util.keystr(path): np.asarray(leaf).tolist()
          for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])}
print(json.dumps({"devices": jax.device_count(), "params": leaves}))
'''


def _train(devices: int, fsdp: int) -> dict:
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    done = subprocess.run([sys.executable, "-c", PROGRAM.replace("FSDP", str(fsdp))], capture_output=True,
                          text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_eight_devices_train_the_parameters_one_device_trains():
    one = _train(1, 1)
    eight = _train(8, 2)  # data 4 x fsdp 2: batch rows split four ways, parameters two ways
    assert (one["devices"], eight["devices"]) == (1, 8)
    assert set(one["params"]) == set(eight["params"])
    for name, value in one["params"].items():
        # The batch mean is summed in a different order across devices, so
        # float32 rounding differs; observed at most 1.8e-7 after six steps.
        np.testing.assert_allclose(eight["params"][name], value, rtol=1e-5, atol=2e-6, err_msg=name)
