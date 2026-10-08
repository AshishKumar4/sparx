# Train, serve and export

One spiking classifier from data to deployment: trained on dew's `Trainer`, loaded back in a fresh process, served to many streams at once, and written to NIR for other simulators and neuromorphic hardware. Each step takes the same model, a flax `nn.Sequential` of sparx layers.

## Data

Event data arrive as spikes over time. Here three classes of recordings differ in which group of ten channels fires more; any dataset of `[N, T, C]` spike counts and integer labels, such as `sparx.datasets.shd`, takes the same path.

```python
import numpy as np


def recordings(count, seed, steps=40, channels=30):
    """Spike trains of `channels` inputs over `steps` steps; one group of ten fires more, the label."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, count)
    rate = np.full((count, 1, channels), 0.02)
    for label in range(3):
        rate[labels == label, :, 10 * label:10 * label + 10] = 0.2
    spikes = (rng.random((count, steps, channels)) < rate).astype(np.uint8)
    return {"spikes": spikes, "label": labels.astype(np.int32)}


train, test = recordings(2048, 0), recordings(256, 1)
```

## Train

A dense layer feeds 32 leaky integrate-and-fire neurons, and a second feeds a leaky integrator per class, whose largest membrane over time scores it (`readout="max"`). `reset="zero"` is NIR's reset, so the network exports as it trained.

```python
import flax.linen as nn
import optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset

from sparx.encode import EventsEncoder
from sparx.metrics import Accuracy
from sparx.nn import LI, LIF
from sparx.objectives import SpikingClassifierObjective

stack = nn.Sequential([nn.Dense(32), LIF(tau=5.0, reset="zero"), nn.Dense(3), LI(tau=5.0)])
objective = SpikingClassifierObjective(stack, Field("spikes", (40, 30)), EventsEncoder(), readout="max")
data = Dataset.from_records(train, batch=64, validation=test)
trainer = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=Checkpoints("runs/events"))
state = trainer.fit(data, steps=150, eval_every=150, metrics=[Accuracy()], validation={"test": data.val})
```

The trainer prints the loss, each layer's firing rate and the test accuracy, and checkpoints the run with a record of the model, encoder and readout.

## Load

`dew.pipeline` rebuilds the classifier from the run directory, in this process or another. `trust` lets the run's record import sparx's classes, as `trust_remote_code` does in transformers:

```python
import dew

classifier = dew.pipeline("runs/events", trust=("sparx",))
accuracy = np.mean(np.asarray(classifier(test["spikes"])) == test["label"])   # 1.0 on this data
```

## Serve

A deployed classifier sees a recording as it arrives, a few steps at a time. `StreamServer` keeps each session's neuron state in a row of one batch and runs one frame of every waiting session in one compiled call. A session's outputs over its frames equal one call over its whole stream:

```python
from sparx.serve import StreamServer

server = StreamServer(classifier.model, classifier.variables, call=classifier.call, slots=4, frame=10,
                      sample_shape=(30,))
session = server.open()
recording = test["spikes"][0].astype(np.float32)          # [40, 30], time first
futures = [server.submit(session, recording[t:t + 10]) for t in range(0, 40, 10)]
server.run()
streamed = np.concatenate([future.result() for future in futures])   # [40, 3], the membranes
predicted = int(streamed.max(axis=0).argmax())                         # the readout over the stream
```

## Export

NIR (Pedersen et al. 2024) describes the network in continuous time, a format snnTorch, Norse, Lava, Rockpool and several neuromorphic platforms support. `to_nir` names the step in seconds, and `from_nir` reads the graph back:

```python
import nir

from sparx.nir import from_nir, to_nir

graph = to_nir(classifier.model, classifier.variables, dt=1e-3)
nir.write("events.nir", graph)
back, back_variables = from_nir(nir.read("events.nir"), dt=1e-3)
x = np.moveaxis(test["spikes"][:8], 1, 0).astype(np.float32)              # [T, B, C]
same = np.array_equal(back.apply(back_variables, x), classifier.model.apply(classifier.variables, x))
```

The graph holds an `Affine`, a `LIF`, an `Affine` and an `LI` node, and the network read back computes the trained one's membranes exactly. [The NIR module's documentation](../../src/sparx/nir.py) lists what exports: dense and convolutional layers, `Flatten`, `LIF` and `IF` with NIR's reset, `LI`, and `Recurrent(LIF)`.
