"""Train a spiking network on the Spiking Heidelberg Digits with dew's Trainer.

    pip install -e ".[dew,datasets]"
    python examples/train_shd.py --steps 3000

Downloads SHD (169 MB) into ~/.cache/sparx on first use, bins it into 100
steps of 14 ms over 700 channels, and trains a layer of adaptive LIF neurons
with a leaky integrator readout scored by its maximum over time. The test
split serves as the validation set (SHD has no separate one) and the script
prints the accuracy over all 2264 test recordings at the end.

`--recurrent` feeds the hidden layer's spikes back to itself. Backpropagation
through that loop explodes with the heavy-tailed ATan surrogate once the
recurrent weights grow (gradient norms past 1e8 within 300 steps), so pair it
with `--surrogate superspike`, whose derivative falls off steeply.
"""

import argparse
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dew import Checkpoints, Field, Trainer
from dew.data import Dataset

import sparx
from sparx.datasets import shd
from sparx.dew import Events, RateBand, SpikingClassifier, accuracy


class Net(nn.Module):
    """700 channels -> adaptive LIF, optionally recurrent -> 20-class leaky integrator readout."""

    hidden: int
    tau: float
    tau_adapt: float
    dropout: float
    recurrent: bool
    surrogate: sparx.surrogate.Surrogate

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.hidden)(x)
        neuron = sparx.nn.ALIF(tau=self.tau, tau_adapt=self.tau_adapt, beta=0.2, learn_tau=True,
                               detach_reset=True, surrogate=self.surrogate)
        x = sparx.nn.Recurrent(neuron)(x) if self.recurrent else neuron(x)
        x = nn.Dropout(self.dropout, deterministic=not train)(x)
        return sparx.nn.LI(tau=self.tau, learn_tau=True)(nn.Dense(20)(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--tau", type=float, default=5.0, help="membrane time constant, in 14 ms steps")
    parser.add_argument("--tau-adapt", type=float, default=20.0, help="adaptation time constant, in steps")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--clip", type=float, default=1.0, help="global gradient norm limit")
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--surrogate", choices=["atan", "superspike"], default="atan")
    parser.add_argument("--run", default="runs/shd")
    args = parser.parse_args()

    train, test = shd("train"), shd("test")
    data = Dataset.from_records(train, batch=args.batch, validation=test)
    surrogate = sparx.surrogate.ATan() if args.surrogate == "atan" else sparx.surrogate.FastSigmoid(100.0)
    net = Net(args.hidden, args.tau, args.tau_adapt, args.dropout, args.recurrent, surrogate)
    objective = SpikingClassifier(net, Field("spikes", train["spikes"].shape[1:]), Events(), readout="max",
                                  rates=RateBand(lower=0.01, upper=0.3, weight=1.0))
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.steps)
    optimizer = optax.chain(optax.clip_by_global_norm(args.clip), optax.adamw(schedule, weight_decay=1e-4))
    trainer = Trainer(objective, optimizer, key=jax.random.key(0), checkpoints=Checkpoints(args.run))
    start = time.perf_counter()
    state = trainer.fit(data, steps=args.steps, log_every=100, eval_every=500, metrics=[accuracy])
    trained = time.perf_counter() - start

    @jax.jit
    def predict(variables, spikes):
        outputs = net.apply(variables, jnp.moveaxis(spikes, 1, 0).astype(jnp.float32), train=False)
        return jnp.argmax(jnp.max(outputs, axis=0), -1)

    predictions = np.concatenate([predict(state.variables, test["spikes"][i:i + 256])
                                  for i in range(0, len(test["label"]), 256)])
    print(f"test accuracy {np.mean(predictions == test['label']):.4f} over {len(predictions)} recordings "
          f"after {args.steps} steps ({trained:.0f} s on {jax.devices()[0].device_kind})")


if __name__ == "__main__":
    main()
