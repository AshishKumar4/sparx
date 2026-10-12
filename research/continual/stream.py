"""Forgetting under local learning rules: a recurrent spiking network learns a stream of SHD tasks one after
another, with no replay, and is tested on every task after each.

Two streams. `classes`: SHD's 20 classes in 5 tasks of 4, drawn by the seed, all sharing one 20-way
readout; a test counts the argmax over all 20 (class-incremental) and over the task's own 4 (task-
incremental). `permuted`: the same 20 classes in every task, each task's 700 channels permuted (the first
task's not), one readout.

Learners, on one network, 700 inputs into 256 recurrent LIF neurons into 20 leaky readouts
(`sparx.learn.eprop_forward`):

- `bptt`: backpropagation through time (`sparx.learn.bptt_loss`) with Adam;
- `ewc`: the same with elastic weight consolidation (Kirkpatrick et al. 2017), online: after each task the
  diagonal empirical Fisher of its training data is added to a running sum, and the penalty
  `lambda / 2 * sum F (theta - theta*)^2` pulls toward the parameters at that task's end;
- `eprop`: e-prop (`sparx.learn.eprop`, Bellec et al. 2020), each synapse's eligibility trace times a
  learning signal fed back through the readout's weights; `eprop-random`, through fixed random ones;
- `joint`: BPTT on every task's data at once, for as many epochs, the reference a stream is measured
  against (its accuracy matrix has the one row).

Each writes `<out>/<stream>-<learner>-<seed>.json`, the learner `ewc<lambda>` for EWC: the accuracy matrix
(row i, after task i; column j, task j's test set), the forgetting of each task (its best accuracy before
the last task minus its last), the mean final accuracy, the loss after each epoch, the hidden neurons'
mean rate, and the time taken.

With `--select`, a learner trains on the first task alone, less a tenth of its training recordings held
out, and writes its accuracy on those to `<out>/<stream>-<learner>-<seed>-lr<lr>.json`: the score a
learning rate is chosen by, with no test recording seen.

    python research/continual/stream.py --stream classes --learner eprop --seed 0 --out runs/stream
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sparx.datasets import shd
from sparx.dynamics import LIFCell, decay
from sparx.learn import EPropParams, bptt_loss, eprop, eprop_forward

CLASSES = 20
TASKS = 5
CHANNELS = 700


def tasks(stream: str, seed: int) -> list[dict]:
    """Each task's classes, and its channel permutation (None for the identity)."""
    rng = np.random.default_rng(seed)
    if stream == "classes":
        order = rng.permutation(CLASSES)
        return [{"classes": sorted(order[4 * k:4 * k + 4].tolist()), "permutation": None}
                for k in range(TASKS)]
    return [{"classes": list(range(CLASSES)), "permutation": None if k == 0 else rng.permutation(CHANNELS)}
            for k in range(TASKS)]


def subset(data: dict, task: dict) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(data["label"], task["classes"])
    spikes = data["spikes"][keep]
    if task["permutation"] is not None:
        spikes = spikes[:, :, task["permutation"]]
    return spikes, data["label"][keep]


def time_major(spikes: np.ndarray) -> jax.Array:
    return jnp.asarray(spikes.swapaxes(0, 1), jnp.float32)


class Learner:
    """The network, its rule and its optimizer: `update` takes a batch, `consolidate` ends a task, `predict`
    gives the time-averaged readout and the hidden neurons' mean rate."""

    def __init__(self, args: argparse.Namespace, steps: int):
        self.args = args
        self.cell = LIFCell(decay(args.tau), reset="zero")
        self.mask = 1 - jnp.eye(args.hidden)
        self.steps = steps
        # Clipping is off by default: at norm 1 it did not steady BPTT and stalled EWC (README, results/).
        clip = [optax.clip_by_global_norm(args.clip)] if args.clip else []
        self.optimizer = optax.chain(*clip, optax.adam(args.lr))
        k_in, k_rec, k_out, k_feedback = jax.random.split(jax.random.key(args.seed), 4)
        hidden = args.hidden
        self.params = EPropParams(
            w_in=jax.random.normal(k_in, (CHANNELS, hidden)) * 2.0 / np.sqrt(CHANNELS),
            w_rec=jax.random.normal(k_rec, (hidden, hidden)) / np.sqrt(hidden) * self.mask,
            w_out=jax.random.normal(k_out, (hidden, CLASSES)) / np.sqrt(hidden),
            b_out=jnp.zeros(CLASSES),
        )
        self.feedback = (jax.random.normal(k_feedback, (hidden, CLASSES)) / np.sqrt(hidden)
                         if args.learner == "eprop-random" else None)
        self.state = self.optimizer.init(self.params)
        self.fisher = jax.tree.map(jnp.zeros_like, self.params)
        self.anchor = self.params
        self._update = jax.jit(self.update_fn)
        self._predict = jax.jit(self.predict_fn)
        self._squares = jax.jit(self.squares_fn)

    def step_loss(self, y: jax.Array, label: jax.Array) -> jax.Array:
        """A step's cross entropy, averaged over the batch; summed over time, the per-step mean."""
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y, label)) / self.steps

    def loss(self, params: EPropParams, x: jax.Array, labels: jax.Array) -> jax.Array:
        targets = jnp.broadcast_to(labels, (x.shape[0], *labels.shape))
        return bptt_loss(self.cell, params, x, targets, self.step_loss, tau=self.args.readout_tau)

    def update_fn(self, params, state, x, labels, feedback, fisher, anchor):
        if self.args.learner in ("bptt", "ewc", "joint"):
            loss, grads = jax.value_and_grad(self.loss)(params, x, labels)
        else:
            targets = jnp.broadcast_to(labels, (x.shape[0], *labels.shape))
            loss, grads = eprop(self.cell, params, x, targets, self.step_loss, tau=self.args.readout_tau,
                                feedback=feedback)
        if self.args.learner == "ewc":
            grads = jax.tree.map(lambda g, f, p, a: g + self.args.ewc * f * (p - a), grads, fisher, params,
                                 anchor)
        grads = grads._replace(w_rec=grads.w_rec * self.mask)
        changes, state = self.optimizer.update(grads, state, params)
        return optax.apply_updates(params, changes), state, loss, optax.global_norm(grads)

    def predict_fn(self, params, x):
        y, z = eprop_forward(self.cell, params, x, tau=self.args.readout_tau)
        return jnp.mean(y, 0), jnp.mean(z)

    def squares_fn(self, params, x, labels):
        """The sum over the batch of each recording's squared gradient: the diagonal empirical Fisher's
        terms."""
        def one(xi, label):
            return jax.grad(self.loss)(params, xi[:, None], label[None])
        grads = jax.vmap(one, in_axes=(1, 0))(x, labels)
        return jax.tree.map(lambda g: jnp.sum(g**2, 0), grads)

    def update(self, x: jax.Array, labels: np.ndarray) -> tuple[float, float]:
        """The batch's loss and its gradient's global norm before clipping."""
        self.params, self.state, loss, norm = self._update(self.params, self.state, x, jnp.asarray(labels),
                                                           self.feedback, self.fisher, self.anchor)
        return float(loss), float(norm)

    def consolidate(self, spikes: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> None:
        """For EWC, the task's Fisher into the running sum, and the parameters as the new anchor."""
        if self.args.learner != "ewc":
            return
        rows = rng.permutation(len(labels))[:self.args.fisher_samples]
        squares = jax.tree.map(jnp.zeros_like, self.params)
        for start in range(0, len(rows), 32):
            chunk = rows[start:start + 32]
            more = self._squares(self.params, time_major(spikes[chunk]), jnp.asarray(labels[chunk]))
            squares = jax.tree.map(jnp.add, squares, more)
        self.fisher = jax.tree.map(lambda f, s: f + s / len(rows), self.fisher, squares)
        self.anchor = self.params

    def predict(self, spikes: np.ndarray) -> tuple[np.ndarray, float]:
        scores, rate = [], 0.0
        for start in range(0, len(spikes), 256):
            y, r = self._predict(self.params, time_major(spikes[start:start + 256]))
            scores.append(np.asarray(y))
            rate += float(r) * len(y)
        return np.concatenate(scores), rate / len(spikes)


def fit(learner: Learner, spikes: np.ndarray, labels: np.ndarray, epochs: int, rng: np.random.Generator,
        task: int) -> list[dict]:
    """`epochs` passes over a task's recordings in shuffled batches: each epoch's mean loss and its batches'
    median and largest gradient norm."""
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        batches = [order[s:s + learner.args.batch] for s in range(0, len(order) - learner.args.batch + 1,
                                                                  learner.args.batch)]
        results = np.array([learner.update(time_major(spikes[b]), labels[b]) for b in batches])
        loss = float(np.mean(results[:, 0]))
        losses.append({"task": task, "loss": loss, "grad_norm_median": float(np.median(results[:, 1])),
                       "grad_norm_max": float(np.max(results[:, 1]))})
        if not np.isfinite(loss):
            raise SystemExit(f"task {task}: the loss is not finite")
    return losses


def evaluate(learner: Learner, stream: list[dict], tests: list) -> tuple[list[float], list[float], float]:
    """Each task's test accuracy over all 20 classes and over its own, and the hidden neurons' mean rate."""
    whole, own, rates = [], [], []
    for task, (spikes, labels) in zip(stream, tests, strict=True):
        y, rate = learner.predict(spikes)
        whole.append(float(np.mean(np.argmax(y, -1) == labels)))
        classes = np.asarray(task["classes"])
        own.append(float(np.mean(classes[np.argmax(y[:, classes], -1)] == labels)))
        rates.append(rate)
    return whole, own, float(np.mean(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", choices=("classes", "permuted"), required=True)
    parser.add_argument("--learner", choices=("bptt", "ewc", "eprop", "eprop-random", "joint"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="per task; 20 for classes, 5 for permuted")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=0.0, help="the gradient's largest global norm; 0, none")
    parser.add_argument("--ewc", type=float, default=1e4, help="the EWC penalty's lambda")
    parser.add_argument("--fisher-samples", type=int, default=256)
    parser.add_argument("--tau", type=float, default=10.0, help="the neurons' time constant, in steps")
    parser.add_argument("--readout-tau", type=float, default=20.0)
    parser.add_argument("--limit", type=int, help="only this many recordings of each split, for a smoke run")
    parser.add_argument("--select", action="store_true",
                        help="score the learning rate on held-out recordings of the first task alone")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    epochs = args.epochs or (20 if args.stream == "classes" else 5)
    started = time.time()

    pick = np.random.default_rng(0)

    def load(split: str) -> dict:
        data = shd(split)
        if not args.limit:
            return data
        # A random subset: the files are ordered, so their first recordings hold only a few classes.
        rows = np.sort(pick.permutation(len(data["label"]))[:args.limit])
        return {k: v[rows] for k, v in data.items()}

    train = load("train")
    stream = tasks(args.stream, args.seed)
    learner = Learner(args, train["spikes"].shape[1])
    rng = np.random.default_rng(args.seed)
    matrix, own_matrix, losses, rate = [], [], [], 0.0
    phases = [subset(train, task) for task in stream]
    rule = f"ewc{args.ewc:g}" if args.learner == "ewc" else args.learner
    args.out.mkdir(parents=True, exist_ok=True)
    if args.select:
        spikes, labels = phases[0]
        held = np.isin(np.arange(len(labels)), rng.permutation(len(labels))[:len(labels) // 10])
        losses = fit(learner, spikes[~held], labels[~held], epochs, rng, 0)
        y, _ = learner.predict(spikes[held])
        report = {"stream": args.stream, "learner": rule, "seed": args.seed, "lr": args.lr,
                  "epochs": epochs, "held_out": float(np.mean(np.argmax(y, -1) == labels[held])),
                  "recordings": int(held.sum()), "losses": losses, "seconds": round(time.time() - started, 1)}
        print(json.dumps({k: report[k] for k in ("lr", "held_out", "seconds")}), flush=True)
        (args.out / f"{args.stream}-{rule}-{args.seed}-lr{args.lr:g}.json").write_text(
            json.dumps(report, indent=1) + "\n")
        return
    test = load("test")
    tests = [subset(test, task) for task in stream]
    if args.learner == "joint":
        phases = [(np.concatenate([x for x, _ in phases]), np.concatenate([y for _, y in phases]))]
    for k, (spikes, labels) in enumerate(phases):
        losses += fit(learner, spikes, labels, epochs, rng, k)
        learner.consolidate(spikes, labels, rng)
        whole, own, rate = evaluate(learner, stream, tests)
        matrix.append(whole)
        own_matrix.append(own)
        print(json.dumps({"after_task": k, "accuracy": [round(a, 3) for a in whole],
                          "own_classes": [round(a, 3) for a in own], "rate": round(rate, 4),
                          "seconds": round(time.time() - started)}), flush=True)

    final = matrix[-1]
    forgetting = [max(matrix[i][j] for i in range(j, len(matrix) - 1)) - final[j]
                  for j in range(len(matrix) - 1)]
    report = {
        "stream": args.stream, "learner": args.learner, "seed": args.seed, "epochs": epochs,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "tasks": [task["classes"] for task in stream],
        "accuracy": matrix, "own_classes": own_matrix,
        "final_mean": float(np.mean(final)), "own_classes_final_mean": float(np.mean(own_matrix[-1])),
        "forgetting": forgetting, "mean_forgetting": float(np.mean(forgetting)) if forgetting else None,
        "losses": losses, "rate": rate, "seconds": round(time.time() - started, 1),
        "device": str(jax.devices()[0]), "jax": jax.__version__,
    }
    (args.out / f"{args.stream}-{rule}-{args.seed}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
