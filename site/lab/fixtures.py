"""What sparx computes for the Learn pages' live figures, in float64, for the browser's engines to match
(site/test/*.test.ts):

    python site/lab/fixtures.py --out site/test/fixtures/sparx.json

- neurons: `LIFCell` (each reset), `LeakyIntegrateAndFire` on a current and `Izhikevich` in each of
  Izhikevich's 2003 classes, on recorded inputs;
- teach: the loss and gradient of the surrogate-gradients page's neuron, by `jax.grad` through
  `sparx.spike`, for each surrogate;
- delays: `delay_kernel` and the gradient of the delays page's readout through `DelayedDense`;
- stdp: `PairSTDP.step` on recorded pre- and postsynaptic spikes, additive and multiplicative;
- brunel: a Brunel network `sparx.graph.Network` builds and steps, its edges read back, with its
  external input recorded as `ArrivalInput`s.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sparx
from sparx import surrogate as surrogates
from sparx.dynamics import (
    IZHIKEVICH_2003,
    ALIFCell,
    Izhikevich,
    LeakyIntegrateAndFire,
    LIFCell,
    PairSTDP,
    SynapticInput,
    decay,
    run,
)
from sparx.nn import LI, DelayedDense, delay_kernel


def neurons(rng: np.random.Generator) -> dict:
    steps = 400
    drive = 0.08 + 0.25 * rng.random(steps)
    lif = {}
    for reset in ("subtract", "zero", "none"):
        cell = LIFCell(decay=decay(tau=12.0), threshold=1.0, reset=reset)
        (out, v), _ = run(cell, jnp.asarray(drive), record=lambda s: s.v)
        lif[reset] = {"spikes": np.flatnonzero(np.asarray(out.value)).tolist(), "v": np.asarray(v).tolist()}
    cell = ALIFCell(decay=decay(tau=12.0), adapt_decay=decay(tau=200.0), beta=0.3)
    (out, v), _ = run(cell, jnp.asarray(drive), record=lambda s: s.v)
    lif["alif"] = {"spikes": np.flatnonzero(np.asarray(out.value)).tolist(), "v": np.asarray(v).tolist()}
    current = 300.0 + 250.0 * rng.random(3000)
    cell = LeakyIntegrateAndFire()
    (out, v), _ = run(cell, SynapticInput(current=jnp.asarray(current)), dt=0.1, record=lambda s: s.v)
    physical = {
        "current": current.tolist(),
        "dt": 0.1,
        "v": np.asarray(v).tolist(),
        "spikes": np.flatnonzero(np.asarray(out.value)).tolist(),
    }
    izhikevich = {}
    # Compiled, XLA fuses the quadratic membrane's arithmetic and rounds its last bit differently, which
    # the membrane amplifies into a step's difference in a spike (docs/fidelity.md); run op by op.
    for name, (a, b, c, d) in IZHIKEVICH_2003.items():
        steps_i = 2000
        current_i = np.where(np.arange(steps_i) > 200, 10.0, 0.0)
        cell = Izhikevich(a=a, b=b, c=c, d=d)
        with jax.disable_jit():
            (out, v), _ = run(
                cell, SynapticInput(current=jnp.asarray(current_i)), dt=0.1, record=lambda s: s.v
            )
        izhikevich[name] = {
            "spikes": np.flatnonzero(np.asarray(out.value)).tolist(),
            "v": np.asarray(v).tolist(),
        }
    return {
        "drive": drive.tolist(),
        "lif": lif,
        "physical": physical,
        "izhikevich": izhikevich,
        "izhikevich_current": {"after": 200, "amplitude": 10.0, "dt": 0.1, "steps": 2000},
    }


def filtered(x: jax.Array, keep: float) -> jax.Array:
    return jax.lax.scan(lambda f, xt: (keep * f + xt, keep * f + xt), jnp.zeros(()), x)[1]


def teach(rng: np.random.Generator) -> dict:
    """One LIF neuron on 40 Poisson inputs through weights `w`; the loss is the mean squared difference
    of its spikes and the target's, each filtered by an exponential of 10 steps."""
    steps, inputs = 200, 40
    trains = (rng.random((steps, inputs)) < 0.04).astype(np.float64)
    w = rng.normal(0.0, 0.35, inputs)
    target = np.zeros(steps)
    target[[40, 90, 150]] = 1.0
    keep = math.exp(-1 / 10)
    cases = {}
    for name, made in (
        ("ATan", surrogates.ATan()),
        ("FastSigmoid", surrogates.FastSigmoid()),
        ("Triangle", surrogates.Triangle()),
    ):
        cell = LIFCell(decay=decay(tau=10.0), threshold=1.0, reset="subtract", surrogate=made)

        def loss(w, cell=cell):
            out, _ = run(cell, jnp.asarray(trains) @ w)
            return jnp.mean((filtered(out.value, keep) - filtered(jnp.asarray(target), keep)) ** 2)

        value, grad = jax.value_and_grad(loss)(jnp.asarray(w))
        cases[name] = {"loss": float(value), "grad": np.asarray(grad).tolist()}
    return {
        "trains": trains.astype(int).tolist(),
        "w": w.tolist(),
        "target": [40, 90, 150],
        "tau": 10.0,
        "filter_tau": 10.0,
        "cases": cases,
    }


def encoders(rng: np.random.Generator) -> dict:
    """`LatencyEncoder` on values and `DeltaEncoder` on a signal, both deterministic."""
    from sparx.encode import DeltaEncoder, LatencyEncoder

    values = np.concatenate([rng.random(30), [0.0, 0.005, 0.01, 0.3, 0.5, 1.0]])
    signal = np.cumsum(rng.normal(0.0, 0.08, 120))
    key = jax.random.key(0)
    out = {"values": values.tolist(), "signal": signal.tolist(), "latency": {}, "delta": {}}
    for steps in (8, 16, 33):
        spikes = np.asarray(LatencyEncoder(steps=steps)(key, jnp.asarray(values)[None]))[:, 0]
        out["latency"][str(steps)] = [np.flatnonzero(spikes[:, i]).tolist() for i in range(len(values))]
    for threshold in (0.05, 0.1):
        events = DeltaEncoder(threshold=threshold, off_spikes=True)(key, jnp.asarray(signal)[None, :, None])
        out["delta"][str(threshold)] = np.asarray(events[:, 0, 0]).astype(int).tolist()
    return out


def delays(rng: np.random.Generator) -> dict:
    """Three inputs that each fire once, delayed onto one leaky integrator; the loss is minus its peak, or
    minus its value at step 50, the delays page's (`read_at`)."""
    steps = 80
    x = np.zeros((steps, 1, 3))
    for i, t in enumerate((5, 18, 30)):
        x[t, 0, i] = 1.0
    weight = np.full((3, 1), 0.6)
    readout = LI(tau=4.0)
    out = {"x_times": [5, 18, 30], "steps": steps, "weight": 0.6, "tau": 4.0, "cases": []}
    for max_delay, delay, read_at in ((30, [24.3, 12.7, 3.1], None), (45, [4.0, 16.0, 22.0], 50)):
        layer = DelayedDense(1, max_delay, use_bias=False)
        for sigma in (6.0, 1.5, 0.4):

            def loss(delay, sigma=sigma, layer=layer, read_at=read_at):
                y = layer.apply(
                    {"params": {"kernel": jnp.asarray(weight), "delay": delay}}, jnp.asarray(x), sigma
                )
                v = readout.apply({}, y)
                return -(jnp.max(v) if read_at is None else v[read_at, 0, 0])

            value, grad = jax.value_and_grad(loss)(jnp.asarray(delay)[:, None])
            kernel = delay_kernel(jnp.asarray(delay), max_delay, sigma)
            out["cases"].append(
                {
                    "max_delay": max_delay,
                    "delay": delay,
                    "read_at": read_at,
                    "sigma": sigma,
                    "loss": float(value),
                    "grad": np.asarray(grad)[:, 0].tolist(),
                    "kernel": np.asarray(kernel).T.tolist(),
                }
            )
    return out


def stdp(rng: np.random.Generator) -> dict:
    steps, pre_n, post_n = 3000, 6, 2
    pre_spikes = (rng.random((steps, pre_n)) < 0.02).astype(np.float64)
    post_spikes = (rng.random((steps, post_n)) < 0.02).astype(np.float64)
    pre = np.repeat(np.arange(pre_n), post_n)
    post = np.tile(np.arange(post_n), pre_n)
    cases = {}
    for name, mu in (("additive", 0.0), ("multiplicative", 1.0)):
        rule = PairSTDP(
            tau_plus=16.8, tau_minus=33.7, lambda_=0.03, alpha=0.6, mu_plus=mu, mu_minus=mu, w_max=1.0
        )
        traces = rule.init_state(pre_n, post_n, len(pre), jnp.float64)
        w = jnp.full(len(pre), 0.5)

        def step(carry, spikes, rule=rule):
            traces, w = carry
            traces, w = rule.step(
                traces, w, spikes[0], spikes[1], jnp.asarray(pre), jnp.asarray(post), 0.1, modulators={}
            )
            return (traces, w), w

        _, ws = jax.lax.scan(step, (traces, w), (jnp.asarray(pre_spikes), jnp.asarray(post_spikes)))
        cases[name] = {
            "mu": mu,
            "weights": np.asarray(ws)[::50].tolist(),
            "final": np.asarray(ws)[-1].tolist(),
        }
    return {
        "pre_spikes": [np.flatnonzero(r).tolist() for r in pre_spikes],
        "post_spikes": [np.flatnonzero(r).tolist() for r in post_spikes],
        "pre": pre.tolist(),
        "post": post.tolist(),
        "dt": 0.1,
        "rule": {"tau_plus": 16.8, "tau_minus": 33.7, "lambda": 0.03, "alpha": 0.6, "w_max": 1.0},
        "cases": cases,
    }


def brunel(rng: np.random.Generator) -> dict:
    """`brunel`'s network at order 50 with its Poisson input drawn here and replayed as `ArrivalInput`s."""
    from sparx.graph import ArrivalInput, Network, SpikeRaster, simulate
    from sparx.graph.models import brunel as build

    order, g, eta = 50, 5.0, 2.0
    made = build(order, g=g, eta=eta)
    network = Network(
        made.populations,
        made.projections,
        tuple(ArrivalInput(p.name, f"external_{p.name}", "ampa") for p in made.populations),
        dt=made.dt,
        dtype=jnp.float64,
    )
    duration, dt = 200.0, made.dt
    steps = round(duration / dt)
    ce = round(0.1 * 4 * order)
    mean = eta * (20.0 / (0.1 * ce * 20.0)) * 1000.0 * ce * dt / 1000.0
    external = {p.name: 0.1 * rng.poisson(mean, (steps, p.size)) for p in made.populations}
    variables = network.init(jax.random.key(3))
    result = simulate(
        network,
        variables,
        duration=duration,
        drive={f"external_{k}": jnp.asarray(v) for k, v in external.items()},
        monitors={"e": SpikeRaster("e"), "i": SpikeRaster("i")},
    )
    connections = network.connections(variables)
    sizes = {p.name: p.size for p in made.populations}
    offset = {"e": 0, "i": sizes["e"]}
    edges = {"pre": [], "post": [], "weight": [], "delay": []}
    for p in made.projections:
        c = connections[p.key]
        edges["pre"] += (np.asarray(c.pre) + offset[p.pre]).tolist()
        edges["post"] += (np.asarray(c.post) + offset[p.post]).tolist()
        edges["weight"] += np.broadcast_to(np.asarray(c.weight), np.shape(c.pre)).tolist()
        edges["delay"] += np.broadcast_to(np.asarray(c.delay), np.shape(c.pre)).tolist()
    spikes = np.concatenate([np.asarray(result.records["e"]), np.asarray(result.records["i"])], axis=1)
    return {
        "order": order,
        "g": g,
        "eta": eta,
        "dt": dt,
        "steps": steps,
        "sizes": sizes,
        "edges": edges,
        "external": np.concatenate([external["e"], external["i"]], axis=1).round(12).tolist(),
        "spikes": [np.flatnonzero(row).tolist() for row in spikes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="site/test/fixtures/sparx.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    makers = {
        "neurons": neurons,
        "teach": teach,
        "delays": delays,
        "stdp": stdp,
        "brunel": brunel,
        "encoders": encoders,
    }
    with jax.enable_x64(new_val=True):
        data = {
            name: make(np.random.default_rng(k))
            for k, (name, make) in enumerate(makers.items())
            if args.only is None or name in args.only
        }
    data["versions"] = {"jax": jax.__version__, "sparx": sparx.__version__}
    Path(args.out).write_text(json.dumps(data))


if __name__ == "__main__":
    main()
