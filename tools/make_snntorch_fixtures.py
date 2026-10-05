"""Write snnTorch's spike encodings for sparx's encoder parity tests.

Saves inputs and snnTorch's `spikegen.latency` and `spikegen.delta` outputs
to `tests/fixtures/snntorch.npz`, which `tests/test_encode.py` compares
`sparx.encode` against. The committed fixture came from snnTorch 1.0.0 with
torch 2.14.1+cpu:

    pip install snntorch==1.0.0
    python tools/make_snntorch_fixtures.py
"""

from pathlib import Path

import numpy as np
import snntorch
import snntorch.functional as SF
import snntorch.spikegen as spikegen
import torch

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "snntorch.npz"


def _overshoots(x: np.ndarray, beta: float, alpha: float | None) -> bool:
    """Whether any membrane is still at or above threshold 1 after its own soft reset, in float64."""
    i = np.zeros(x.shape[1:])
    v = np.zeros(x.shape[1:])
    for xt in x.astype(np.float64):
        i = (alpha * i if alpha is not None else 0) + xt
        v = beta * v + (i if alpha is not None else xt)
        v = v - (v >= 1)
        if (v >= 1).any():
            return True
    return False


def _run(layer, x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xt = torch.tensor(x, requires_grad=True)
    spikes, membranes = [], []
    for step in xt:
        out = layer(step)
        spikes.append(out[0])
        membranes.append(out[-1].detach())
    spike = torch.stack(spikes)
    (spike * torch.tensor(weights)).sum().backward()
    return spike.detach().numpy(), xt.grad.numpy(), float((torch.stack(membranes) - 1.0).abs().min())


def neuron_cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Leaky and Synaptic with both reset rules and both reset timings.

    `neuron_x` keeps every soft-reset membrane below threshold after its
    reset, where snnTorch's immediate reset is sparx's model; `overshoot_x`
    drives some above it, where snnTorch's double-reset guard makes such a
    neuron need twice the threshold to fire again (docs/fidelity.md). Every
    membrane stays at least 1e-4 from threshold, so the strict and inclusive
    thresholds (snnTorch fires on `> 0`, sparx on `>= 0`) agree.
    """
    for _ in range(100):
        try:
            return _neuron_cases(rng)
        except _NearThreshold:
            continue
    raise RuntimeError("no draw in 100 kept every membrane away from threshold")


class _NearThreshold(Exception):
    pass


def _neuron_cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    cases: dict[str, np.ndarray] = {}
    inputs = {"neuron_x": rng.normal(0.16, 0.3, (40, 3, 6)).astype(np.float32),
              "overshoot_x": rng.normal(0.35, 0.7, (40, 3, 6)).astype(np.float32)}
    if _overshoots(inputs["neuron_x"], 0.8, None) or _overshoots(inputs["neuron_x"], 0.8, 0.6):
        raise _NearThreshold
    assert _overshoots(inputs["overshoot_x"], 0.8, None)
    weights = rng.normal(size=(40, 3, 6)).astype(np.float32)
    cases |= inputs | {"neuron_weights": weights}
    for input_name, x in inputs.items():
        for name, make in (
                ("leaky", lambda reset, delay: snntorch.Leaky(beta=0.8, threshold=1.0, reset_mechanism=reset,
                                                              reset_delay=delay)),
                ("synaptic", lambda reset, delay: snntorch.Synaptic(
                    alpha=0.6, beta=0.8, threshold=1.0, reset_mechanism=reset, reset_delay=delay))):
            for reset in ("subtract", "zero"):
                for delay in (False, True):
                    spikes, grad_x, margin = _run(make(reset, delay), x, weights)
                    if margin <= 1e-4:
                        raise _NearThreshold
                    key = f"{input_name}/{name}_{reset}_{'delayed' if delay else 'immediate'}"
                    cases[f"{key}/spikes"] = spikes
                    cases[f"{key}/grad_x"] = grad_x
    return cases


def main():
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1, (6, 7)).astype(np.float32)
    # The edges: zero and a value under the threshold never fire, one fires first.
    data[0, :3] = [0.0, 0.005, 1.0]
    latency = spikegen.latency(torch.tensor(data), num_steps=9, threshold=0.01, linear=True,
                               normalize=True, clip=True).numpy()
    signal = np.cumsum(rng.normal(0, 1, (12, 5)), 0).astype(np.float32)
    delta = spikegen.delta(torch.tensor(signal), threshold=0.5, off_spike=True).numpy()
    delta_on = spikegen.delta(torch.tensor(signal), threshold=0.5).numpy()
    # Loss inputs: T = 10 steps, so the target counts 8 and 2 are whole.
    outputs = rng.normal(0, 2, (10, 4, 3)).astype(np.float32)
    spikes = (rng.uniform(size=(10, 4, 3)) < 0.4).astype(np.float32)
    labels = rng.integers(0, 3, 4)
    ce_rate = SF.ce_rate_loss(reduction="none")(torch.tensor(outputs), torch.tensor(labels)).numpy()
    mse_count = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2, reduction="none")(
        torch.tensor(spikes), torch.tensor(labels)).numpy()
    neurons = neuron_cases(rng)
    np.savez_compressed(OUT, **neurons, latency_input=data, latency=latency, delta_input=signal, delta=delta,
                        delta_on=delta_on, loss_outputs=outputs, loss_spikes=spikes, loss_labels=labels,
                        ce_rate=ce_rate, mse_count=mse_count, snntorch=np.array(snntorch.__version__),
                        torch=np.array(torch.__version__))
    print(f"wrote {OUT} (snntorch {snntorch.__version__}, torch {torch.__version__})")


if __name__ == "__main__":
    main()
