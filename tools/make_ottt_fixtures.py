"""Write OTTT gradients from Xiao et al.'s own modules for sparx's parity test.

Builds a small spiking MLP from their repository's `OnlineLIFNode` and
`WrapedSNNOp` (github.com/pkuxmq/OTTT-SNN, cloned next to sparx as
`ref-ottt`): a linear layer on the input, two online LIF layers, a wrapped
hidden layer and a wrapped readout. It trains one batch as their
`train_cifar.py` does: a cross-entropy loss each step, divided by the number
of steps and backpropagated at once, gradients accumulating over steps.
Saves inputs, labels, weights and gradients to `tests/fixtures/ottt.npz`.

    pip install torch
    python tools/make_ottt_fixtures.py [path to the cloned repository]
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "ottt.npz"
STEPS, BATCH, SIZES, TAU = 12, 4, (6, 10, 8, 3), 2.0


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ref-ottt"
    sys.path.insert(0, str(repo))
    from models.spiking_vgg import WrapedSNNOp
    from modules.neuron import OnlineLIFNode

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    first = torch.nn.Linear(SIZES[0], SIZES[1])
    hidden = torch.nn.Linear(SIZES[1], SIZES[2])
    readout = torch.nn.Linear(SIZES[2], SIZES[3])
    with torch.no_grad():  # weights large enough that every layer spikes
        for layer in (first, hidden, readout):
            layer.weight.mul_(3.0)
    sn1, sn2 = OnlineLIFNode(tau=TAU), OnlineLIFNode(tau=TAU)
    wrapped_hidden, wrapped_readout = WrapedSNNOp(hidden), WrapedSNNOp(readout)
    inputs = torch.rand(STEPS, BATCH, SIZES[0])
    labels = torch.randint(0, SIZES[3], (BATCH,))
    spikes = []
    for t in range(STEPS):
        h = sn1(first(inputs[t]), init=t == 0, output_type="spike_rate")
        h = sn2(wrapped_hidden(h, require_wrap=True), init=t == 0, output_type="spike_rate")
        spikes.append(float(h[:BATCH].sum()))
        out = wrapped_readout(h, require_wrap=True)
        loss = F.cross_entropy(out, labels) / STEPS
        loss.backward()
    cases = {"inputs": inputs.numpy(), "labels": labels.numpy(), "tau": np.array(TAU),
             "meta/torch": np.array(torch.__version__)}
    for name, layer in (("first", first), ("hidden", hidden), ("readout", readout)):
        cases[f"{name}/weight"] = layer.weight.detach().numpy().T.copy()
        cases[f"{name}/bias"] = layer.bias.detach().numpy().copy()
        cases[f"{name}/grad_weight"] = layer.weight.grad.numpy().T.copy()
        cases[f"{name}/grad_bias"] = layer.bias.grad.numpy().copy()
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT}; last-layer spikes per step {spikes}")


if __name__ == "__main__":
    main()
