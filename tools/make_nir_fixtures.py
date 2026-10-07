"""Write NIR graphs exported by snnTorch, with snnTorch's outputs, for sparx's NIR round trips.

Three snnTorch networks, all with hard-reset neurons, are exported with
`snntorch.export_nir` and run on random input:

- `snntorch.nir`, `nir.npz`: `Linear`, `Leaky`, `Linear`, `Leaky`.
- `snntorch_conv.nir`, `nir_conv.npz`: two `Conv2d` layers (a non-square
  input, and a stride, padding and dilation that differ between height and
  width) with `Leaky` neurons, `Flatten`, `Linear`, `Leaky`. The kernels
  are square and ungrouped because NIR 1.0.8's type inference reads a 2-d
  kernel's size from its height alone and its input channels from the
  weight as if ungrouped, so it rejects other graphs on reading. Inputs and hidden spikes are stored NCHW,
  as snnTorch ran them.
- `snntorch_rleaky.nir`, `nir_rleaky.npz`: `Linear`, `RLeaky` (all-to-all
  feedback with a bias), `Linear`, `Leaky`.

Each `.npz` holds the input, the output spikes and the first spiking
layer's spikes (`hidden`). `tests/test_nir.py` imports each graph into
sparx, runs it, and exports it back.

    pip install snntorch==1.0.0 nir==1.0.8 nirtorch==2.6
    python tools/make_nir_fixtures.py

`snntorch.export_nir` imports nirtorch, which builds a graph's edges from a
set, so the order of the edges in a `.nir` file follows `PYTHONHASHSEED`;
the graph and every array are the same whatever the order.
"""

from pathlib import Path

import nir
import numpy as np
import snntorch as snn
import torch
from snntorch import utils as snn_utils
from snntorch.export_nir import export_to_nir

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
STEPS, BATCH = 30, 4


def leaky(beta: float, shape: tuple[int, ...], output: bool = False) -> snn.Leaky:
    """A hard-reset `Leaky` with per-neuron parameters, which snnTorch's exporter needs for NIR's types."""
    return snn.Leaky(beta=torch.full(shape, beta), threshold=torch.ones(shape), reset_mechanism="zero",
                     init_hidden=True, reset_delay=False, output=output)


def run(net: torch.nn.Sequential, inputs: torch.Tensor, hidden: int) -> tuple[np.ndarray, np.ndarray]:
    """The output spikes of `net` over `inputs` `[T, B, ...]`, and the spikes out of layer `hidden`."""
    snn_utils.reset(net)
    outputs, taps = [], []
    with torch.no_grad():
        for t in range(STEPS):
            x = inputs[t]
            for k, module in enumerate(net):
                x = module(x)
                if isinstance(x, tuple):
                    x = x[0]
                if k == hidden:
                    taps.append(x.numpy())
            outputs.append(x.numpy())
    return np.stack(outputs), np.stack(taps)


def write(name: str, net: torch.nn.Sequential, inputs: torch.Tensor, sample: torch.Tensor,
          hidden: int, **export: list[int]) -> None:
    graph = export_to_nir(net, sample, **export)
    nir.write(FIXTURES / f"snntorch{name}.nir", graph)
    outputs, taps = run(net, inputs, hidden)
    np.savez_compressed(FIXTURES / f"nir{name}.npz", inputs=inputs.numpy(), spikes=outputs, hidden=taps,
                        meta_snntorch=np.array(snn.__version__))
    print("wrote", FIXTURES / f"snntorch{name}.nir", "output spikes", outputs.sum(),
          "hidden spikes", taps.sum(), "nodes", list(graph.nodes))


def dense() -> None:
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        leaky(0.85, (8,)),
        torch.nn.Linear(8, 3),
        leaky(0.9, (3,), output=True),
    )
    with torch.no_grad():
        net[0].weight.mul_(2.5)
        net[2].weight.mul_(2.5)
    inputs = torch.rand(STEPS, BATCH, 5)
    write("", net, inputs, inputs[0, 0], hidden=1)


def conv() -> None:
    torch.manual_seed(1)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, kernel_size=3, stride=(2, 1), padding=1),  # [4, 4, 6]
        leaky(0.9, (4, 4, 6)),
        torch.nn.Conv2d(4, 6, kernel_size=3, padding=(1, 2), dilation=(1, 2)),  # [6, 4, 6]
        leaky(0.8, (6, 4, 6)),
        torch.nn.Flatten(),
        torch.nn.Linear(144, 5),
        leaky(0.85, (5,), output=True),
    )
    with torch.no_grad():
        net[0].weight.mul_(2.0)
        net[2].weight.mul_(2.5)
        net[5].weight.mul_(8.0)
    inputs = torch.rand(STEPS, BATCH, 2, 8, 6)
    # A batch of one with the batch axis ignored: NIR types describe one example.
    write("_conv", net, inputs, inputs[0, :1], hidden=1, ignore_dims=[0])


def rleaky() -> None:
    torch.manual_seed(2)
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        snn.RLeaky(beta=0.85, threshold=1.0, linear_features=8, reset_mechanism="zero",
                   init_hidden=True, reset_delay=False),
        torch.nn.Linear(8, 3),
        leaky(0.9, (3,), output=True),
    )
    with torch.no_grad():
        net[0].weight.mul_(2.0)
        net[1].recurrent.weight.mul_(3.0)
        net[1].recurrent.bias.mul_(0.5)
        net[2].weight.mul_(4.0)
    inputs = torch.rand(STEPS, BATCH, 5)
    write("_rleaky", net, inputs, inputs[0, 0], hidden=1)


if __name__ == "__main__":
    dense()
    conv()
    rleaky()
