"""Write a NIR graph exported by snnTorch, with snnTorch's outputs, for sparx's NIR round trip.

A two-layer snnTorch network (`Linear`, `Leaky` with a hard reset,
`Linear`, `Leaky`) is exported with `snntorch.export_nir` to
`tests/fixtures/snntorch.nir`, and run on random input; the input and its
spikes go to `tests/fixtures/nir.npz`. `tests/test_nir.py` imports the graph
into sparx, runs it, and exports it back.

    pip install snntorch nir
    python tools/make_nir_fixtures.py
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


def main():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        snn.Leaky(beta=torch.full((8,), 0.85), threshold=torch.ones(8), reset_mechanism="zero",
                  init_hidden=True, reset_delay=False),
        torch.nn.Linear(8, 3),
        snn.Leaky(beta=torch.full((3,), 0.9), threshold=torch.ones(3), reset_mechanism="zero",
                  init_hidden=True, reset_delay=False, output=True),
    )
    with torch.no_grad():
        net[0].weight.mul_(2.5)
        net[2].weight.mul_(2.5)
    inputs = torch.rand(STEPS, BATCH, 5)
    graph = export_to_nir(net, inputs[0, 0])
    nir.write(FIXTURES / "snntorch.nir", graph)
    snn_utils.reset(net)
    outputs = []
    with torch.no_grad():
        for t in range(STEPS):
            spikes, _ = net(inputs[t])
            outputs.append(spikes.numpy())
    outputs = np.stack(outputs)
    np.savez_compressed(FIXTURES / "nir.npz", inputs=inputs.numpy(), spikes=outputs,
                        meta_snntorch=np.array(snn.__version__))
    print("wrote", FIXTURES / "snntorch.nir", "output spikes", outputs.sum(), "nodes", list(graph.nodes))


if __name__ == "__main__":
    main()
