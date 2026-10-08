"""Write FLYNN's connectome RNN activity and gradients from Wang and Chen's own cell for sparx's parity test.

Builds their `LeakyConnectomeRNNCell` (`models/connectome_rnn_model.py` of
github.com/ben-gitdev/fly-gym, the code of Wang and Chen's FLYNN, arXiv
2607.00025) on a small random connectome in float64: 40 neurons, 240
signed synapse counts of 5 to 30 synapses (30 % inhibitory), five cell types
and unknown, the weights rescaled by their power iteration's estimate of the
spectral radius (`core/utils.py`; it estimates 20.1 where the radius is
48.1, so the radius it leaves is 2.15, not their target 0.9), 6 input and 5
output neurons, and each type's leak logit drawn at random. It runs 15
steps from rest on random inputs, records the output neurons' activity
every step, and backpropagates a loss weighing each step's outputs by
random coefficients to the weights (in their CSR order, saved with the
edges), the biases and the leak logits.

Saves to `tests/fixtures/flynn.npz`. The committed fixture came from fly-gym
8d964599119b420de67ee259434d273651a1a76d with torch 2.14.1+cpu:

    git clone https://github.com/ben-gitdev/fly-gym ../ref-fly-gym
    python tools/make_flynn_fixtures.py [path to the cloned repository]
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from lifted import lift
from references import require, require_checkout

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "flynn.npz"
NEURONS, EDGES, TYPES, INPUTS, OUTPUTS, STEPS, BATCH = 40, 240, 5, 6, 5, 15, 3


def spectral_rescaling(path: Path):
    """Their `rescale_spectral_radius_` and the power iteration it calls, lifted from `core/utils.py`,
    which imports pandas for its table readers."""
    names = ("spectral_radius_power_iter", "rescale_spectral_radius_")
    return lift(path, *names, torch=torch, math=math)["rescale_spectral_radius_"]


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ref-fly-gym"
    require("torch")
    require_checkout(repo, "fly-gym")
    sys.path.insert(0, str(repo))
    rescale_spectral_radius_ = spectral_rescaling(repo / "core" / "utils.py")
    from models.connectome_rnn_model import LeakyConnectomeRNNCell

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pairs = rng.choice(NEURONS * NEURONS, EDGES, replace=False)
    pre, post = pairs % NEURONS, pairs // NEURONS
    synapses = rng.integers(5, 31, EDGES) * rng.choice([-1, 1], EDGES, p=[0.3, 0.7])
    counts = torch.as_tensor(synapses, dtype=torch.float64)
    weights = torch.sparse_coo_tensor(torch.as_tensor(np.stack([post, pre])), counts, (NEURONS, NEURONS))
    weights = weights.coalesce()
    rescale_spectral_radius_(weights, target=0.9)
    types = torch.as_tensor(rng.integers(0, TYPES + 1, NEURONS))
    inputs = rng.choice(NEURONS, INPUTS, replace=False)
    outputs = rng.choice(NEURONS, OUTPUTS, replace=False)
    cell = LeakyConnectomeRNNCell(weights, inputs.tolist(), outputs.tolist(), types, TYPES + 1,
                                  leak_alpha=0.2, train_rnn_weights=True, dtype=torch.float64)
    with torch.no_grad():
        cell.alpha_logits.copy_(torch.randn(TYPES + 1))
        cell.bias.copy_(0.1 * torch.randn(NEURONS))
    drive = torch.randn(STEPS, BATCH, INPUTS)
    coefficients = torch.randn(STEPS, BATCH, OUTPUTS)
    _, _, activity = cell(torch.zeros(BATCH, NEURONS), drive, store_sequence=True)
    loss = (activity * coefficients).sum()
    loss.backward()
    # Their weights run in CSR order, by postsynaptic then presynaptic neuron; the edges are saved so.
    rows = np.repeat(np.arange(NEURONS), np.diff(cell.W_crow_indices.numpy()))
    order = np.lexsort((pre, post))
    weight = cell.W_values.detach().numpy()
    np.savez_compressed(
        OUT, pre=cell.W_col_indices.numpy(), post=rows, synapses=synapses[order], types=types.numpy(),
        inputs=inputs, outputs=outputs, weight=weight, bias=cell.bias.detach().numpy(),
        alpha_logits=cell.alpha_logits.detach().numpy(), drive=drive.numpy(),
        coefficients=coefficients.numpy(), activity=activity.detach().numpy(), loss=loss.detach().numpy(),
        grad_weight=cell.W_values.grad.numpy(),
        grad_bias=cell.bias.grad.numpy(), grad_alpha_logits=cell.alpha_logits.grad.numpy(),
        scale=np.asarray(weight[0] / synapses[order][0]), **{"meta/torch": np.array(torch.__version__)})
    print(f"wrote {OUT}; loss {loss.item():.6f}, mean |activity| {activity.abs().mean().item():.3f}")


if __name__ == "__main__":
    main()
