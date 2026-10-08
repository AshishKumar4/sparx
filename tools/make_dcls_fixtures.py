"""Write DCLS-Delays' delayed-synapse outputs and gradients for sparx's parity tests.

Runs `DCLS.construct.modules.Dcls1d`, the layer Hammouamri et al. (ICLR 2024)
train delays with (github.com/Thvnvtos/SNN-delays, which pads the input on
the left by `K - 1`), in its two modes: `gauss` during training and `max`
with rounded positions and zero `SIG` at evaluation. Saves inputs,
parameters, outputs and gradients to `tests/fixtures/dcls.npz`, which
`tests/test_reference.py` compares `sparx.nn.DelayedDense` against.

    uv pip sync tools/environments/torch.txt
    python tools/make_dcls_fixtures.py

The reference's right padding, which lengthens the output by `(K - 1) // 2`
steps, is a choice of its training script, not of the layer; it is left out
here and recorded in docs/fidelity.md.
"""

import importlib.metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from DCLS.construct.modules import Dcls1d
from references import require

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "dcls.npz"
INPUTS, OUTPUTS, KERNEL, STEPS, BATCH = 4, 5, 7, 12, 2


def run(layer: Dcls1d, x: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Time-major `x` `[T, B, in]` through the layer, causally; outputs and gradients."""
    xt = torch.tensor(x, requires_grad=True)
    padded = F.pad(xt.permute(1, 2, 0), (KERNEL - 1, 0))
    out = layer(padded).permute(2, 0, 1)  # [T, B, out]
    (out * torch.tensor(weights)).sum().backward()
    grads = {"grad_x": xt.grad.numpy(), "grad_weight": layer.weight.grad.numpy()[..., 0]}
    if layer.P.grad is not None:
        grads["grad_P"] = layer.P.grad.numpy()[0, :, :, 0]
    return {"out": out.detach().numpy(), **grads}


def main():
    require("torch", "dcls")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(STEPS, BATCH, INPUTS)).astype(np.float32)
    weights = rng.normal(size=(STEPS, BATCH, OUTPUTS)).astype(np.float32)
    layer = Dcls1d(INPUTS, OUTPUTS, kernel_count=1, groups=1, dilated_kernel_size=KERNEL, bias=True,
                   version="gauss")
    with torch.no_grad():
        layer.P.uniform_(-(KERNEL // 2), KERNEL // 2)
        layer.SIG.fill_(1.3)
    params = {"weight": layer.weight.detach().numpy()[..., 0].copy(),
              "bias": layer.bias.detach().numpy().copy(),
              "P": layer.P.detach().numpy()[0, :, :, 0].copy(), "SIG": np.float32(1.3)}
    gauss = run(layer, x, weights)
    # Evaluation as the reference's eval_model does it: zero SIG, the max
    # version, positions rounded and clamped.
    layer.zero_grad()
    with torch.no_grad():
        layer.SIG *= 0
        layer.P.round_()
        layer.clamp_parameters()
    layer.version = "max"
    layer.DCK.version = "max"
    rounded = run(layer, x, weights)
    cases = {"x": x, "weights": weights, **params,
             **{f"gauss/{k}": v for k, v in gauss.items()}, **{f"rounded/{k}": v for k, v in rounded.items()},
             "rounded_P": layer.P.detach().numpy()[0, :, :, 0].copy(),
             "meta/dcls": np.array(importlib.metadata.version("dcls")),
             "meta/torch": np.array(torch.__version__)}
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT} (dcls {importlib.metadata.version('dcls')}, torch {torch.__version__})")


if __name__ == "__main__":
    main()
