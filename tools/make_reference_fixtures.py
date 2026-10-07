"""Write SpikingJelly's outputs and gradients for sparx's parity tests.

Runs the reference implementations in PyTorch on fixed inputs and saves
inputs, parameters, spikes and gradients to `tests/fixtures/spikingjelly.npz`,
which `tests/test_reference.py` compares sparx against.

Needs torch and a SpikingJelly checkout that has `neuron/psn.py` (the PyPI
release 0.0.0.0.14 predates it). The committed fixture came from commit
c6cb8e46738bf6010cb94904fe5995e66d6451fc with torch 2.14.1+cpu:

    git clone https://github.com/fangwei123456/spikingjelly ../spikingjelly
    git -C ../spikingjelly checkout c6cb8e46738bf6010cb94904fe5995e66d6451fc
    pip install torch torchvision loguru packaging    # SpikingJelly's own imports
    PYTHONPATH=../spikingjelly python tools/make_reference_fixtures.py

Every case asserts that no membrane comes within `MARGIN` of its threshold,
so float32 rounding in either framework cannot flip a spike, and that the
case fires a meaningful share of spikes.
"""

import subprocess
from pathlib import Path

import numpy as np
import spikingjelly
import torch
from spikingjelly.activation_based import neuron, surrogate

MARGIN = 1e-4
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "spikingjelly.npz"


def _check(margin: float, spikes: np.ndarray, name: str) -> None:
    rate = spikes.mean()
    assert margin > MARGIN, f"{name}: a membrane is {margin:.2e} from threshold"
    assert 0.05 < rate < 0.95, f"{name}: firing rate {rate:.2f} tests little"


def _lif_margin(x: np.ndarray, tau: float, hard: bool) -> float:
    """The closest approach of the charged membrane to threshold 1, in float64."""
    v = np.zeros(x.shape[1:])
    closest = np.inf
    for xt in x.astype(np.float64):
        h = (1 - 1 / tau) * v + xt
        closest = min(closest, np.abs(h - 1).min())
        s = (h >= 1).astype(np.float64)
        v = h * (1 - s) if hard else h - s
    return closest


def lif_cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    cases = {}
    for name, tau, hard, detach in (("lif_soft", 2.0, False, False), ("lif_hard", 3.0, True, False),
                                    ("lif_soft_detached", 2.0, False, True)):
        x = rng.normal(0.4, 0.8, (16, 2, 5)).astype(np.float32)
        weights = rng.normal(size=x.shape).astype(np.float32)
        node = neuron.LIFNode(tau=tau, decay_input=False, v_threshold=1.0, v_reset=0.0 if hard else None,
                              surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=detach,
                              step_mode="m", backend="torch")
        xt = torch.tensor(x, requires_grad=True)
        spikes = node(xt)
        (spikes * torch.tensor(weights)).sum().backward()
        out = spikes.detach().numpy()
        _check(_lif_margin(x, tau, hard), out, name)
        cases |= {f"{name}/x": x, f"{name}/weights": weights, f"{name}/tau": np.float32(tau),
                  f"{name}/spikes": out, f"{name}/v": node.v.detach().numpy(),
                  f"{name}/grad_x": xt.grad.numpy()}
    return cases


def _psn_case(name: str, module: torch.nn.Module, x: np.ndarray, rng: np.random.Generator,
              weight_mix) -> dict[str, np.ndarray]:
    """Run `module` on `x`; `weight_mix(weight)` is the matrix it applies, for the margin check."""
    weights = rng.normal(size=x.shape).astype(np.float32)
    # A bias drawn near zero instead of the initial -1, so the case fires
    # often and its threshold gradient has many terms.
    with torch.no_grad():
        bias = rng.normal(-0.3, 0.3, tuple(module.bias.shape))
        module.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    xt = torch.tensor(x, requires_grad=True)
    spikes = module(xt)
    (spikes * torch.tensor(weights)).sum().backward()
    out = spikes.detach().numpy()
    w = weight_mix(module.weight.detach().numpy().astype(np.float64))
    h = np.tensordot(w, x.astype(np.float64), axes=(1, 0))
    bias = module.bias.detach().numpy().astype(np.float64)
    h = h + bias.reshape(bias.shape[:1] + (1,) * (x.ndim - 1)) if bias.ndim else h + bias
    _check(np.abs(h).min(), out, name)
    return {f"{name}/x": x, f"{name}/weights": weights, f"{name}/weight": module.weight.detach().numpy(),
            f"{name}/bias": module.bias.detach().numpy().reshape(-1 if bias.ndim else ()),
            f"{name}/spikes": out, f"{name}/grad_x": xt.grad.numpy(),
            f"{name}/grad_weight": module.weight.grad.numpy(),
            f"{name}/grad_bias": module.bias.grad.numpy().reshape(-1 if bias.ndim else ())}


def psn_cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    steps = 8
    cases = {}
    psn = neuron.PSN(T=steps, surrogate_function=surrogate.ATan())
    cases |= _psn_case("psn", psn, rng.normal(0.5, 1.0, (steps, 3, 4)).astype(np.float32), rng,
                       lambda w: w)
    for masking in (0.4, 1.0):
        masked = neuron.MaskedPSN(k=3, T=steps, lambda_init=masking, surrogate_function=surrogate.ATan(),
                                  step_mode="m")
        mask = masked.mask0.numpy().astype(np.float64)
        x = rng.normal(0.5, 1.0, (steps, 3, 4)).astype(np.float32)
        cases |= _psn_case(f"masked_psn_{masking}", masked, x, rng,
                           lambda w, mask=mask, m=masking: (m * mask + (1 - m)) * w)
        cases[f"masked_psn_{masking}/masking"] = np.float32(masking)
    for exp_init in (True, False):
        sliding = neuron.SlidingPSN(k=3, exp_init=exp_init, surrogate_function=surrogate.ATan(),
                                    step_mode="m", backend="gemm")
        with torch.no_grad():
            sliding.weight.add_(torch.tensor(rng.normal(0, 0.2, 3), dtype=torch.float32))
        name = f"sliding_psn_{'exp' if exp_init else 'kaiming'}"
        cases |= _psn_case(name, sliding, rng.normal(0.5, 1.0, (10, 3, 4)).astype(np.float32), rng,
                           lambda w, s=sliding: s.gen_gemm_weight(10).detach().numpy().astype(np.float64)
                           if w.ndim == 1 else w)
    return cases


def sew_cases(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """SpikingJelly's SEW BasicBlock, with and without its downsampling shortcut, for each connect
    function, in float64 with BatchNorm in evaluation mode and integrate-and-fire neurons
    (hard reset to zero), multi-step over time-major input."""
    from spikingjelly.activation_based import functional, layer
    from spikingjelly.activation_based.model.sew_resnet import BasicBlock

    cases = {}
    for name, inplanes, planes, stride in (("sew_same", 8, 8, 1), ("sew_down", 8, 16, 2)):
        for cnf in ("ADD", "AND", "IAND"):
            downsample = None
            if stride != 1 or inplanes != planes:
                downsample = torch.nn.Sequential(layer.Conv2d(inplanes, planes, 1, stride, bias=False),
                                                 layer.BatchNorm2d(planes))
            block = BasicBlock(inplanes, planes, stride, downsample, cnf=cnf, spiking_neuron=neuron.IFNode,
                               surrogate_function=surrogate.ATan(), detach_reset=True).double()
            with torch.no_grad():
                for module in block.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.running_mean.uniform_(-0.2, 0.2)
                        module.running_var.uniform_(0.5, 1.5)
                        module.weight.uniform_(0.5, 1.5)
                        module.bias.uniform_(0.2, 1.0)
            block.eval()
            functional.set_step_mode(block, "m")
            x = (rng.uniform(size=(3, 2, inplanes, 8, 8)) < 0.5).astype(np.float64)
            out = block(torch.tensor(x)).detach().numpy()
            key = f"{name}_{cnf.lower()}"
            cases[f"{key}/x"] = x.transpose(0, 1, 3, 4, 2)  # time-major, channels last
            cases[f"{key}/out"] = out.transpose(0, 1, 3, 4, 2)
            for param, value in block.state_dict().items():
                if value.dtype.is_floating_point:
                    cases[f"{key}/{param}"] = value.numpy()
            assert 0.05 < out.mean() < 0.95, f"{key}: output rate {out.mean():.2f} tests little"
    return cases


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    source = Path(spikingjelly.__file__).resolve().parent.parent
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    cases = lif_cases(rng) | psn_cases(rng) | sew_cases(rng)
    cases["meta/spikingjelly_commit"] = np.array(commit)
    cases["meta/torch"] = np.array(torch.__version__)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **cases)
    print(f"wrote {len(cases)} arrays to {OUT} (spikingjelly {commit[:8]}, torch {torch.__version__})")


if __name__ == "__main__":
    main()
