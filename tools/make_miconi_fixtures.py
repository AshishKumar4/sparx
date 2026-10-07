"""Write fast-weight trajectories and gradients from Miconi et al.'s own networks for sparx's parity test.

Four networks, one per Hebbian rule, each run in float64 from zero activity
and zero traces, with the gradient of a loss over the run:

- `hebb`, differentiable plasticity's decaying trace (Miconi et al. 2018,
  eq. 2): the `NETWORK` of `simple/simple.py`, on an episode of its own
  pattern-completion task (`generateInputsAndTarget`) shrunk to 8-bit
  patterns, with its loss, the squared error of the last step's activity.
- `oja`, their Oja's rule (2018, eq. 3): the `Network` of `maze/maze.py`
  with `type="plastic"` and `rule="oja"`.
- `modulated`, Backpropamine's simple neuromodulation (Miconi et al. 2019,
  eq. 3, with the fan-out of its appendix): the `Network` of
  `simplemaze/maze.py`. Its fan-out weights are scaled up so the clip at
  2 binds on part of the trace.
- `retroactive`, Backpropamine's eligibility trace (2019, eqs. 4 and 5):
  the `Network` of `maze/batch.py` with `type="modul"`, `da="tanh"` and
  `addpw=3`, the hard clip at 1. Its modulator weights are scaled up so the
  clip binds on part of the trace.

Each class is lifted out of its script's syntax tree, since the scripts
train at import or import cluster tools, and `.cuda()` is the identity
while they build, so they run on the CPU. The last three take random inputs
through their input layer and weigh every step's activity by random
coefficients for the loss. Their weights are drawn at the scale a trained
network reaches, so the recurrence and the trace both shape the activity.
Matrices are saved in sparx's layout, `[pre, post]`; `maze/batch.py`
keeps `[post, pre]`, and every input layer's weight is transposed to
`[in, out]`.

Saves all four to `tests/fixtures/miconi.npz`. The committed fixture came
from differentiable-plasticity 5bd29a18 and backpropamine 180c9101, with
torch 2.14.1+cpu:

    pip install torch
    python tools/make_miconi_fixtures.py [differentiable-plasticity] [backpropamine]

The repositories default to `ref-differentiable-plasticity` and
`ref-backpropamine` cloned next to sparx.
"""

import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lifted import lift
from torch import nn
from torch.autograd import Variable

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "miconi.npz"
# simple.py's task, shrunk: 2 patterns of 8 bits, each shown twice for 3 steps with 2 blank steps after.
TASK = {"PATTERNSIZE": 8, "NBNEUR": 9, "PROBADEGRADE": 0.5, "NBPATTERNS": 2, "NBPRESCYCLES": 2,
        "PRESTIME": 3, "PRESTIMETEST": 3, "INTERPRESDELAY": 2}
TASK["NBSTEPS"] = TASK["NBPRESCYCLES"] * (TASK["PRESTIME"] + TASK["INTERPRESDELAY"]) * TASK["NBPATTERNS"] \
    + TASK["PRESTIMETEST"]
# Both maze scripts see a 3 x 3 view, 4 extra inputs and their last action of 4.
MAZE = {"NBACTIONS": 4, "NBDA": 1, "TOTALNBINPUTS": 17}
STEPS, BATCH, INPUTS, HIDDEN = 20, 3, 5, 6


def lifted(path: Path, name: str, **constants: object) -> Callable:
    """The class or function `name` of the script at `path`, over the modules their scripts import."""
    modules = {"torch": torch, "nn": nn, "F": F, "Variable": Variable, "np": np, "random": random}
    found = lift(path, name, **modules, **constants)[name]
    assert callable(found)
    return found


def on_cpu(build: Callable[[], nn.Module]) -> nn.Module:
    """`build()` with `.cuda()` the identity on tensors and modules."""
    tensor, module = torch.Tensor.cuda, nn.Module.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    nn.Module.cuda = lambda self, *args, **kwargs: self
    try:
        return build()
    finally:
        torch.Tensor.cuda, nn.Module.cuda = tensor, module


def numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy().copy()


def clipped(traces: np.ndarray, bound: float) -> float:
    """The fraction of a trace's entries the clip holds at some step; asserted to be part of them."""
    held = float(np.mean((np.abs(traces) == bound).any(axis=0)))
    assert 0.05 < held < 0.95, f"the clip holds {held:.0%} of the trace; it should hold part"
    return held


def saved(parameters: dict[str, tuple[torch.Tensor, bool]]) -> dict[str, np.ndarray]:
    """Each parameter and its gradient, transposed to `[pre, post]` or `[in, out]` where marked."""
    case = {}
    for name, (parameter, transpose) in parameters.items():
        value, grad = numpy(parameter), parameter.grad.numpy().copy()
        case[name], case[f"grad_{name}"] = (value.T.copy(), grad.T.copy()) if transpose else (value, grad)
    return case


def hebb(plasticity: Path) -> dict[str, np.ndarray]:
    script = plasticity / "simple" / "simple.py"
    episode = lifted(script, "generateInputsAndTarget", ttype=torch.DoubleTensor, **TASK)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    inputs, target = episode()
    net = lifted(script, "NETWORK", ttype=torch.DoubleTensor, **TASK)()
    size = TASK["NBNEUR"]
    net.w = (0.1 * torch.randn(size, size)).requires_grad_()
    net.alpha = (0.3 * torch.randn(size, size)).requires_grad_()
    net.eta = torch.tensor([0.2]).requires_grad_()
    y, trace = net.initialZeroState(), net.initialZeroHebb()
    ys, traces = [], []
    for step in range(TASK["NBSTEPS"]):
        y, trace = net(Variable(inputs[step], requires_grad=False), y, trace)
        ys.append(numpy(y))
        traces.append(numpy(trace))
    loss = (y[0][:TASK["PATTERNSIZE"]] - target).pow(2).sum()
    loss.backward()
    return {"inputs": inputs.numpy(), "target": target.numpy(), "outputs": np.stack(ys),
            "traces": np.stack(traces), "loss": numpy(loss),
            **saved({"weight": (net.w, False), "alpha": (net.alpha, False), "eta": (net.eta, False)})}


def oja(plasticity: Path) -> dict[str, np.ndarray]:
    network = lifted(plasticity / "maze" / "maze.py", "Network", **MAZE)
    torch.manual_seed(1)
    net = on_cpu(lambda: network({"rule": "oja", "type": "plastic", "activ": "tanh", "hiddensize": HIDDEN}))
    with torch.no_grad():
        net.w.copy_(0.3 * torch.randn(HIDDEN, HIDDEN))
        net.alpha.copy_(0.5 * torch.randn(HIDDEN, HIDDEN))
        net.eta.fill_(0.3)
    inputs = torch.randn(STEPS, 1, MAZE["TOTALNBINPUTS"])
    coefficients = torch.randn(STEPS, 1, HIDDEN)
    hidden, trace = on_cpu(net.initialZeroState), on_cpu(net.initialZeroHebb)
    loss = torch.zeros(())
    hs, traces = [], []
    for step in range(STEPS):
        _, _, hidden, trace = net(inputs[step], hidden, trace)
        loss = loss + (hidden * coefficients[step]).sum()
        hs.append(numpy(hidden))
        traces.append(numpy(trace))
    loss.backward()
    return {"inputs": inputs.numpy(), "coefficients": coefficients.numpy(), "outputs": np.stack(hs),
            "traces": np.stack(traces), "loss": numpy(loss),
            **saved({"i2h_weight": (net.i2h.weight, True), "i2h_bias": (net.i2h.bias, False),
                     "weight": (net.w, False), "alpha": (net.alpha, False), "eta": (net.eta, False)})}


def modulated(backpropamine: Path) -> dict[str, np.ndarray]:
    network = lifted(backpropamine / "simplemaze" / "maze.py", "Network", **MAZE)
    torch.manual_seed(2)
    net = network(INPUTS, HIDDEN)
    with torch.no_grad():
        net.w.copy_(0.3 * torch.randn(HIDDEN, HIDDEN))
        net.alpha.copy_(0.5 * torch.randn(HIDDEN, HIDDEN))
        net.modfanout.weight.mul_(5.0)
    inputs = torch.randn(STEPS, BATCH, INPUTS)
    coefficients = torch.randn(STEPS, BATCH, HIDDEN)
    hidden = (net.initialZeroState(BATCH), net.initialZeroHebb(BATCH))
    loss = torch.zeros(())
    hs, traces = [], []
    for step in range(STEPS):
        _, _, hidden = net(inputs[step], hidden)
        loss = loss + (hidden[0] * coefficients[step]).sum()
        hs.append(numpy(hidden[0]))
        traces.append(numpy(hidden[1]))
    loss.backward()
    traces = np.stack(traces)
    print(f"modulated: the clip holds {clipped(traces, net.clipval):.1%} of the trace's entries at some step")
    return {"inputs": inputs.numpy(), "coefficients": coefficients.numpy(), "outputs": np.stack(hs),
            "traces": traces, "loss": numpy(loss), "clip": np.array(net.clipval),
            **saved({"i2h_weight": (net.i2h.weight, True), "i2h_bias": (net.i2h.bias, False),
                     "weight": (net.w, False), "alpha": (net.alpha, False),
                     "modulator": (net.h2mod.weight, False), "modulator_bias": (net.h2mod.bias, False),
                     "fanout": (net.modfanout.weight, False), "fanout_bias": (net.modfanout.bias, False)})}


def retroactive(backpropamine: Path) -> dict[str, np.ndarray]:
    network = lifted(backpropamine / "maze" / "batch.py", "Network", **MAZE)
    torch.manual_seed(3)
    net = on_cpu(lambda: network({"type": "modul", "bs": BATCH, "hs": HIDDEN, "da": "tanh", "addpw": 3}))
    with torch.no_grad():
        net.w.copy_(0.3 * torch.randn(HIDDEN, HIDDEN))
        net.alpha.copy_(0.5 * torch.randn(HIDDEN, HIDDEN))
        net.etaet.fill_(0.5)
        net.h2DA.weight.mul_(10.0)
    inputs = torch.randn(STEPS, BATCH, MAZE["TOTALNBINPUTS"])
    coefficients = torch.randn(STEPS, BATCH, HIDDEN)
    hidden = on_cpu(net.initialZeroState)
    unused, eligibility, trace = (on_cpu(net.initialZeroHebb), on_cpu(net.initialZeroPlasticWeights),
                                  on_cpu(net.initialZeroPlasticWeights))
    loss = torch.zeros(())
    hs, traces, eligibilities = [], [], []
    for step in range(STEPS):
        _, _, hidden, unused, eligibility, trace = net(inputs[step], hidden, unused, eligibility, trace)
        loss = loss + (hidden * coefficients[step]).sum()
        hs.append(numpy(hidden))
        traces.append(numpy(trace).swapaxes(-1, -2))
        eligibilities.append(numpy(eligibility).swapaxes(-1, -2))
    loss.backward()
    traces = np.stack(traces)
    print(f"retroactive: the clip holds {clipped(traces, 1.0):.1%} of the trace's entries at some step")
    return {"inputs": inputs.numpy(), "coefficients": coefficients.numpy(), "outputs": np.stack(hs),
            "traces": traces, "eligibilities": np.stack(eligibilities), "loss": numpy(loss),
            **saved({"i2h_weight": (net.i2h.weight, True), "i2h_bias": (net.i2h.bias, False),
                     "weight": (net.w, True), "alpha": (net.alpha, True), "eta": (net.etaet, False),
                     "modulator": (net.h2DA.weight, False), "modulator_bias": (net.h2DA.bias, False)})}


def main():
    plasticity = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ref-differentiable-plasticity"
    backpropamine = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "ref-backpropamine"
    torch.set_default_dtype(torch.float64)
    cases = {"meta/torch": np.array(torch.__version__)}
    for name, case in (("hebb", hebb(plasticity)), ("oja", oja(plasticity)),
                       ("modulated", modulated(backpropamine)), ("retroactive", retroactive(backpropamine))):
        cases.update({f"{name}/{key}": value for key, value in case.items()})
    np.savez_compressed(OUT, **cases)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
