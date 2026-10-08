"""Write SNN-delays' whole network, loss and gradients for sparx's parity tests.

Builds `SnnDelays` from Hammouamri et al.'s code (github.com/Thvnvtos/SNN-delays at d169b4e3,
ICLR 2024) on a small configuration of their SHD recipe: two hidden layers
with every synapse a DCLS `Dcls1d`, batch norm, LIF neurons with SpikingJelly's
`decay_input=False`, hard reset and `detach_reset`, ATan(5), no bias,
Kaiming-uniform weights, the inputs padded left by `K - 1` and right by
`(K - 1) // 2`, and the non-spiking LIF readout scored by `loss='sum'`. It
records a training step (the Gaussian kernels at a width, batch statistics)
and an evaluation as their `eval_model` runs it (zero width, rounded
positions, running statistics), one recording binned by their
SpikingJelly's `integrate_events_by_fixed_duration_shd`, and the learning
rates, Adam momentum and width their full SHD configuration trains each of
its 150 epochs with. Saves to `tests/fixtures/snn_delays.npz`, which
`tests/test_reference.py` compares `sparx.models.SpikingMLP`,
`sparx.datasets.bin_events` and dew's schedules against.

Their code needs the SpikingJelly of 2023 they ran on (its SHD frames are
event-anchored, later releases bin on a grid), DCLS and torch. SpikingJelly
6fbee6ed34ed5a65187f4721d1a412f6a526ca6a (2023-12-04) reproduces the
committed fixture, as does 6dca147a (2023-05-30); their `shd.py` matches
the hash the fixture records. Commits fb03f787 (2023-12-05) to 06303ba8
(2024-02-19) refuse the network's input to `layer.BatchNorm1d`, whose shape
check always fails there:

    git -C <SpikingJelly checkout> checkout 6fbee6ed34ed5a65187f4721d1a412f6a526ca6a
    uv pip sync tools/environments/torch.txt
    PYTHONPATH=<SpikingJelly checkout>:<SNN-delays checkout> python tools/make_snn_delays_fixtures.py

`model.py` imports wandb, and `datasets.py` torchaudio, at module level;
neither runs here, so stubs on the path do.
Dropout is 0, since a random mask cannot be matched; the tests check the
sequence-held mask on its own.
"""

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np
import spikingjelly
import torch
import torch.nn.functional as F
from config import Config
from references import require, require_checkout
from snn_delays import SnnDelays
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.datasets.shd import integrate_events_by_fixed_duration_shd

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "snn_delays.npz"
STEPS, BATCH, INPUTS, HIDDEN, CLASSES = 9, 4, 6, 5, 3
SIG = 1.3  # DCLS's raw width in the training step; the effective width is SIG + 0.27


class Small(Config):
    seed = 0
    n_inputs = INPUTS
    n_hidden_layers = 2
    n_hidden_neurons = HIDDEN
    n_outputs = CLASSES
    dropout_p = 0.0
    max_delay = 5  # 50 ms at a 10 ms step, odd as theirs
    sigInit = max_delay // 2
    left_padding = max_delay - 1
    right_padding = (max_delay - 1) // 2
    init_pos_a = -max_delay // 2
    init_pos_b = max_delay // 2
    surrogate_function = surrogate.ATan(alpha=5.0)


def lif_inputs(model: SnnDelays) -> list[list[torch.Tensor]]:
    """Record what each hidden LIF receives, to measure how close its membrane comes to threshold."""
    seen: list[list[torch.Tensor]] = [[] for _ in range(Small.n_hidden_layers)]
    for layer in range(Small.n_hidden_layers):
        node = model.blocks[layer][1][0]
        node.register_forward_hook(
            lambda module, args, out, layer=layer: seen[layer].append(args[0].detach()))
    return seen


def margin(seen: list[torch.Tensor]) -> float:
    """The smallest distance of a hidden membrane from threshold before it fires or not."""
    keep = 1 - 1 / Small.init_tau
    closest = np.inf
    for x in seen:
        v = torch.zeros_like(x[0])
        for t in range(x.shape[0]):
            v = keep * v + x[t]
            closest = min(closest, float((v - Small.v_threshold).abs().min()))
            v = torch.where(v >= Small.v_threshold, torch.zeros_like(v), v)
    return closest


def record(seed: int) -> tuple[dict[str, np.ndarray], float]:
    """The fixture for inputs drawn from `seed`, and how close a hidden membrane came to threshold."""
    config = Small()
    model = SnnDelays(config)
    seen = lif_inputs(model)
    rng = np.random.default_rng(seed)
    x = rng.poisson(0.8, size=(STEPS, BATCH, INPUTS)).astype(np.float32)
    labels = np.array([0, 2, 1, 2])
    weights = {f"layer{i}/weight": block[0][0].weight.detach().numpy()[..., 0].copy()
               for i, block in enumerate(model.blocks)}
    positions = {f"layer{i}/P": block[0][0].P.detach().numpy()[0, :, :, 0].copy()
                 for i, block in enumerate(model.blocks)}

    # A training step: Gaussian kernels of width SIG, batch statistics.
    model.train()
    with torch.no_grad():
        for block in model.blocks:
            block[0][0].SIG.fill_(SIG)
    xt = torch.tensor(x, requires_grad=True)
    out = model(xt)
    loss = model.calc_loss(out, F.one_hot(torch.tensor(labels), CLASSES).float())
    loss.backward()
    train = {"train/out": out.detach().numpy(), "train/loss": loss.detach().numpy(),
             "train/grad_x": xt.grad.numpy()}
    for i, block in enumerate(model.blocks):
        train[f"train/grad_weight{i}"] = block[0][0].weight.grad.numpy()[..., 0]
        train[f"train/grad_P{i}"] = block[0][0].P.grad.numpy()[0, :, :, 0]
        if i < Small.n_hidden_layers:
            bn = block[0][1]
            train[f"train/grad_bn_weight{i}"] = bn.weight.grad.numpy()
            train[f"train/grad_bn_bias{i}"] = bn.bias.grad.numpy()
            train[f"train/running_mean{i}"] = bn.running_mean.numpy().copy()
            train[f"train/running_var{i}"] = bn.running_var.numpy().copy()
    trained_margin = margin([seen[i][-1] for i in range(Small.n_hidden_layers)])

    # Evaluation as their eval_model runs it.
    functional.reset_net(model)
    model.eval()
    with torch.no_grad():
        for block in model.blocks:
            block[0][0].SIG *= 0
            block[0][0].version = "max"
            block[0][0].DCK.version = "max"
        model.round_pos()
        out = model(torch.tensor(x))
        loss = model.calc_loss(out, F.one_hot(torch.tensor(labels), CLASSES).float())
    evaluated = {"eval/out": out.numpy(), "eval/loss": loss.numpy()}
    evaluated |= {f"eval/P{i}": block[0][0].P.detach().numpy()[0, :, :, 0].copy()
                  for i, block in enumerate(model.blocks)}
    closest = min(trained_margin, margin([seen[i][-1] for i in range(Small.n_hidden_layers)]))

    # One recording with silences longer than a frame, binned as their SHD loader bins it.
    times = np.sort(rng.uniform(0, 0.3, 80)).astype(np.float16)
    times[40:] += np.float16(0.05)
    units = rng.integers(0, 700, 80).astype(np.uint16)
    frames = integrate_events_by_fixed_duration_shd({"t": times, "x": units}, 10, 700)

    source = Path(inspect.getfile(integrate_events_by_fixed_duration_shd)).read_bytes()
    cases = {"x": x, "labels": labels, "tau": np.float64(Small.init_tau), "sig": np.float32(SIG), **weights,
             **positions, **train, **evaluated, "margin": np.float64(closest),
             "events/times": times, "events/units": units, "events/frames": frames,
             "meta/spikingjelly": np.array(str(Path(spikingjelly.__file__).parent)),
             "meta/spikingjelly_shd_sha256": np.array(hashlib.sha256(source).hexdigest()),
             "meta/dcls": np.array(importlib.metadata.version("dcls")),
             "meta/torch": np.array(torch.__version__), "meta/seed": np.array(seed)}
    return cases, closest


def schedules() -> dict[str, np.ndarray]:
    """Their full SHD configuration's per-epoch rates, Adam momentum and raw DCLS width.

    Their training loop steps each scheduler and `decrease_sig` once at the
    end of every epoch, so entry `e` is what epoch `e` trains with.
    """
    model = SnnDelays(Config())
    optimizers = model.optimizers()
    stepping = model.schedulers(optimizers)
    rows = []
    for epoch in range(Config.epochs):
        weights, positions = optimizers[0].param_groups[0], optimizers[1].param_groups[0]
        rows.append((weights["lr"], weights["betas"][0], positions["lr"],
                     float(model.blocks[-1][0][0].SIG[0, 0, 0, 0])))
        for scheduler in stepping:
            scheduler.step()
        model.decrease_sig(epoch)
    lr_w, b1, lr_pos, sig = (np.array(column) for column in zip(*rows, strict=True))
    return {"schedule/epochs": np.array(Config.epochs), "schedule/lr_w": lr_w, "schedule/b1": b1,
            "schedule/lr_pos": lr_pos, "schedule/sig": sig}


def main():
    require("torch", "dcls")
    require_checkout(Path(spikingjelly.__file__).resolve().parent.parent, "spikingjelly-2023")
    require_checkout(Path(inspect.getfile(SnnDelays)).resolve().parent, "SNN-delays")
    # The first inputs whose hidden membranes all stay 1e-3 from threshold, so
    # float32 rounding cannot flip a spike between the two implementations.
    for seed in range(100):
        cases, closest = record(seed)
        if closest > 1e-3:
            break
    else:
        raise RuntimeError("no seed below 100 keeps the membranes away from threshold")
    np.savez_compressed(OUT, **cases, **schedules())
    print(f"wrote {OUT} (seed {seed}, dcls {importlib.metadata.version('dcls')}, torch {torch.__version__}, "
          f"closest membrane {closest:.2e} from threshold)")


if __name__ == "__main__":
    main()
