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
    np.savez_compressed(OUT, latency_input=data, latency=latency, delta_input=signal, delta=delta,
                        delta_on=delta_on, loss_outputs=outputs, loss_spikes=spikes, loss_labels=labels,
                        ce_rate=ce_rate, mse_count=mse_count, snntorch=np.array(snntorch.__version__),
                        torch=np.array(torch.__version__))
    print(f"wrote {OUT} (snntorch {snntorch.__version__}, torch {torch.__version__})")


if __name__ == "__main__":
    main()
