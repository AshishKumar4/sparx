"""Train a spiking classifier with dew's recipe machinery (`sparx.config.SNNRunConfig`).

    python recipes/snn/train.py --data.channels 140 --trainer.batch-size 64 --trainer.steps 3000 \\
        --trainer.eval-every 500 --trainer.checkpoint-dir runs --trainer.name shd \\
        --model.hidden 128 --model.classes 20 --model.delays 15 \\
        --model.neuron '{"class": "sparx.nn.neurons:ALIF", "fields": {"tau": 5.0, "tau_adapt": 20.0,
                         "beta": 0.2, "learn_tau": true, "detach_reset": true}}' \\
        --objective.schedules '{"sigma": {"class": "linear", "fields": {"peak": 7.5, "end": 0.5}}}'
    JAX_PLATFORMS=cpu python recipes/snn/train.py --smoke --trainer.checkpoint-dir /tmp/snn-smoke

The run is dew's `RunConfig` (model, data, optimizer, trainer, objective) plus
the encoder and the field it reads, and `run.json` records all of it, so
`dew.pipeline(run_dir, trust=("sparx",))` loads the trained classifier back,
and `dew train run_dir/run.json --trust sparx` rebuilds the run and trains on
from its latest checkpoint. The model is any spiking model over `[T, B,
channels]` by import path (`--model my_package.models:Net`), the encoder any
of `sparx.encode`'s, and the schedules are records of dew's schedules.

`--smoke` trains a small network for a few steps on synthetic recordings in
SHD's layout (`sparx.datasets.write_synthetic_shd`), written under the
checkpoint directory, so it needs no download.
"""

from sparx.config import SNNRunConfig

if __name__ == "__main__":
    SNNRunConfig.cli().run()
