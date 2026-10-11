# Changelog

sparx's releases, newest first. The version is `sparx.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## 0.1.1

Fixes and additions from GitHub issues #1 to #5, on the same dependencies as
0.1.0.

- `datasets.bin_events` refuses an event time that is negative or not
  finite, and a channel that is not an integer in `[0, source_channels)`;
  before, NumPy's negative indices put such events in the last bins (#1).
- `nir.to_nir` exports a stack whose layers declare no parameters, such as
  fixed neurons alone, and names a layer that has parameters but was given
  none (#2).
- `nn.Recurrent` and `nn.Homeostatic` left without `dt` step at their
  neuron's (`nn.INHERITED`); a `dt` given must still equal it (#3).
- `nir.to_nir` exports any model that states itself as a sequential stack
  (`nir.Sequentialized`), and `models.SpikingMLP.as_sequential` does, for a
  network without delays or batch norm (#4). Export refuses a weight layer
  that holds or computes in other than float32, and recurrence around other
  than LIF, which NIR cannot state.
- NIR carries time constants learned per feature (`learn_tau=True`) both
  ways: they export as per-neuron `tau`, and a node whose `tau` differs
  between channels imports as a learned decay, trainable as before (#5).

Outside the package, the repository gained the racer study
(`research/racer`): a spiking convolutional network that drives every one of
200 unseen tracks at five seeds of five, beside graded, few-bit, sigma-delta
and dendritic variants, and the network the site's racer now runs.

## 0.1.0

The first release on PyPI, as `sparxml`: `uv pip install sparxml` for the CPU,
with the `cuda12`, `cuda13` and `tpu` extras for accelerators, on Python 3.12
or newer, over dew 0.1 (`dewml`).

- One neuron protocol for machine learning and neuroscience: dimensionless
  cells (LIF, ALIF, synaptic, rate, escape-noise and RNeuralNet's pulse
  units) and physical models in mV and ms (LIF, AdEx, Izhikevich,
  Hodgkin-Huxley, graded-potential neurons), with current, conductance,
  graded and stochastic synapses, gap junctions and neuromodulators.
- Flax layers over time-major arrays, with surrogate gradients, learned
  delays (DCLS), parallel spiking neurons, and one recurrence over dense,
  sparse and delayed wirings with Hebbian fast weights; `SpikingMLP` and
  `SEWResNet`, which declare their parameters' logical axes to dew's layout.
- Learning: surrogate BPTT, e-prop, OTTT, exact spike-time gradients
  (EventProp), REINFORCE, predictive coding and PC-ALM, reward diffusion,
  ANN-to-SNN conversion, and STDP, triplet, dopamine-modulated and
  short-term plasticity in simulation; homeostasis by intrinsic plasticity
  of any spiking model's threshold and by synaptic scaling, with `Rules`
  to run several plasticity rules on one projection.
- Circuits: populations and projections on one clock with dense, edge-list
  and event delivery, NEST's connection rules, `simulate` in compiled chunks
  over dew's mesh with checkpoints and continued trials, the cortical
  microcircuit, Brunel's network, and whole-fly-brain connectomes.
- dew integration: objectives for classification, activity fitting, e-prop,
  predictive coding and rewards on dew's `Trainer`; runs that reload with
  `dew.pipeline(run, trust=("sparx",))`; a stream server for many sessions;
  NIR export and import.
- Evidence: every model is checked against its reference (NEST, Brian2, the
  authors' code), with fixtures regenerated in locked environments
  (`tools/references.py`); `docs/status.md` states what is supported and
  what is not. Hammouamri et al.'s SHD recipe, three seeds beside the
  authors' code on an A100: 93.99 ± 0.29% at the last epoch against its
  93.89 ± 0.26% (`research/shd`).
- [sparxml.dev](https://sparxml.dev): a fifteen-chapter course whose figures
  run sparx's models in the browser, held to sparx's float64 runs, beside the
  docs and an API reference generated from the source.
