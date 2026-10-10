# Changelog

sparx's releases, newest first. The version is `sparx.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## 0.1.0

The first release on PyPI, as `sparxml`: `pip install sparxml` for the CPU,
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
