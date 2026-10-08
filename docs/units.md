# Time, units and rates

sparx has two conventions for time and units, one for each half of the library, and they meet in a few places. This page lists both and where each applies.

## Steps: the trainable layers

The dimensionless cells in `sparx.dynamics.ml` and the layers in `sparx.nn` count time in steps. Every model takes a step `dt`, 1 by default, and its time constants are in the unit of `dt`.

- **Decay.** A model stores its decay per unit of time, `decay(tau) = exp(-1 / tau)`, and a step of `dt` multiplies by `decay ** dt = exp(-dt / tau)`. `LIF(tau=20.0)` at the default `dt=1` keeps `exp(-1/20)` of its membrane each step. A learned decay is stored the same way, so a checkpoint means the same at any `dt`.
- **Input.** An array input is a jump of the membrane, unitless, added unscaled each step (snnTorch's `Leaky` convention). A dimensionless model refuses a current or a conductance.
- **A step of real time.** To bin recordings in 14 ms steps and give time constants in ms, set the layer's `dt` to 14: `ALIF(tau=20.0, tau_adapt=200.0, dt=14.0)` decays by `exp(-14/20)` per step, as `ALIF(tau=20/14)` does at `dt=1`. A `Recurrent` layer steps at its neuron's `dt`, and every layer of a `SpikingMLP` steps at its neuron's `dt`, the readout included.
- **Refractoriness.** `ALIF(refractory=r)` is a duration in the unit of `dt`. The spike and the silence after it span `round(r / dt)` steps, Bellec et al.'s `n_refractory`, so `r` of 0 or one step leaves the neuron free to fire on the next step.
- **Delays.** In steps. A `Sparse` wiring's per-edge `delay` runs from 1 to `longest_delay`. `DelayedDense(features, max_delay)` learns a delay from 0 to `max_delay` per connection, and `SpikingMLP(delays=...)` gives each synapse's largest delay.
- **Encoders.** `steps` is the length of the time axis they add. `RateEncoder` fires with probability `x` per step, whatever the step stands for. Encoders read a uint8 field as `x / 255`; `EventsEncoder` passes counts unscaled.
- **Fast weights.** A Hebbian rule's `eta` is a rate per unit of time: `DecayingHebb` keeps `(1 - eta) ** dt` of its trace per step, and the other rules add `dt` times their rate.
- **Online rules.** e-prop, OTTT and REINFORCE take `dt` and time constants in its unit, as the cells do. `PulseCell` has no time constant, so an `RNeuralNet` counts ticks and ignores `dt`.

## Physical units: the simulator

The physical models in `sparx.dynamics.neurons`, the synapses and plasticity rules, and the networks of `sparx.graph` use one set of units, chosen so that `pF * mV / ms = pA` and `nS * mV = pA` hold without factors.

| Quantity | Unit |
| --- | --- |
| time, `dt`, time constants, delays, `t_ref`, `duration`, `chunk` | ms |
| voltage | mV |
| current | pA |
| conductance | nS |
| capacitance | pF |
| rates of spiking (`PoissonInput`, `PopulationRate`) | Hz |
| gating rates inside Hodgkin-Huxley | 1/ms |

- **A step.** `Network(dt=0.1)` steps every population 0.1 ms. A physical model run by itself steps `sparx.run(model, inputs, dt=...)` ms, 1 by default; `nn.Dynamics(AdEx(), dt=0.1)` runs it as a layer at 0.1 ms.
- **What reaches a membrane** (`SynapticInput`). `current` in pA, held over the step. `conductance` in nS per receptor, held over the step at its exact mean by default (`PointNeuron.hold`). `jump` in mV, added at the end of the step before the threshold test.
- **Weights.** A projection's receptor sets the unit of its weight: pA of peak current for a `"current"` receptor, nS of peak conductance for a `"conductance"` receptor, mV for a `Delta` synapse. A projection from a graded population weighs its synapses at full release.
- **Delays.** `Projection(delay=...)` is in ms and must be a whole number of steps, as must `simulate`'s `duration` and `chunk`. A delta synapse's jump lands before the threshold test, so it needs a delay of at least one step; a kinetic synapse takes 0.
- **Refractoriness.** A neuron that fires holds its reset for `round(t_ref / dt)` steps after the step it fired in, as NEST counts it. Brian2 counts one step less (see [fidelity.md](fidelity.md)).
- **Inputs.** `PoissonInput(rate=...)` is in Hz per source. `CurrentInput` drives in pA. A `GapJunction`'s weight is a conductance in nS.
- **Neuromodulators.** A `Modulator`'s concentration has no unit. Each spike of its source adds `release`, and the concentration decays with `tau` ms.
- **Plasticity.** STDP's, triplet STDP's, dopamine STDP's and Tsodyks-Markram's time constants are in ms.
- **EventProp.** `sparx.learn.spike_times` works in continuous time in ms, with the membrane in mV above rest and currents in pA.

## Rates

A rate has one of two units, depending on which half reports it.

| Rate | Unit |
| --- | --- |
| `sparx.firing_rates`, `sparx.rate_penalty`, `RateBand`, the `rate/<layer>` training metric, `ActivityFitObjective`'s rates, `sparx.learn.run_converted` | spikes per step |
| `PopulationRate`, `PoissonInput.rate`, `sparx.spiketrains.rates_hz`, a connectome's stimulus rates | Hz |

A rate per step `p` at steps of `dt` ms is `1000 p / dt` Hz. `RateBand(lower=0.01, upper=0.3)` asks each neuron to fire on between 1% and 30% of steps.

## Where the halves meet

- **A physical model as a layer.** `nn.Dynamics(model, dt=...)` takes `dt` in ms and its input as a current in pA (`drive="current"`) or as a jump in mV (`drive="jump"`). Train it with a steep surrogate; the [guide](guide.md#surrogate-gradients) explains why.
- **A dimensionless cell in a network.** A `Network` steps every population at its `dt` in ms, so a dimensionless cell there decays by `decay ** dt` per step and its decay is per ms. Build it with `decay(tau_in_ms)`, or give the network `dt=1.0`, as the tests do. Delays and `PopulationRate` stay in ms and Hz.
- **Spike times.** A spike in step `n` with offset `o` in the step happened at `(n + o) * dt`. A dimensionless model's offset is 1, the end of the step.

## Seconds

Two interfaces take seconds, because their formats do:

- `sparx.datasets.shd(steps=100, max_time=1.4)` bins the first `max_time` seconds of each recording into `steps` bins, 14 ms each by default, the binning of Zenke's SpyTorch tutorial.
- NIR stores time constants in seconds, and `sparx.nir.to_nir` and `from_nir` take the step `dt` in seconds.
