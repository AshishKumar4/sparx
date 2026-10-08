# From NEST and Brian2

`sparx.graph` simulates networks of point neurons in NEST's order of operations, and it matches NEST 3.10 and Brian2 2.10 spike for spike where the dynamics are deterministic ([fidelity.md](../fidelity.md)). This page maps their names onto sparx's, then builds a network in each style and runs Potjans and Diesmann's cortical microcircuit.

## A network, line by line

A balanced network of current-based neurons in PyNEST:

```python
nest.resolution = 0.1
params = {"tau_m": 10.0, "C_m": 250.0, "E_L": -65.0, "V_th": -50.0, "V_reset": -65.0, "t_ref": 2.0,
          "tau_syn_ex": 0.5, "tau_syn_in": 0.5}
e = nest.Create("iaf_psc_exp", 4000, params=params)
i = nest.Create("iaf_psc_exp", 1000, params=params)
(e + i).V_m = nest.random.normal(-58.0, 10.0)
for pre, k, w, d in ((e, 400, 87.8, 1.5), (i, 100, -439.0, 0.8)):
    nest.Connect(pre, e + i, {"rule": "fixed_indegree", "indegree": k}, {"weight": w, "delay": d})
noise = nest.Create("poisson_generator", params={"rate": 8.0 * 1000})
nest.Connect(noise, e + i, syn_spec={"weight": 87.8, "delay": 0.1})
recorder = nest.Create("spike_recorder")
nest.Connect(e, recorder)
nest.Simulate(500.0)
```

The same network in sparx. A population names its receptors, and each projection says which one it feeds, which sets the unit of its weight (pA for a current):

```python
import jax
from sparx.dynamics import Exponential, LeakyIntegrateAndFire, Receptor
from sparx.graph import FixedInDegree, Network, PoissonInput, Population, Projection, SpikeRaster, simulate
from sparx.spiketrains import cv_isi, rates_hz

neuron = LeakyIntegrateAndFire(tau_m=10.0, c_m=250.0, e_l=-65.0, v_th=-50.0, v_reset=-65.0, t_ref=2.0)
receptors = {"ampa": Receptor(Exponential(0.5)), "gaba_a": Receptor(Exponential(0.5))}
start = {"v": lambda rng, n: rng.normal(-58.0, 10.0, n)}
network = Network(
    populations=(Population("e", 4000, neuron, receptors, initial=start),
                 Population("i", 1000, neuron, receptors, initial=start)),
    projections=tuple(
        Projection(pre, post, FixedInDegree(400 if pre == "e" else 100, autapses=True, multapses=True),
                   weight=87.8 if pre == "e" else -439.0, delay=1.5 if pre == "e" else 0.8,
                   receptor="ampa" if pre == "e" else "gaba_a")
        for pre in ("e", "i") for post in ("e", "i")),
    inputs=tuple(PoissonInput(name, rate=8.0, weight=87.8, receptor="ampa", count=1000) for name in ("e", "i")),
    dt=0.1,
)
result = simulate(network, network.init(jax.random.key(0)), duration=500.0, key=jax.random.key(1),
                  monitors={"e": SpikeRaster("e")})
spikes = result.records["e"][2000:]        # [steps, 4000] after the first 200 ms
print(rates_hz(spikes, 0.1).mean(), cv_isi(spikes).mean())   # about 13 Hz, CV 0.4
```

`network.init(key)` draws the connections and initial voltages, as `nest.Connect` and `nest.random` do, and `simulate`'s `key` drives the Poisson input. The network is a Flax module: its connections, weights and state are arrays in its variables, so the whole run compiles to one program and runs on any JAX device.

In Brian2 the neuron is an equation, and a synapse is a statement on its variables:

```python
defaultclock.dt = 0.1*ms
eqs = """dv/dt = -(v + 65*mV) / (10*ms) + (I_ex + I_in) / (250*pF) : volt (unless refractory)
         dI_ex/dt = -I_ex / (0.5*ms) : amp
         dI_in/dt = -I_in / (0.5*ms) : amp"""
G = NeuronGroup(5000, eqs, threshold="v > -50*mV", reset="v = -65*mV", refractory=2*ms, method="exact")
S = Synapses(G[:4000], G, on_pre="I_ex += 87.8*pA")
```

sparx's receptor is that synaptic variable: `Exponential(0.5)` is `dI/dt = -I / (0.5 ms)`, incremented by the weight on each spike, and `Receptor(Exponential(5.0), "conductance")` is a conductance against the receptor's reversal potential, `g (E - v)`.

## Names

| NEST | Brian2 | sparx |
| --- | --- | --- |
| `nest.resolution = 0.1` | `defaultclock.dt = 0.1*ms` | `Network(..., dt=0.1)` |
| `nest.Create("iaf_psc_exp", n, params)` | `NeuronGroup(n, eqs, threshold=..., reset=..., refractory=...)` | `Population("e", n, LeakyIntegrateAndFire(...), receptors)` |
| `tau_syn_ex`, `tau_syn_in` | `dI/dt = -I / tau` | `{"ampa": Receptor(Exponential(tau)), "gaba_a": ...}` |
| `iaf_cond_exp` | `g*(E - v)` in the equations | `Receptor(Exponential(tau), "conductance")` |
| `iaf_psc_delta` | `on_pre="v += w"` | `Receptor(Delta())`, its weight in mV |
| `aeif_cond_exp`, `izhikevich`, `hh_psc_alpha` | their equations | `AdEx`, `Izhikevich`, `HodgkinHuxley` |
| `I_e` | a constant current in the equations | `LeakyIntegrateAndFire(i_e=...)` |
| `V_m = nest.random.normal(m, s)` | `G.v = "m + s*randn()"` | `Population(..., initial={"v": lambda rng, n: rng.normal(m, s, n)})` |
| `{"rule": "pairwise_bernoulli", "p": p}` | `S.connect(p=p)` | `FixedProbability(p)` |
| `"fixed_indegree"`, `"fixed_outdegree"`, `"fixed_total_number"` | `S.connect(i=..., j=...)` | `FixedInDegree(k)`, `FixedOutDegree(k)`, `FixedTotalNumber(n)` |
| `"all_to_all"`, `"one_to_one"`, explicit lists | `S.connect()`, `S.connect(j="i")`, `S.connect(i=..., j=...)` | `AllToAll()`, `OneToOne()`, `FromEdges(pre, post)` |
| `allow_autapses`, `allow_multapses`, both True by default | `condition="i != j"` | `autapses=`, `multapses=`, both False by default |
| `syn_spec={"weight": w, "delay": d}` | `S.w = ...`, `S.delay = ...` | `weight=`, `delay=`, each one value, an array per edge or `f(rng, n)` |
| `poisson_generator` | `PoissonInput(G, "v", N, rate, weight)` | `PoissonInput("e", rate, weight, receptor=..., count=N)` |
| `dc_generator`, `step_current_generator` | `TimedArray` | `CurrentInput("e", "name")`, and `simulate(..., drive={"name": values})` |
| `stdp_synapse`, `stdp_triplet_synapse`, `stdp_dopamine_synapse` | `Synapses` with traces | `Projection(..., plasticity=PairSTDP(...))`, `TripletSTDP`, `DopamineSTDP` |
| `tsodyks2_synapse` | `Synapses` with `u` and `x` | `Projection(..., short_term=TsodyksMarkram(...))` |
| `gap_junction` | `Synapses` with a `(summed)` current | `GapJunction("a", "b", connectivity, weight=g)` |
| `spike_recorder` | `SpikeMonitor` | `SpikeRaster("e")`, `SpikeTimes("e")`, `SpikeCounts("e")` |
| `multimeter` | `StateMonitor(G, "v", record=[...])` | `StateMonitor("e", neurons=(...))` |
| | `PopulationRateMonitor` | `PopulationRate("e")` |
| `nest.rng_seed` | `seed(...)` | `network.init(jax.random.key(seed))`, `simulate(..., key=...)` |
| `nest.Simulate(t)` | `run(t*ms)` | `simulate(network, variables, duration=t, monitors=...)` |
| `nest.GetConnections()` | `S.i`, `S.j`, `S.w` | `network.connections(result.variables)` |

Quantities are plain numbers in ms, mV, pA, nS, pF and Hz ([units.md](../units.md)). Three differences matter when porting a model:

- NEST's connection rules allow self-connections and repeated pairs unless told otherwise; sparx's allow neither unless asked, so a NEST model ported faithfully passes `autapses=True, multapses=True`, as above.
- A delay must be a whole number of steps: sparx raises where NEST would round. Round it yourself to keep NEST's behaviour.
- A neuron holds its reset for `round(t_ref / dt)` steps after the step it fired in, as NEST counts; Brian2 counts one step fewer.

## The cortical microcircuit

Potjans and Diesmann's (2014) model of 1 mm² of cortex is the standard test of a spiking simulator: eight populations, an excitatory and an inhibitory one in each of layers 2/3, 4, 5 and 6, 77,169 neurons and about 300 million synapses at full size. `sparx.graph.models.microcircuit` builds it as its reference implementation, INM-6's PyNEST code, builds it, with the same scaling. At a fifth of the neurons and of their inputs:

```python
from sparx.graph import PopulationRate
from sparx.graph.models import MICROCIRCUIT_POPULATIONS, microcircuit

network = microcircuit(0.2, 0.2)    # 15,435 neurons, 12 million synapses
monitors = {name: PopulationRate(name) for name in MICROCIRCUIT_POPULATIONS}
result = simulate(network, network.init(jax.random.key(0)), duration=1500.0, monitors=monitors)
rates = {name: float(result.records[name][5000:].mean()) for name in MICROCIRCUIT_POPULATIONS}  # Hz
```

The first 500 ms are the network settling from its initial voltages. Over the next second the populations fire at, in Hz:

| | L2/3E | L2/3I | L4E | L4I | L5E | L5I | L6E | L6I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparx, key 0 | 0.55 | 2.21 | 3.82 | 4.88 | 7.46 | 7.63 | 0.84 | 6.77 |
| NEST, 15 seeds | 0.55 to 0.64 | 2.16 to 2.28 | 3.70 to 3.81 | 4.84 to 4.91 | 6.62 to 7.29 | 7.53 to 7.68 | 0.79 to 0.85 | 6.68 to 6.78 |

The test of this port runs NEST twice. On the very network sparx draws, exported edge by edge, NEST and sparx in float64 fire the same 12,689 spikes over 300 ms. Over NEST's own draws, every population's distribution of rates, interspike irregularity and pairwise correlations lies as close to NEST's as NEST's seeds lie to each other, the reference's own criterion. In float32, the default, the run follows NEST's for about 140 ms before rounding steers the chaotic network onto another trajectory with the same statistics. At a tenth of the neurons and inputs, the model falls silent in both simulators.

On a 4-core CPU, sparx runs this network at 8.8 s per simulated second and NEST at 2.9 s ([performance.md](../performance.md#against-nest-and-brian2)). What sparx adds is what the rest of the library gives a network: the same model with trainable weights (`Projection(trainable=True)`), gradients through it, batches of trials under `jax.vmap`, and a run sharded over devices.
