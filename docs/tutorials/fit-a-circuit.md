# Fit a circuit to recordings

A network built with `sparx.graph` is differentiable, so a circuit's unknown parameters can be fit to recordings by gradient descent through its simulation. This page recovers a circuit's synaptic weights from its neurons' membrane potentials, then fits the same circuit to spike times, and ends with what the gradient is in each case.

## Recordings

Forty neurons, driven by injected currents, connect to ten more through 400 synapses whose weights the fit will look for. The ten are recorded as a patch clamp or a voltage indicator records them, their membrane potential at every step:

```python
import jax
import numpy as np
from sparx.dynamics import Exponential, LeakyIntegrateAndFire, Receptor
from sparx.graph import AllToAll, CurrentInput, Network, Population, Projection, StateMonitor

neuron = LeakyIntegrateAndFire(tau_m=10.0, c_m=250.0, e_l=-65.0, v_th=-50.0, v_reset=-65.0, t_ref=2.0,
                               detach_reset=True)
receptors = {"ampa": Receptor(Exponential(2.0))}


def circuit(weight):
    """40 neurons driven by injected currents, connected to 10 more through synapses of `weight` pA."""
    return Network(
        populations=(Population("in", 40, neuron, receptors), Population("out", 10, neuron, receptors)),
        projections=(Projection("in", "out", AllToAll(), weight=weight, delay=1.0, receptor="ampa",
                                trainable=True),),
        inputs=(CurrentInput("in", "stimulus"),),
        dt=0.1,
    )


rng = np.random.default_rng(0)
weights = rng.normal(0.0, 100.0, 400)                            # pA, one per synapse
drive = {"stimulus": rng.normal(420.0, 300.0, (3000, 40))}       # pA, 300 ms
teacher = circuit(weights)
truth = teacher.init(jax.random.key(0))
recorded, _ = teacher.apply(truth, drive, monitors={"v": StateMonitor("out")}, mutable=["state"])
```

The input neurons fire at about 40 Hz. The recorded membranes stay between -73 and -58 mV, below the threshold of -50 mV.

## Fit the voltages

The student is the same circuit with every weight at zero. A projection built with `trainable=True` keeps its weights in `params`, and the loss is the mean squared difference between the student's membranes and the recorded ones over the whole 300 ms. optax's L-BFGS minimizes it:

```python
import jax.numpy as jnp
import optax

student = circuit(np.zeros(400))
variables = student.init(jax.random.key(0))
params = variables.pop("params")                                 # {"weight:in->out:ampa": 400 zeros}


def loss(params):
    traced, _ = student.apply({**variables, "params": params}, drive, monitors={"v": StateMonitor("out")},
                              mutable=["state"])
    return jnp.mean((traced["v"] - recorded["v"]) ** 2)          # mV²


optimizer = optax.lbfgs()
value_and_grad = optax.value_and_grad_from_state(loss)


@jax.jit
def update(params, state):
    value, grads = value_and_grad(params, state=state)
    updates, state = optimizer.update(grads, state, params, value=value, grad=grads, value_fn=loss)
    return optax.apply_updates(params, updates), state


state = optimizer.init(params)
for _ in range(100):
    params, state = update(params, state)
fitted = student.connections({**variables, "params": params})["in->out:ampa"].weight
actual = teacher.connections(truth)["in->out:ampa"].weight
correlation, error = np.corrcoef(fitted, actual)[0, 1], np.abs(fitted - actual).max()
```

After 100 iterations, 10 s on a 4-core CPU, the fitted weights correlate with the true ones at 0.9999 and differ from them by 8 pA at most, for weights from -377 to 307 pA.

This gradient is exact. The recorded neurons never fire, so their membranes are a smooth function of the weights, and `jax.grad` differentiates that function. `tests/test_graph.py` compares such a gradient with central differences, on a network whose driven neurons fire, and the two agree within 3e-9 relative.

## Fit the spikes

With stronger synapses the same circuit's outputs fire, at about 21 Hz, and the voltage loss no longer serves. A spike one step earlier or later moves the membrane by 15 mV for milliseconds, so the loss jumps as the weights change, and fit from zero weights to those voltages, L-BFGS reaches a correlation of 0.06 after 100 iterations.

Spikes are fit through a surrogate gradient instead (Neftci, Mostafa and Zenke 2019). The backward pass replaces the derivative of the spike, a step function of the membrane, with a smooth function of the membrane's distance to threshold, and the loss compares spike trains, here by van Rossum's distance, as SuperSpike does (Zenke and Ganguli 2018):

```python
from sparx.graph import OutputTrace
from sparx.losses import van_rossum
from sparx.spiketrains import coincidence_factor

firing = circuit(weights + 100.0)
recorded_spikes, _ = firing.apply(firing.init(jax.random.key(0)), drive, monitors={"s": OutputTrace("out")},
                                  mutable=["state"])
target = recorded_spikes["s"]                                    # [3000, 10], 0 or 1


def spike_loss(params):
    traced, _ = student.apply({**variables, "params": params}, drive, monitors={"s": OutputTrace("out")},
                              mutable=["state"])
    return van_rossum(traced["s"], target, tau=5.0, dt=0.1), traced["s"]


adam = optax.adam(2.0)


@jax.jit
def step(params, state):
    grads, _ = jax.grad(spike_loss, has_aux=True)(params)
    updates, state = adam.update(grads, state)
    return optax.apply_updates(params, updates), state


params = {name: jnp.zeros_like(value) for name, value in params.items()}
state = adam.init(params)
for _ in range(400):
    params, state = step(params, state)
_, fired = spike_loss(params)
gamma = np.nanmean(coincidence_factor(fired, target, dt=0.1, window=2.0))
```

After 400 steps, 11 s, the student fires 58 spikes to the recording's 63, and its mean coincidence factor is 0.84 at 2 ms precision. The coincidence factor (Kistler et al. 1997) is the score of spike-time prediction in Jolivet et al.'s (2008) benchmark: 1 for the same spikes, 0 on average for a Poisson train of the student's rate. The student's weights correlate with the true ones at 0.25, since 63 spike times constrain 400 weights far less than 30,000 voltage samples do.

## What the gradient is

- **Below threshold**, exact, as above.
- **Where a neuron fires**, the surrogate's. The default for `LeakyIntegrateAndFire` is `ATan()`; [the guide](../guide.md#surrogate-gradients) lists the others and when a steeper one trains better.
- **Through the reset**, stopped by `detach_reset=True`. A reset jumps the membrane by `v_th - v_reset`, so the reset's surrogate path multiplies a membrane's gradient by about `1 - slope (v_th - v_reset)` each step the membrane spends near threshold: -14 with `ATan()`'s slope of 1 per mV at threshold and this page's 15 mV reset, which overflows float32 within a few dozen steps. Zenke's SpyTorch tutorials stop the same path, and SpikingJelly names the option the same way.
- **Through synapses that deliver by events**, the default for fixed weights between spiking populations, the gradient is that of the same synapses as an edge list. A membrane near threshold has a surrogate gradient without a spike, so the backward pass visits every edge at every step, as `format="edges"` does, while the forward pass still visits only the edges of neurons that spiked. `tests/test_graph.py` checks the two against each other to 8e-15.
- **Through recurrence**, a product of one factor per step, which can grow with the length of the run. In a network of 400 excitatory and 100 inhibitory neurons firing at about 80 Hz, the gradient of the last step's mean membrane with respect to the input weights had norm 0.3 over 20 ms, 1e5 over 40 ms and 1e17 over 80 ms with `ATan()`, and 0.003, 0.01 and 0.3 with `FastSigmoid(100.0)`. Fit a recurrent network with a steep surrogate, as the guide's recurrent SHD run does, and over windows short enough that the gradient stays finite.
