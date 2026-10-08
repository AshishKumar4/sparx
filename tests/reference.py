"""Neuron dynamics written as plain NumPy loops, independently of sparx.

Each function follows the equations in the cell's docstring, step by step in
float64, so a test compares sparx's scan against the arithmetic it claims.
"""

import numpy as np


def _reset(v, s, threshold, reset):
    if reset == "subtract":
        return v - s * threshold
    if reset == "zero":
        return np.where(s > 0, 0.0, v)
    return v


def lif(xs, decay, threshold=1.0, reset="subtract"):
    v = np.zeros(xs.shape[1:])
    spikes, membranes = [], []
    for x in xs:
        v = decay * v + x
        s = (v >= threshold).astype(np.float64)
        v = _reset(v, s, threshold, reset)
        spikes.append(s)
        membranes.append(v)
    return np.stack(spikes), np.stack(membranes)


def bernoulli(xs, noise, decay, beta, threshold=1.0, reset="subtract"):
    """BernoulliCell: the LIF membrane, firing where the noise falls below sigmoid(beta (v - threshold))."""
    v = np.zeros(xs.shape[1:])
    spikes, probabilities = [], []
    for x, u in zip(xs, noise, strict=True):
        v = decay * v + x
        p = 1 / (1 + np.exp(-beta * (v - threshold)))
        s = (u < p).astype(np.float64)
        v = _reset(v, s, threshold, reset)
        spikes.append(s)
        probabilities.append(p)
    return np.stack(spikes), np.stack(probabilities), v


def li(xs, decay):
    v = np.zeros(xs.shape[1:])
    out = []
    for x in xs:
        v = decay * v + x
        out.append(v)
    return np.stack(out)


def synaptic(xs, decay, synapse_decay, threshold=1.0, reset="subtract"):
    i = np.zeros(xs.shape[1:])
    v = np.zeros(xs.shape[1:])
    spikes = []
    for x in xs:
        i = synapse_decay * i + x
        v = decay * v + i
        s = (v >= threshold).astype(np.float64)
        v = _reset(v, s, threshold, reset)
        spikes.append(s)
    return np.stack(spikes)


def alif(xs, decay, adapt_decay, beta, threshold=1.0, reset="subtract", n_refractory=0):
    """With `n_refractory`, Bellec et al.'s refractory counter (`models.py`, `EligALIF.__call__`)."""
    v = np.zeros(xs.shape[1:])
    a = np.zeros(xs.shape[1:])
    r = np.zeros(xs.shape[1:])
    spikes = []
    for x in xs:
        theta = threshold + beta * a
        v = decay * v + x
        s = np.where(r > 0, 0.0, (v >= theta).astype(np.float64))
        v = np.where(s > 0, 0.0, v) if reset == "zero" else v - s * threshold
        a = adapt_decay * a + s
        r = np.clip(r + n_refractory * s - 1, 0, n_refractory)
        spikes.append(s)
    return np.stack(spikes)


def izhikevich(xs, a=0.02, b=0.2, c=-65.0, d=8.0):
    """Izhikevich (2003)'s published MATLAB loop at 1 ms a step, transcribed.

    His loop finds the neurons that fired (v >= 30) at the start of each step
    and resets them before integrating; a spike is returned here at the step
    whose integration crossed, one step before his loop records it.
    """
    v = np.full(xs.shape[1:], c)
    u = b * v
    crossed = []
    for x in xs:
        fired = v >= 30
        v = np.where(fired, c, v)
        u = np.where(fired, u + d, u)
        v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + x)
        v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + x)
        u = u + a * (b * v - u)
        crossed.append((v >= 30).astype(np.float64))
    return np.stack(crossed)


ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0),
               "sigmoid": lambda x: 1 / (1 + np.exp(-x))}


def flynn(xs, weight, alpha, bias, activation="tanh"):
    """FLYNN's recurrence (Wang and Chen, arXiv 2607.00025, eq. 1) as the paper writes it,

        h_{t+1} = alpha h_t + (1 - alpha) f(W h_t + x_t + b),    h_0 = 0,

    with `W h_t` taken as `h_t @ weight`; returns `h_1 ... h_T`.
    """
    h = np.zeros(xs.shape[1:])
    out = []
    for x in xs:
        h = alpha * h + (1 - alpha) * ACTIVATIONS[activation](h @ weight + x + bias)
        out.append(h)
    return np.stack(out)


def recurrent_lif(xs, weight, decay, threshold=1.0, reset="subtract"):
    v = np.zeros(xs.shape[1:])
    s = np.zeros(xs.shape[1:])
    spikes = []
    for x in xs:
        v = decay * v + x + s @ weight
        s = (v >= threshold).astype(np.float64)
        v = _reset(v, s, threshold, reset)
        spikes.append(s)
    return np.stack(spikes)


def eprop_alif(xs, alpha, rho, beta, v_th, reset):
    """The ALIF of Bellec et al. (2020), as their code steps it
    (IGITUGraz/eligibility_propagation, Figure_4_and_5_ATARI/alif_eligibility_propagation.py,
    without refractoriness): the previous step's spike raises the adaptation
    and subtracts `reset` from the membrane, after the decay."""
    v = np.zeros(xs.shape[1:])
    a = np.zeros(xs.shape[1:])
    z = np.zeros(xs.shape[1:])
    spikes = []
    for x in xs:
        a = rho * a + z
        v = alpha * v + x - z * reset
        z = (v >= v_th + beta * a).astype(np.float64)
        spikes.append(z)
    return np.stack(spikes)


def conductance_lif(arrivals, current, *, c_m, g_l, e_l, v_th, v_reset, t_ref, reversal, kernels, dt,
                    substeps=200):
    """A conductance-based LIF integrated by RK4 at `dt / substeps`, the ground truth.

    `arrivals[name]` `[T, N]` are weights arriving at the end of each step on
    receptor `name`, `kernels[name]` its conductance kernel as a function of
    the time since arrival (summed over arrivals, so each kernel must be
    linear: exponential, alpha or bi-exponential). Spikes are detected and
    the voltage reset at step boundaries, refractoriness counts whole steps,
    as NEST and sparx do; only the membrane between them is integrated
    finely. Returns the voltage after each step and the spikes.
    """
    steps, n = current.shape
    names = list(arrivals)
    times = {name: [[] for _ in range(n)] for name in names}
    v = np.full(n, float(e_l))
    held = np.zeros(n, dtype=int)
    hold_steps = round(t_ref / dt)
    vs, spikes = [], []
    h = dt / substeps

    def conductance(name, i, t):
        return sum(w * kernels[name](t - s) for s, w in times[name][i] if t >= s)

    def dv(i, t, v, current):
        g = {name: conductance(name, i, t) for name in names}
        return (-g_l * (v - e_l) + sum(g[k] * (reversal[k] - v) for k in names) + current) / c_m

    for step in range(steps):
        t0 = step * dt
        fired = np.zeros(n)
        for i in range(n):
            if held[i] > 0:
                held[i] -= 1
                v[i] = v_reset
            else:
                x = v[i]
                for k in range(substeps):
                    t = t0 + k * h
                    k1 = dv(i, t, x, current[step, i])
                    k2 = dv(i, t + h / 2, x + h / 2 * k1, current[step, i])
                    k3 = dv(i, t + h / 2, x + h / 2 * k2, current[step, i])
                    k4 = dv(i, t + h, x + h * k3, current[step, i])
                    x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                v[i] = x
                if x >= v_th:
                    fired[i] = 1
                    v[i] = v_reset
                    held[i] = hold_steps
            for name in names:
                if arrivals[name][step, i] != 0:
                    times[name][i].append(((step + 1) * dt, arrivals[name][step, i]))
        vs.append(v.copy())
        spikes.append(fired)
    return np.stack(vs), np.stack(spikes)


def pulse(s, threshold):
    """RNeuralNet's activation (`Soma_t::ActivationFunction`, A_CONST 1): `s` at or above the threshold,
    `exp(s - threshold) - 1` below."""
    return np.where(s >= threshold, s, np.exp(np.minimum(s - threshold, 0.0)) - 1)


def delayed_pulses(xs, pre, post, weight, delay, threshold):
    """Units that each step output `pulse` of what arrived and send `weight[e]` times their output down
    each connection, to arrive `delay[e]` steps later; returns each step's outputs."""
    arrivals = np.zeros((len(xs) + int(np.max(delay)) + 1, *xs.shape[1:]))
    out = []
    for t, x in enumerate(xs):
        o = pulse(arrivals[t] + x, threshold)
        for e in range(len(pre)):
            arrivals[t + delay[e], ..., post[e]] += weight[e] * o[..., pre[e]]
        out.append(o)
    return np.stack(out)


def reward_spread(pre, post, activity, reward, root, size, eta=0.01):
    """`Global_RewardSpreader` and `Global_Teacher` of RNeuralNet-Research (d4b7803) as written.

    Recursive and depth first from `root`: a unit whose `Var2` has not
    counted all its incoming connections, entered with a nonzero reward,
    gives each source the share `exp|a| / sum exp|a|` of that reward,
    counting each, then enters each source with what the source holds.
    Then every connection whose target holds a reward changes by `eta`
    times that reward times its share. Returns the changes, the local
    rewards and the shares.
    """
    dendrites = [[e for e in range(len(pre)) if post[e] == u] for u in range(size)]
    credit, share, var2 = np.zeros(size), np.zeros(len(pre)), np.zeros(size, int)
    credit[root] = reward

    def spread(unit, r):
        if var2[unit] >= len(dendrites[unit]) or r == 0:
            return
        total = sum(np.exp(abs(activity[pre[e]])) for e in dendrites[unit])
        for e in dendrites[unit]:
            share[e] = np.exp(abs(activity[pre[e]])) / total
            credit[pre[e]] += r * share[e]
            var2[unit] += 1
        for e in dendrites[unit]:
            spread(pre[e], credit[pre[e]])

    spread(root, reward)
    change = np.zeros(len(pre))
    for e in range(len(pre)):
        if eta * credit[post[e]] != 0:
            change[e] = eta * credit[post[e]] * share[e]
    return change, credit, share
