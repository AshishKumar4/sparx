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


def alif(xs, decay, adapt_decay, beta, threshold=1.0, reset="subtract"):
    v = np.zeros(xs.shape[1:])
    a = np.zeros(xs.shape[1:])
    spikes = []
    for x in xs:
        theta = threshold + beta * a
        v = decay * v + x
        s = (v >= theta).astype(np.float64)
        v = _reset(v, s, theta, reset)
        a = adapt_decay * a + s
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
