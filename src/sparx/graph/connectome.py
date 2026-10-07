"""Connectomes as networks: reading synapse tables, and the models published on them.

    brain = Connectome.from_shiu("Completeness_783.csv", "Connectivity_783.parquet")
    sugar = brain.index(SUGAR_IDS)                      # FlyWire root IDs -> neuron indices
    network = shiu2024(brain, stimuli=[(sugar, 150.0)])
    result = simulate(network, network.init(key), duration=1000.0, key=key,
                      monitors={"counts": SpikeCounts("brain")})

A `Connectome` is a neuron table and an edge list with signed synapse
counts: the sign is the presynaptic neuron's predicted transmitter's
(acetylcholine excitatory; GABA and glutamate inhibitory, as Shiu et al.
assign them for the fly), the count is how many synapses join the pair.
Reading the tables needs the `connectome` extra (pyarrow). `shiu2024` has
the short name `shiu2024` in `sparx.registry.networks`, and its record names
the reader of its tables by import path (`Connectome.from_shiu` for Shiu et
al.'s tables, `Connectome.from_malecns` for Janelia's), so
`sparx.graph.from_record` rebuilds a model on a connectome from its record.

`FLYNN` is Wang and Chen's trainable fly connectome network (arXiv
2607.00025): a leaky tanh unit per neuron, recurrent through the synapses
(`sparx.dynamics.SparseRecurrentCell`), with every weight, bias and cell
class's leak trained by gradient descent, a Flax layer for dew's trainer.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sparx.dynamics import RateCell, SparseRecurrentCell, SynapticInput
from sparx.dynamics.neurons import LIF
from sparx.dynamics.synapses import Delta, Exponential, Receptor
from sparx.graph.connectivity import FromEdges
from sparx.graph.network import Network, PoissonInput, Population, Projection
from sparx.nn.neurons import Neuron

__all__ = ["FLYNN", "FLYWIRE_630_MEDIAN_INPUTS", "SIGNS", "Connectome", "matched_w_syn", "shiu2024",
           "spectral_radius"]

SIGNS: dict[str, int] = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1, "dopamine": 1,
                         "serotonin": 1, "octopamine": 1, "unclear": 1}
"""The sign of each transmitter's synapses. Acetylcholine excites and GABA and glutamate inhibit, as Shiu
et al. (2024) assign them (glutamate's sign depends on the receptor in the fly, and is a choice).
Histamine inhibits (it opens chloride channels in the fly). Shiu et al. treat the modulators as
excitatory, and so does this table; a neuron whose transmitter is unclear excites."""


@dataclass(frozen=True)
class Connectome:
    """Neurons `ids` (e.g. FlyWire root IDs) and edges `pre -> post` (indices) with signed synapse counts."""

    ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    synapses: np.ndarray

    def __post_init__(self):
        if not len(self.pre) == len(self.post) == len(self.synapses):
            raise ValueError("pre, post and synapses must have one entry per edge")

    @property
    def size(self) -> int:
        return len(self.ids)

    def inputs(self) -> np.ndarray:
        """Each neuron's input synapses, excitatory and inhibitory alike."""
        return np.bincount(self.post, weights=np.abs(self.synapses), minlength=self.size)

    def index(self, ids: Iterable[int]) -> np.ndarray:
        """The neuron indices of `ids`; raises for an ID not in the connectome."""
        ids = np.asarray(list(ids), np.int64)
        order = np.argsort(self.ids)
        found = np.searchsorted(self.ids, ids, sorter=order)
        found = order[np.minimum(found, len(order) - 1)]
        missing = ids[self.ids[found] != ids]
        if len(missing):
            shown = missing.tolist()[:5]
            raise KeyError(f"not in the connectome: {shown}{'...' if len(missing) > 5 else ''}")
        return found.astype(np.int32)

    def without(self, silenced: Iterable[int]) -> Connectome:
        """The connectome with every synapse from the neurons `silenced` (indices) removed."""
        drop = np.isin(self.pre, np.asarray(list(silenced), np.int64))
        return Connectome(self.ids, self.pre[~drop], self.post[~drop], self.synapses[~drop])

    @classmethod
    def from_shiu(cls, completeness: str | Path, connectivity: str | Path) -> Connectome:
        """Read Shiu et al.'s (2024) tables (github.com/philshiu/Drosophila_brain_model).

        `completeness` is the CSV of neurons, indexed by FlyWire root ID, whose
        row order defines the neuron indices; `connectivity` the parquet file
        of edges with `Presynaptic_Index`, `Postsynaptic_Index` and
        `Excitatory x Connectivity` (the signed synapse count). Both FlyWire
        v630, which their paper used, and the public v783 tables read alike.
        """
        try:
            import pyarrow.csv as csv
            import pyarrow.parquet as parquet
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("reading connectome tables needs pyarrow: "
                              "pip install 'sparx[connectome]'") from error
        neurons = csv.read_csv(completeness)
        ids = np.asarray(neurons.column(0).to_numpy(), np.int64)
        columns = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
        table = parquet.read_table(connectivity, columns=columns)
        pre, post, synapses = (np.asarray(table.column(name).to_numpy()) for name in columns)
        return cls(ids, pre.astype(np.int32), post.astype(np.int32), synapses.astype(np.int32))

    @classmethod
    def from_malecns(cls, annotations: str | Path, neurotransmitters: str | Path, weights: str | Path, *,
                     statuses: Sequence[str] = ("Traced", "Anchor"),
                     signs: dict[str, int] | None = None) -> Connectome:
        """Read Janelia's male CNS (brain and nerve cord) release tables, `gs://flyem-male-cns/v0.9`.

        `annotations` (`body-annotations-*.feather`) lists the bodies; those
        whose `status` is in `statuses` are the neurons (165,114 `Traced`
        and 785 `Anchor` in v0.9, against 1.8M segments in all).
        `neurotransmitters` (`body-neurotransmitters-*.feather`) gives each
        body's predicted transmitter, `consensus_nt` or, where that is
        `unclear`, `predicted_nt`; `signs` maps it to the sign of the
        neuron's synapses. `weights` (`connectome-weights-*.feather`) holds
        the synapse count of every connected pair of segments, filtered here
        to pairs of neurons. Neurons are ordered by body ID.
        """
        try:
            import pyarrow.feather as feather
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("reading connectome tables needs pyarrow: "
                              "pip install 'sparx[connectome]'") from error
        signs = dict(SIGNS if signs is None else signs)
        bodies = feather.read_table(annotations, columns=["bodyId", "status"])
        status = np.asarray(bodies.column("status").to_pylist(), object)
        body = np.asarray(bodies.column("bodyId").to_numpy(), np.int64)
        ids = np.sort(body[np.isin(status, list(statuses))])

        def locate(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Indices of `body` among the neurons, and which of them are neurons."""
            index = np.minimum(np.searchsorted(ids, body), len(ids) - 1)
            return index.astype(np.int32), ids[index] == body

        table = feather.read_table(weights, columns=["body_pre", "body_post", "weight"])
        pre, pre_kept = locate(np.asarray(table.column("body_pre").to_numpy(), np.int64))
        post, post_kept = locate(np.asarray(table.column("body_post").to_numpy(), np.int64))
        kept = pre_kept & post_kept
        pre, post = pre[kept], post[kept]
        counts = np.asarray(table.column("weight").to_numpy(), np.int32)[kept]
        del table
        transmitters = feather.read_table(neurotransmitters, columns=["body", "consensus_nt", "predicted_nt"])
        index, known = locate(np.asarray(transmitters.column("body").to_numpy(), np.int64))
        rows = zip(transmitters.column("consensus_nt").to_pylist(),
                   transmitters.column("predicted_nt").to_pylist(), known, strict=True)
        chosen = [predicted if consensus in (None, "unclear") else consensus
                  for consensus, predicted, kept in rows if kept]
        fallback = signs.get("unclear", 1)
        sign = np.full(len(ids), fallback, np.int32)
        sign[index[known]] = [signs.get(t, fallback) for t in chosen]
        return cls(ids, pre, post, counts * sign[pre])


FLYWIRE_630_MEDIAN_INPUTS = 206.0
"""The median neuron's input synapses in FlyWire v630, the connectome Shiu et al. fit `w_syn` on."""


def matched_w_syn(connectome: Connectome, *, w_syn: float = 0.275,
                  reference: float = FLYWIRE_630_MEDIAN_INPUTS) -> float:
    """Shiu et al.'s `w_syn` scaled so `connectome`'s median neuron gets the input FlyWire's did.

    `w_syn` was fit to FlyWire v630's synapse counts, and a connectome that
    counts more synapses per neuron drives every neuron harder at the same
    weight. The male CNS v0.9 counts a median 347 input synapses per neuron
    against FlyWire's 206 (its mean, 748 against 414, grows alike), so it
    takes 0.163 mV: with 0.275 mV its sugar neurons recruit 18,000 neurons,
    with 0.163 mV about 670 (FlyWire: about 400), and MN9 fires as Shiu et
    al.'s does (`docs/fidelity.md`).
    """
    return w_syn * reference / float(np.median(connectome.inputs()))


def shiu2024(connectome: Connectome, *, stimuli: Sequence[tuple[Sequence[int], float]] = (),
             silenced: Sequence[int] = (), dt: float = 0.1, capacity: int = 4096,
             w_syn: float = 0.275, stimulus_scale: float = 250.0) -> Network:
    """Shiu et al.'s (Nature 2024) leaky integrate-and-fire model of the whole fly brain.

    Every neuron is one LIF: membrane 20 ms, rest and reset -52 mV,
    threshold -45 mV, 2.2 ms refractory, driven by
    `dv/dt = (v_0 - v + g) / tau_m` with `dg/dt = -g / tau`, `tau = 5 ms`.
    A spike adds `w_syn` (0.275 mV) times the signed synapse count to the
    target's `g` after 1.8 ms. `stimuli` are `(neuron indices, rate in Hz)`:
    each neuron gets Poisson spikes at the rate that move its voltage by
    `stimulus_scale * w_syn` (enough to fire it) and has no refractory
    period, their model of optogenetic activation. `silenced` neurons
    (indices) lose their outgoing synapses. On a connectome other than
    FlyWire v630, `matched_w_syn` scales `w_syn` to its synapse counts.

    The model is theirs as their Brian2 code runs it (`model.py`, checked
    spike for spike on a small graph in `tests/test_graph.py`), quirks
    included: a spike also clears `g`, `g` holds while refractory and drops
    input arriving then, and a stimulus lands after the threshold test.
    Brian2 counts the refractory period from the start of the spiking step,
    so its 2.2 ms is sparx's 2.1 (21 steps of 0.1 ms).
    """
    if silenced:
        connectome = connectome.without(silenced)
    t_ref = np.full(connectome.size, 2.2 - 0.1)
    for neurons, _ in stimuli:
        t_ref[np.asarray(neurons)] = 0.0
    # With C = tau_m, g_L = 1 nS and a current in pA is Brian2's g in mV.
    neuron = LIF(tau_m=20.0, c_m=20.0, e_l=-52.0, v_th=-45.0, v_reset=-52.0, t_ref=jnp.asarray(t_ref))
    receptors = {"syn": Receptor(Exponential(5.0)), "stimulus": Receptor(Delta(after_threshold=True))}
    brain = Population("brain", connectome.size, neuron, receptors, reset_synapses=True, freeze_synapses=True)
    synapses = Projection("brain", "brain", FromEdges(connectome.pre, connectome.post),
                          weight=connectome.synapses * w_syn, delay=1.8, receptor="syn", format="events",
                          capacity=capacity)
    inputs = tuple(PoissonInput("brain", rate=rate, weight=stimulus_scale * w_syn, receptor="stimulus",
                                neurons=tuple(int(i) for i in neurons)) for neurons, rate in stimuli)
    return Network((brain,), (synapses,), inputs, dt=dt)

def spectral_radius(connectome: Connectome) -> float:
    """The spectral radius of `connectome`'s signed synapse-count matrix: its eigenvalues' largest magnitude.

    Computed exactly for up to 2,000 neurons, and above that by ARPACK's
    Arnoldi iteration (`scipy.sparse.linalg.eigs`), which finds a complex
    dominant pair as well as a real one. FLYNN's code estimates it by power
    iteration from a random vector and the Rayleigh quotient
    (`spectral_radius_power_iter` in their `core/utils.py`), which settles
    only when the dominant eigenvalue is real and alone on the spectrum's
    rim. A signed connectome's is often a complex pair: on a random
    connectome of 40 neurons with 30 % inhibitory synapses, their estimate
    was 20.1 and the same iteration from other starts gives 1.8 to 46,
    where the radius is 48.1 (`tests/test_flynn.py`'s fixture).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigs

    size = connectome.size
    matrix = coo_matrix((np.asarray(connectome.synapses, np.float64), (connectome.post, connectome.pre)),
                        shape=(size, size))
    if size <= 2000:
        values = np.linalg.eigvals(matrix.toarray())
    else:
        values = eigs(matrix.tocsr(), k=1, which="LM", v0=np.ones(size), return_eigenvectors=False)
    return float(np.abs(values).max())


class FLYNN(Neuron):
    """Wang and Chen's fly connectome network (FLYNN, arXiv 2607.00025): a leaky tanh unit per neuron of
    `connectome`, recurrent through its synapses, with input on some neurons and output read from others.

        h[t] = (1 - a) h[t-1] + a tanh(W h[t-1] + x[t] + b)

    their code's form of their eq. 1, with `a` the update fraction of each
    neuron's cell class, `a = 0.99 sigmoid(l) + 0.01` for a learned logit
    `l` per class (`types[i]` is neuron `i`'s class, 0 for unknown, as
    their `load_cell_types` numbers them). `W` holds one learned weight per
    edge of the connectome, starting at its signed synapse count scaled so
    that the matrix's spectral radius is `radius` (their 0.9; computed
    exactly by `spectral_radius`, where theirs is a power iteration's
    estimate), and `b` one learned bias per neuron, starting at 0.
    The input `[T, B, len(input_neurons)]` adds to the neurons
    `input_neurons` (their sensory neurons; an index may repeat), and the
    output `[T, B, len(output_neurons)]` is the activity of the neurons
    `output_neurons` (their descending neurons). `update` is where each class's update fraction's
    logit starts, `log(update / (1 - update))`, their `leak_alpha` of 0.2.

    Their readout and input scales sit outside the network, and so do they
    here. To train only the biases and leaks, as their code's default does,
    give the `weight` parameter no optimizer (dew's `ParamGroup`).
    """

    connectome: Connectome = dataclasses.field(kw_only=True)
    types: Sequence[int] = dataclasses.field(kw_only=True)
    input_neurons: Sequence[int] = dataclasses.field(kw_only=True)
    output_neurons: Sequence[int] = dataclasses.field(kw_only=True)
    radius: float = 0.9
    update: float = 0.2
    activation: Literal["tanh", "relu", "sigmoid"] = "tanh"

    def __call__(self, x: jax.Array) -> jax.Array:
        return super().__call__(x)[..., np.asarray(self.output_neurons)]

    def inputs(self, x: jax.Array) -> SynapticInput:
        """`x` `[..., len(input_neurons)]` added onto the neurons `input_neurons`, as their jumps."""
        drive = jnp.zeros((*x.shape[:-1], self.connectome.size), x.dtype)
        return SynapticInput(jump=drive.at[..., np.asarray(self.input_neurons)].add(x))

    def build(self, x: jax.Array) -> SparseRecurrentCell:
        graph = self.connectome
        pre, post = jnp.asarray(graph.pre), jnp.asarray(graph.post)
        counts = jnp.asarray(graph.synapses, jnp.float32)

        def scaled(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jax.Array:
            # The radius comes from the connectome's own counts, so it is computed once, at initialization.
            return (counts * (self.radius / spectral_radius(graph))).astype(dtype)

        weight = self.param("weight", scaled, counts.shape, jnp.float32)
        bias = self.param("bias", nn.initializers.zeros_init(), (graph.size,), jnp.float32)
        start = math.log(self.update / (1 - self.update))
        classes = int(np.max(self.types)) + 1
        logits = self.param("update_logits", nn.initializers.constant(start), (classes,), jnp.float32)
        fraction = 0.99 * jax.nn.sigmoid(logits) + 0.01
        decay = 1 - fraction[jnp.asarray(self.types)]
        return SparseRecurrentCell(RateCell(decay, bias, self.activation), pre, post, weight, graph.size)
