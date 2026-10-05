"""Connectomes as networks: reading synapse tables, and the models published on them.

    brain = Connectome.from_shiu("Completeness_783.csv", "Connectivity_783.parquet")
    sugar = brain.index(SUGAR_IDS)                      # FlyWire root IDs -> neuron indices
    network = shiu2024(brain, stimuli=[(sugar, 150.0)])
    result = simulate(network, network.init(key), duration=1000.0, key=key,
                      monitors=(SpikeCounts("brain"),))

A `Connectome` is a neuron table and an edge list with signed synapse
counts: the sign is the presynaptic neuron's predicted transmitter's
(acetylcholine excitatory; GABA and glutamate inhibitory, as Shiu et al.
assign them for the fly), the count is how many synapses join the pair.
Reading the tables needs the `connectome` extra (pyarrow).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sparx.dynamics.neurons import LIF
from sparx.dynamics.synapses import Delta, Exponential, Receptor
from sparx.graph.connectivity import FromEdges
from sparx.graph.network import Network, PoissonInput, Population, Projection

__all__ = ["SIGNS", "Connectome", "shiu2024"]

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
        consensus = np.asarray(transmitters.column("consensus_nt").to_pylist(), object)[known]
        predicted = np.asarray(transmitters.column("predicted_nt").to_pylist(), object)[known]
        unclear = np.asarray([c is None or c == "unclear" for c in consensus])
        chosen = np.where(unclear, predicted, consensus)
        fallback = signs.get("unclear", 1)
        sign = np.full(len(ids), fallback, np.int32)
        sign[index[known]] = [signs.get(t, fallback) for t in chosen]
        return cls(ids, pre, post, counts * sign[pre])


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
    (indices) lose their outgoing synapses.

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
