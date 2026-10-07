"""Learning rules beyond surrogate-gradient backpropagation through time (design.md section 7)."""

from sparx.learn.convert import SpikingMaxPool, convert, fold_batch_norm, normalize, run_converted
from sparx.learn.events import EventLIF, first_spike_cross_entropy, spike_times
from sparx.learn.online import (
    EPropParams,
    OTTTLayer,
    accumulate,
    bptt_loss,
    eligibility_traces,
    eprop,
    eprop_forward,
    ottt,
    ottt_dense,
)
from sparx.learn.predictive import (
    PredictiveCoding,
    ResidualBlock,
    Settled,
    residual_mlp,
    sequential_blocks,
    squared_error,
)
from sparx.learn.reinforce import ReinforceParams, policy_gradient, reinforce

__all__ = [
    "EPropParams",
    "EventLIF",
    "OTTTLayer",
    "PredictiveCoding",
    "ReinforceParams",
    "ResidualBlock",
    "Settled",
    "SpikingMaxPool",
    "accumulate",
    "bptt_loss",
    "convert",
    "eligibility_traces",
    "eprop",
    "eprop_forward",
    "first_spike_cross_entropy",
    "fold_batch_norm",
    "normalize",
    "ottt",
    "ottt_dense",
    "policy_gradient",
    "reinforce",
    "residual_mlp",
    "run_converted",
    "sequential_blocks",
    "spike_times",
    "squared_error",
]
