"""Learning rules beyond surrogate-gradient backpropagation through time (design.md section 7)."""

from sparx.learn.convert import DenseLayer, normalize, relu_forward, run_converted
from sparx.learn.online import (
    EPropParams,
    OTTTLayer,
    accumulate,
    bptt_loss,
    eligibility_traces,
    eprop,
    ottt,
    ottt_dense,
)

__all__ = [
    "DenseLayer",
    "EPropParams",
    "OTTTLayer",
    "accumulate",
    "bptt_loss",
    "eligibility_traces",
    "eprop",
    "normalize",
    "ottt",
    "ottt_dense",
    "relu_forward",
    "run_converted",
]
