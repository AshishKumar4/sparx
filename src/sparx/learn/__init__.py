"""Learning rules beyond surrogate-gradient backpropagation through time (design.md section 7)."""

from sparx.learn.convert import (
    AvgPool,
    ConvLayer,
    DenseLayer,
    Flatten,
    Layer,
    MaxPool,
    fold_batch_norm,
    normalize,
    relu_forward,
    run_converted,
)
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
    "AvgPool",
    "ConvLayer",
    "DenseLayer",
    "EPropParams",
    "Flatten",
    "Layer",
    "MaxPool",
    "OTTTLayer",
    "accumulate",
    "bptt_loss",
    "eligibility_traces",
    "eprop",
    "fold_batch_norm",
    "normalize",
    "ottt",
    "ottt_dense",
    "relu_forward",
    "run_converted",
]
