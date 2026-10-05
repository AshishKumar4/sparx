"""Learning rules beyond surrogate-gradient backpropagation through time (design.md section 7)."""

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

__all__ = ["EPropParams", "OTTTLayer", "accumulate", "bptt_loss", "eligibility_traces", "eprop", "ottt",
           "ottt_dense"]
