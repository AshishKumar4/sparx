# Fidelity ledger

For each model sparx implements: the references it is checked against, every difference found between a reference and the science or another reference, what sparx does, and the test that pins it. [design.md](design.md#11-fidelity-program) describes the tiers. A row is added before a model ships, not after.

Reference versions: SpikingJelly at commit c6cb8e46 (2026-10-03), snnTorch 1.0.0, DCLS 0.1.1 (Hammouamri et al.'s layer), IGITUGraz/eligibility_propagation at its default branch (Bellec et al.), all with torch 2.14.1+cpu. The fixture tools under `tools/` record the versions in each fixture.

## Neurons

| Model | Reference | Checked | Departures and choices |
| --- | --- | --- | --- |
| LIF (`LIFCell`, `nn.LIF`) | SpikingJelly `LIFNode(decay_input=False)`; snnTorch `Leaky(reset_delay=False)` | Spikes exact and gradients within 3.6e-7 of both, soft, hard and detached resets (`tests/test_reference.py`) | Sparx's decay is `exp(-1/tau)`, the exact decay of the leak over a step; SpikingJelly's is `1 - 1/tau`, the Euler step. The cell takes the decay itself, so either convention is a value, and the parity tests pass SpikingJelly's. |
| LIF reset timing | snnTorch's default `reset_delay=True` | Pinned as different (`test_snntorchs_default_delayed_reset_is_a_different_model`) | snnTorch's default subtracts the threshold one step after the spike, without decaying it. That is no discretization of the continuous LIF, whose reset happens at the spike and then decays. Sparx resets on the spiking step. |
| LIF after an overshoot | snnTorch `reset_delay=False` | Pinned (`test_snntorch_loses_spikes_after_an_overshoot_where_sparx_fires`) | When a soft reset leaves the membrane at or above threshold, snnTorch subtracts the next reset before testing for a spike, so the neuron needs twice the threshold to fire again and loses spikes. Every neuron where the two diverge does so on the step after its own overshoot. Sparx and SpikingJelly fire whenever the membrane is at threshold, removing one threshold per spike. |
| LIF spike condition | snnTorch fires on `v - threshold > 0`; SpikingJelly and sparx on `>= 0` | Fixtures keep every membrane 1e-4 from threshold | Differs only at exact equality. |
| Detached reset | snnTorch detaches its delayed reset and keeps the gradient of its immediate one | Gradients match with `detach_reset=False` | Sparx's `detach_reset` is SpikingJelly's option of the same name. |
| Current-based LIF (`SynapticCell`) | snnTorch `Synaptic(reset_delay=False)` | Spikes exact and gradients matched, soft and hard reset | Same reset timing and overshoot departures as LIF. The synapse and the neuron are one cell today; design.md section 4.3 splits them. |
| Adaptive LIF (`ALIFCell`) | Bellec et al. 2020 equations; their code, `Figure_4_and_5_ATARI/alif_eligibility_propagation.py` | Spike for spike against a transcription of their cell with a reset of `decay * threshold` (`test_alif_is_bellecs_model_with_its_reset_decayed`) | A spike subtracts the baseline threshold, as theirs does (sparx subtracted the adaptive threshold before this audit). Their reset lands one step later, undecayed, so their model with reset `decay * threshold` is sparx's. Their repository has a second variant, `LightALIF`, that scales input by `1 - decay` and adaptation by `1 - adapt_decay`; sparx follows the paper's equations, which the ATARI cell implements. Refractoriness is not modeled yet. Their pseudo-derivative is `Triangle(width=threshold, scale=0.3 / threshold)` on `v - A`; sparx's default surrogate is ATan for every neuron. |
| Izhikevich (`IzhikevichCell`) | Izhikevich 2003, the paper's MATLAB code | Spike for spike in float64 against a transcription of the code (`test_izhikevich_matches_his_published_loop_spike_for_spike_in_float64`); regular spiking under input 10 adapts, then fires every 47 to 62 ms | Each 1 ms step takes two half-steps of `v` and one step of `u`, as the code does (sparx took 0.5 ms Euler steps of both before this audit). The firing patterns the model is known for are those of this scheme; an integration closer to the ODE is a separate, named model (design.md section 4.2). The peak is not clipped at 30 mV, as in the 2003 code. |
| Izhikevich numerics | XLA | Measured | Compiled XLA rounds fused arithmetic differently from NumPy in the last bit (22% of elements of one step). The quadratic membrane amplifies that chaotically, so exact comparisons run op by op (`jax.disable_jit`). |
| PSN, MaskedPSN, SlidingPSN | SpikingJelly `neuron/psn.py` | Spikes exact, gradients within 1.2e-6 | None found. MaskedPSN's `lambda_` is the call argument `masking`. |

## Synapses and layers

| Layer | Reference | Checked | Departures and choices |
| --- | --- | --- | --- |
| `DelayedDense` | DCLS `Dcls1d` (version `gauss` in training, `max` with rounded positions at evaluation), as SNN-delays uses it | Outputs within 2.4e-7, gradients (input, weight, delay) within 1.4e-6, both modes (`test_delayed_dense_matches_dcls_delays`) | Sparx's `sigma` is DCLS's effective width, `abs(SIG) + 0.27`, and its delay is `K - 1 - (P + K // 2)`. SNN-delays also pads the input on the right by `(K - 1) // 2`, which lengthens the output so delayed spikes after the input's end reach the readout; sparx keeps the length (the layer is causal and streams exactly), and appending zeros to the input reproduces their setting. Their width decays exponentially to `SIG = 0.23` (effective 0.5) over the first quarter of training; sparx takes any dew schedule, and the examples use a linear one to 0.5. |
| SEW ResNet blocks | SpikingJelly `model/sew_resnet.py` `BasicBlock` | Spike for spike in float64 for ADD, AND and IAND, with and without the downsampling shortcut (`test_sew_block_matches_spikingjelly`) | None found. The `small` stem (one 3x3 convolution, no pooling) is sparx's addition for 32x32 inputs; the `imagenet` stem is SpikingJelly's. |

## Encoders and losses

| Function | Reference | Checked | Departures and choices |
| --- | --- | --- | --- |
| `encode.latency` | snnTorch `spikegen.latency(linear=True, normalize=True, clip=True)` | Exact | snnTorch fires values below the threshold at the last step unless `clip=True`; sparx never fires them. |
| `encode.delta` | snnTorch `spikegen.delta(padding=False)` | Exact, with and without off-spikes | None. |
| `losses.per_step_cross_entropy` | snnTorch `ce_rate_loss` | Within 1e-6 relative | None. |
| `losses.rate_mse` | snnTorch `mse_count_loss` | Within 1e-6 relative, as snnTorch's value divided by `T` | snnTorch's squared error of counts over `T` grows with `T`; sparx's rates do not, and snnTorch rounds its target counts down. |
| Surrogates | The papers' formulas | Gradients and forward-mode tangents, and each surrogate's area | None. |

## Open

- Refractory periods (Bellec's ALIF, every biophysical model) arrive with `sparx.dynamics`.
- Brian2 and NEST comparisons for networks (design.md section 11.3) arrive with the graph engine.
