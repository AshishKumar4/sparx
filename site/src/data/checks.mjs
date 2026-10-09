// What sparx's models are checked against, from docs/fidelity.md, for the front page. `kind` says how
// close: the same numbers, a stated tolerance, or the same statistics of a chaotic system.

export const checks = [
	{ model: 'LIF, current LIF, PSN', reference: 'SpikingJelly and snnTorch', result: 'Spikes exact; gradients within 3.6e-7', kind: 'exact' },
	{ model: 'Biophysical LIF', reference: 'NEST iaf_psc_exp, _alpha, _delta', result: 'Spike for spike; voltages within 1e-11 mV over 300 ms', kind: 'exact' },
	{ model: 'Izhikevich (2004), twenty patterns', reference: "His figure1.m in GNU Octave", result: 'Every spike on his step; voltages to 1e-9 relative', kind: 'exact' },
	{ model: 'Pair, triplet and dopamine STDP', reference: 'NEST stdp_*_synapse', result: 'Every transmitted weight within 1e-10 relative', kind: 'exact' },
	{ model: 'Cortical microcircuit, a fifth', reference: 'NEST on the network sparx draws', result: '12,689 spikes alike over 300 ms, float64', kind: 'exact' },
	{ model: 'DelayedDense (learned delays)', reference: "DCLS, as SNN-delays uses it", result: 'Outputs within 2.4e-7; gradients within 1.4e-6', kind: 'tolerance' },
	{ model: 'Conductance LIF', reference: 'NEST iaf_cond_* (RK45)', result: 'Within 2e-3 mV; error falls 4x when dt halves', kind: 'tolerance' },
	{ model: 'AdEx, Naud et al.\u2019s eight patterns', reference: 'NEST aeif_* (RK45)', result: 'Same spike counts; every spike within 0.8 ms', kind: 'tolerance' },
	{ model: 'Brunel (2000), four regimes', reference: "NEST's brunel_delta_nest.py", result: 'Rate, CV and Fano factor within NEST\u2019s spread over 8 seeds', kind: 'statistics' },
	{ model: 'FlyWire whole brain (Shiu et al.)', reference: 'Their published Brian2 runs', result: 'Rates correlate at 0.9989; MN9 at 67.1 Hz vs 67.0 ± 6.6', kind: 'statistics' },
];

// The README's results: short, untuned runs on a 4-core CPU.
export const results = [
	{ task: 'MNIST, rate-coded, 8 steps', network: '784-512-512 LIF', accuracy: '97.5% after 2 epochs' },
	{ task: "SHD, Hammouamri et al.'s recipe, 20 of 150 epochs", network: '140-256-256 LIF with learned delays', accuracy: '91.9% (their code on the same machine: 93.6%)' },
	{ task: 'SHD, 140 channels', network: '140-128 ALIF, with and without learned delays', accuracy: '74.6% and 64.5%' },
	{ task: "Fashion-MNIST, Seely and Gould's headline cell", network: 'ReLU residual MLP, depth 32', accuracy: 'PC-ALM 75.1%, PC 62.2%, backpropagation 77.8%' },
	{ task: "Pattern completion, Miconi et al.'s task", network: 'plastic recurrent network', accuracy: '0.3% of bits wrong; 50.1% without fast weights' },
];
