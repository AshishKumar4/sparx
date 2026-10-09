// The Learn pages, in reading order.

export const explainers = [
	{ slug: 'neuron', title: 'Neurons', summary: 'Drive a LIF neuron and an Izhikevich cell with current, and watch the membrane, the threshold and the reset.', role: 'membrane' },
	{ slug: 'surrogate-gradients', title: 'Surrogate gradients', summary: 'Pick the times a neuron should fire, and watch gradient descent through its spikes teach it.', role: 'learn' },
	{ slug: 'delays', title: 'Learned delays', summary: 'Each synapse learns how late to deliver its spikes, so a neuron can hear a pattern as a coincidence.', role: 'learn' },
	{ slug: 'plasticity', title: 'Plasticity', summary: 'STDP strengthens the inputs that fire just before a neuron does, until one neuron finds a pattern hidden in noise.', role: 'learn' },
	{ slug: 'networks', title: 'Networks', summary: "Brunel's balanced network in each of its regimes, simulated as sparx simulates it.", role: 'bio' },
	{ slug: 'simulators', title: 'sparx, NEST and Brian2', summary: 'The same neurons in three simulators, overlaid, with the differences measured and named.', role: 'bio' },
	{ slug: 'pilot', title: 'The pilot', summary: 'How the drone on the front page learned to fly, what silencing its neurons does, and how the browser matches sparx.', role: 'spike' },
	{ slug: 'nir', title: 'NIR export', summary: 'The pilot as a NIR graph, the format other simulators and neuromorphic hardware toolchains read.', role: 'spike' },
];
