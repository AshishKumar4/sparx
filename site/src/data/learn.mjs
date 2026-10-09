// The course on /learn/: its parts and chapters in reading order. A chapter is listed once its page exists.
import { existsSync } from 'node:fs';
import path from 'node:path';

export const parts = [
	{ title: 'Before you start', chapters: [{ slug: 'why-spikes', title: 'Why spikes?', role: 'spike', summary: 'Where spiking networks beat conventional ones today, where they lose, and how to choose.' }] },
	{
		title: 'Neurons',
		chapters: [
			{ slug: 'neuron', title: 'What a neuron does', role: 'membrane', summary: 'Synapses, a membrane that keeps charge, and the all-or-none spike.' },
			{ slug: 'membranes', title: 'Membranes and time constants', role: 'membrane', summary: 'The RC circuit, the exact solution, and what a step of dt does to it.' },
			{ slug: 'spikes', title: 'Spikes and thresholds', role: 'membrane', summary: 'Threshold, reset, refractoriness, the f-I curve and adaptation.' },
			{ slug: 'coding', title: 'Coding', role: 'membrane', summary: 'Rate, timing and change: three ways to put a number into spikes.' },
		],
	},
	{
		title: 'Learning',
		chapters: [
			{ slug: 'surrogate-gradients', title: 'Surrogate gradients', role: 'learn', summary: 'Why a spike has no gradient, and the stand-in that lets gradient descent through.' },
			{ slug: 'bptt', title: 'Backpropagation through time', role: 'learn', summary: 'Unrolling a network over time, what it costs, and why gradients explode.' },
			{ slug: 'local-rules', title: 'Local learning rules', role: 'learn', summary: 'STDP, three-factor rules and e-prop: learning from what each synapse can see.' },
			{ slug: 'delays', title: 'Delays', role: 'learn', summary: 'Spikes take time to travel. Learning how long turns sequences into coincidences.' },
		],
	},
	{
		title: 'Circuits',
		chapters: [
			{ slug: 'networks', title: 'Networks and dynamics', role: 'bio', summary: 'Balanced excitation and inhibition, irregular firing, and chaos.' },
			{ slug: 'biology', title: 'Physical units and biology', role: 'bio', summary: 'Millivolts and nanosiemens: conductances, receptors and real neurons.' },
			{ slug: 'simulators', title: 'Simulators and fidelity', role: 'bio', summary: 'How a simulator steps time, and how to tell whether two of them agree.' },
		],
	},
	{
		title: 'Machines',
		chapters: [
			{ slug: 'hardware', title: 'Hardware and events', role: 'spike', summary: 'Event cameras, neuromorphic chips, and NIR, the format that moves a network between them.' },
			{ slug: 'drone', title: 'Case study: the drone', role: 'spike', summary: 'A spiking network that learned to fly by gradients through its physics.' },
			{ slug: 'racer', title: 'Case study: the racer', role: 'spike', summary: 'Driving from events alone, trained end to end through the camera.' },
		],
	},
];

// Builds run from the site's directory; this module is bundled elsewhere, so its own URL cannot say where pages are.
const written = (slug) => ['mdx', 'astro'].some((ext) => existsSync(path.join(process.cwd(), 'src/pages/learn', `${slug}.${ext}`)));

/** Every written chapter in order, numbered from 0. */
export const chapters = parts
	.flatMap((part) => part.chapters.map((chapter) => ({ ...chapter, part: part.title })))
	.map((chapter, number) => ({ ...chapter, number }))
	.filter((chapter) => written(chapter.slug));
