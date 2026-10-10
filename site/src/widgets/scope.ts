// An oscilloscope for one neuron: its spikes along the top, its membrane against the threshold, and its
// input below, scrolling as the neuron steps; drawn by traces.ts.
import { type Band, traces } from './traces';

export interface Trace {
	/** The membrane after the step, its input, and whether it fired. */
	step(): { v: number; input: number; fired: number };
	threshold: number;
	/** The membrane's axis, low to high, in its own unit. */
	range: [number, number];
	inputRange: [number, number];
	unit?: string;
}

export function scope(canvas: HTMLCanvasElement, trace: Trace, { steps = 360, perFrame = 1 } = {}) {
	const threshold = { at: trace.threshold, label: '', role: 'spike' as const };
	const bands: Band[] = [
		{ weight: 4, range: trace.range, lines: ['membrane'], grid: 4, guides: [threshold] },
		{ weight: 1, range: trace.inputRange, lines: ['ink'], fill: true },
	];
	let count = 0;
	const view = traces(
		canvas,
		bands,
		() => {
			const out = trace.step();
			threshold.at = trace.threshold;
			count += out.fired;
			return { values: [out.v, out.input], event: out.fired === 1 };
		},
		{ steps, perFrame, marks: 'spike' },
	);
	return {
		/** Spikes since the last call. */
		take() {
			const n = count;
			count = 0;
			return n;
		},
		redraw: view.redraw,
	};
}
