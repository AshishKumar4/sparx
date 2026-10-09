// An oscilloscope for one neuron: its input below, its membrane against the threshold, and its spikes
// above, scrolling as the neuron steps.
import { alpha, animate, fit, onTheme, type Palette } from './theme';

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
	const v = new Float64Array(steps);
	const input = new Float64Array(steps);
	const fired = new Uint8Array(steps);
	let head = 0;
	let colors: Palette;
	let view = fit(canvas);
	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	let count = 0;

	function advance(n: number) {
		for (let k = 0; k < n; k++) {
			const out = trace.step();
			v[head % steps] = out.v;
			input[head % steps] = out.input;
			fired[head % steps] = out.fired;
			head++;
			count += out.fired;
		}
	}

	function draw() {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const spikeBand = 18;
		const inputBand = Math.max(28, height * 0.18);
		const top = spikeBand + 6;
		const bottom = height - inputBand - 8;
		const [lo, hi] = trace.range;
		const y = (value: number) => bottom - ((value - lo) / (hi - lo)) * (bottom - top);
		const x = (c: number) => (c / (steps - 1)) * width;
		const first = Math.max(0, head - steps);

		ctx.strokeStyle = alpha(colors.ink, 0.07);
		ctx.lineWidth = 1;
		ctx.beginPath();
		for (let k = 1; k < 4; k++) {
			const yy = Math.round(top + (k * (bottom - top)) / 4) + 0.5;
			ctx.moveTo(0, yy);
			ctx.lineTo(width, yy);
		}
		ctx.stroke();

		ctx.setLineDash([4, 4]);
		ctx.strokeStyle = alpha(colors.spike, 0.7);
		ctx.beginPath();
		ctx.moveTo(0, Math.round(y(trace.threshold)) + 0.5);
		ctx.lineTo(width, Math.round(y(trace.threshold)) + 0.5);
		ctx.stroke();
		ctx.setLineDash([]);

		const [ilo, ihi] = trace.inputRange;
		const iy = (value: number) => height - 4 - ((value - ilo) / (ihi - ilo)) * (inputBand - 8);
		ctx.fillStyle = alpha(colors.ink, 0.1);
		ctx.beginPath();
		ctx.moveTo(0, iy(Math.max(ilo, 0)));
		for (let t = first, c = steps - (head - first); t < head; t++, c++) ctx.lineTo(x(c), iy(input[t % steps]));
		ctx.lineTo(width, iy(Math.max(ilo, 0)));
		ctx.fill();

		ctx.strokeStyle = colors.membrane;
		ctx.lineWidth = 1.75;
		ctx.lineJoin = 'round';
		ctx.beginPath();
		for (let t = first, c = steps - (head - first); t < head; t++, c++) {
			const yy = y(Math.min(Math.max(v[t % steps], lo), hi));
			if (t === first) ctx.moveTo(x(c), yy);
			else ctx.lineTo(x(c), yy);
		}
		ctx.stroke();

		ctx.strokeStyle = colors.spike;
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let t = first, c = steps - (head - first); t < head; t++, c++) {
			if (!fired[t % steps]) continue;
			ctx.moveTo(x(c), 2);
			ctx.lineTo(x(c), spikeBand);
		}
		ctx.stroke();
	}

	// The window starts full, so the figure reads as running from its first frame.
	advance(steps);
	draw();
	animate(canvas, () => {
		if (reduced) return;
		advance(perFrame);
		draw();
	});
	return {
		/** Spikes since the last call. */
		take() {
			const n = count;
			count = 0;
			return n;
		},
		redraw() {
			if (reduced) advance(steps);
			draw();
		},
	};
}
