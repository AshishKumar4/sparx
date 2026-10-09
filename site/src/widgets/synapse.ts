// Chapter 1's figure: three input neurons onto one current-based LIF neuron (src/engines/neurons.ts),
// stepped at 1 ms and shown slowed down. The reader fires the inputs; until then a short demonstration plays.
import { CurrentLIF } from '../engines/neurons';
import { alpha, animate, fit, onTheme, type Palette } from './theme';

export const INPUTS = [
	{ name: 'A', weight: 0.2 },
	{ name: 'B', weight: 0.2 },
	{ name: 'C', weight: -0.3 },
];
const WINDOW = 400;
// One input alone, two far apart, two together, and two together after an inhibitory one.
const DEMO: [number, number][] = [[20, 0], [90, 0], [110, 1], [190, 0], [192, 1], [280, 2], [284, 0], [286, 1], [360, 1]];

export function synapse(root: HTMLElement) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const neuron = new CurrentLIF(Math.exp(-1 / 5), Math.exp(-1 / 10));
	const fired = new Uint8Array(WINDOW * 3);
	const output = new Uint8Array(WINDOW);
	const current = new Float64Array(WINDOW);
	const membrane = new Float64Array(WINDOW);
	const queued: number[] = [];
	let t = 0;
	let demo = true;
	let colors: Palette;
	let view = fit(canvas);

	const step = () => {
		const at = t % WINDOW;
		let x = 0;
		for (let k = 0; k < 3; k++) fired[at * 3 + k] = 0;
		const due = demo ? DEMO.filter(([when]) => when === t % 440).map(([, k]) => k) : queued.splice(0);
		for (const k of due) {
			fired[at * 3 + k] = 1;
			x += INPUTS[k].weight;
		}
		output[at] = neuron.step(x);
		current[at] = neuron.i;
		membrane[at] = output[at] ? neuron.cell.peak : neuron.cell.v;
		t++;
	};

	const draw = () => {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const left = 34;
		const right = width - 10;
		const x = (c: number) => left + (c / (WINDOW - 1)) * (right - left);
		const first = Math.max(0, t - WINDOW);
		const lane = 18;
		const top = 34;
		ctx.font = `500 11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.textBaseline = 'middle';
		for (let k = 0; k < 3; k++) {
			const y = top + k * lane;
			ctx.fillStyle = colors.muted;
			ctx.fillText(INPUTS[k].name, 8, y);
			ctx.strokeStyle = alpha(colors.ink, 0.08);
			ctx.beginPath();
			ctx.moveTo(left, y + 0.5);
			ctx.lineTo(right, y + 0.5);
			ctx.stroke();
			ctx.strokeStyle = INPUTS[k].weight > 0 ? colors.ink : colors.membrane;
			ctx.lineWidth = 2.5;
			ctx.beginPath();
			for (let s = first; s < t; s++) {
				if (!fired[(s % WINDOW) * 3 + k]) continue;
				const c = s - (t - WINDOW);
				ctx.moveTo(x(c), y - 7);
				ctx.lineTo(x(c), y + 7);
			}
			ctx.stroke();
		}
		const trace = (values: Float64Array, top: number, bottom: number, lo: number, hi: number, color: string, label: string) => {
			const y = (v: number) => bottom - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (bottom - top);
			ctx.fillStyle = colors.muted;
			ctx.fillText(label, 8, top + 6);
			ctx.strokeStyle = alpha(colors.ink, 0.12);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(left, Math.round(y(0)) + 0.5);
			ctx.lineTo(right, Math.round(y(0)) + 0.5);
			ctx.stroke();
			ctx.strokeStyle = color;
			ctx.lineWidth = 1.75;
			ctx.beginPath();
			for (let s = first; s < t; s++) {
				const c = s - (t - WINDOW);
				if (s === first) ctx.moveTo(x(c), y(values[s % WINDOW]));
				else ctx.lineTo(x(c), y(values[s % WINDOW]));
			}
			ctx.stroke();
			return y;
		};
		const split = top + 3 * lane - 4;
		trace(current, split, split + (height - split) * 0.3, -0.35, 0.45, colors.learn, 'i');
		const y = trace(membrane, split + (height - split) * 0.38, height - 22, -0.4, 1.3, colors.membrane, 'v');
		ctx.setLineDash([4, 4]);
		ctx.strokeStyle = alpha(colors.spike, 0.7);
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(left, Math.round(y(1)) + 0.5);
		ctx.lineTo(right, Math.round(y(1)) + 0.5);
		ctx.stroke();
		ctx.setLineDash([]);
		ctx.strokeStyle = colors.spike;
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let s = first; s < t; s++) {
			if (!output[s % WINDOW]) continue;
			const c = s - (t - WINDOW);
			ctx.moveTo(x(c), height - 16);
			ctx.lineTo(x(c), height - 4);
		}
		ctx.stroke();
		ctx.fillStyle = colors.muted;
		ctx.fillText('out', 8, height - 10);
	};

	for (let k = 0; k < 400; k++) step();
	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	let carry = 0;
	animate(canvas, (seconds) => {
		carry += seconds * 140;
		while (carry >= 1) {
			step();
			carry--;
		}
		draw();
	});
	for (const button of root.querySelectorAll<HTMLButtonElement>('[data-input]')) {
		button.addEventListener('click', () => {
			demo = false;
			queued.push(Number(button.dataset.input));
		});
	}
}
