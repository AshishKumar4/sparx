// Chapter 9's figure: two copies of one Brunel network, identical until the reader nudges one neuron, drawn
// as one raster where spikes in both copies are grey and spikes in only one are coloured.
import { Twins } from '../engines/twins';
import { alpha, animate, fit, onTheme, type Palette } from './theme';

const ROWS = 120;
const WINDOW = 4000;

export function twins(root: HTMLElement, read: () => { g: number; eta: number }) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	let sim = new Twins({ order: 250, ...read() });
	const a = new Uint8Array(WINDOW * ROWS);
	const b = new Uint8Array(WINDOW * ROWS);
	const share = new Float32Array(WINDOW);
	const counts = new Uint16Array(WINDOW * 2);
	let t = 0;
	let nudgedAt = -1;
	let colors: Palette;
	let view = fit(canvas);

	const step = () => {
		const differ = sim.step();
		const at = t % WINDOW;
		let inA = 0;
		let inB = 0;
		for (let i = 0; i < sim.size; i++) {
			inA += sim.a.network.fired[i];
			inB += sim.b.network.fired[i];
		}
		const fired = inA + inB;
		counts[at * 2] = inA;
		counts[at * 2 + 1] = inB;
		for (let r = 0; r < ROWS; r++) {
			a[at * ROWS + r] = sim.a.network.fired[r];
			b[at * ROWS + r] = sim.b.network.fired[r];
		}
		share[at] = fired ? differ / fired : 0;
		t++;
	};
	for (let k = 0; k < 2000; k++) step();

	const draw = () => {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const left = 8;
		const right = width - 8;
		const split = height * 0.72;
		const pitch = (split - 10) / ROWS;
		const first = Math.max(0, t - WINDOW);
		const x = (s: number) => left + ((s - (t - WINDOW)) / WINDOW) * (right - left);
		const size = Math.max(1.5, Math.min(2.5, pitch * 0.9));
		if (nudgedAt >= first) {
			ctx.strokeStyle = alpha(colors.ink, 0.4);
			ctx.setLineDash([3, 3]);
			ctx.beginPath();
			ctx.moveTo(x(nudgedAt), 2);
			ctx.lineTo(x(nudgedAt), height - 2);
			ctx.stroke();
			ctx.setLineDash([]);
		}
		for (let s = first; s < t; s++) {
			const at = s % WINDOW;
			for (let r = 0; r < ROWS; r++) {
				const ina = a[at * ROWS + r];
				const inb = b[at * ROWS + r];
				if (!ina && !inb) continue;
				ctx.fillStyle = ina && inb ? alpha(colors.ink, 0.55) : ina ? colors.spike : colors.learn;
				ctx.fillRect(x(s) - size / 2, 6 + r * pitch, size, size);
			}
		}
		const top = split + 12;
		const bottom = height - 8;
		ctx.strokeStyle = alpha(colors.ink, 0.12);
		ctx.beginPath();
		ctx.moveTo(left, bottom + 0.5);
		ctx.lineTo(right, bottom + 0.5);
		ctx.moveTo(left, top + 0.5);
		ctx.lineTo(right, top + 0.5);
		ctx.stroke();
		ctx.strokeStyle = colors.bio;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		const bin = 20;
		for (let s = Math.max(first, bin); s < t; s += bin) {
			let sum = 0;
			for (let k = 0; k < bin; k++) sum += share[(s - k) % WINDOW];
			const y = bottom - (sum / bin) * (bottom - top);
			if (s === Math.max(first, bin)) ctx.moveTo(x(s), y);
			else ctx.lineTo(x(s), y);
		}
		ctx.stroke();
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.fillText('share of spikes that differ', left + 4, top + 12);
		if (status) {
			const since = nudgedAt >= 0 ? ((t - nudgedAt) * 0.1).toFixed(0) : null;
			let recent = 0;
			let ra = 0;
			let rb = 0;
			for (let k = 1; k <= 200; k++) {
				const at = (t - k + WINDOW) % WINDOW;
				recent += share[at];
				ra += counts[at * 2];
				rb += counts[at * 2 + 1];
			}
			const hz = (n: number) => (n / sim.size / 0.02).toFixed(0);
			status.textContent =
				since === null
					? `The two copies fire identically, at ${hz(ra)} Hz. Nudge one neuron in the second copy.`
					: `${since} ms since the nudge · ${Math.round((recent / 200) * 100)}% of the last 20 ms of spikes differ · rates ${hz(ra)} and ${hz(rb)} Hz`;
		}
	};

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
		carry += seconds * 500;
		let n = 0;
		while (carry >= 1 && n < 200) {
			step();
			carry--;
			n++;
		}
		if (n === 200) carry = 0;
		draw();
	});
	return {
		nudge() {
			sim.nudge();
			nudgedAt = t;
		},
		restart() {
			sim = new Twins({ order: 250, ...read() });
			a.fill(0);
			b.fill(0);
			share.fill(0);
			counts.fill(0);
			nudgedAt = -1;
			for (let k = 0; k < 2000; k++) step();
			draw();
		},
	};
}
