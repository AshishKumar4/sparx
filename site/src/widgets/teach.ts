// The surrogate-gradients page's figure: one LIF neuron on 40 input spike trains, taught to fire at the
// times the reader marks, by Adam on the gradient src/engines/teach.ts computes.
import { generator } from '../engines/brunel';
import { surrogates } from '../engines/surrogate';
import { teach } from '../engines/teach';
import { alpha, animate, fit, onTheme, type Palette } from './theme';

const STEPS = 200;
const INPUTS = 40;

export function teacher(root: HTMLElement) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const trainButton = root.querySelector<HTMLButtonElement>('[data-act="train"]') as HTMLButtonElement;
	const random = generator(11);
	const trains = Array.from({ length: STEPS }, () => Array.from({ length: INPUTS }, () => (random() < 0.04 ? 1 : 0)));
	const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
	const start = Float64Array.from({ length: INPUTS }, () => 0.35 * gaussian());
	let w = Float64Array.from(start);
	let m = new Float64Array(INPUTS);
	let s = new Float64Array(INPUTS);
	let updates = 0;
	let target = [40, 90, 150];
	let surrogate = 'ATan';
	let training = false;
	const losses: number[] = [];
	let pass = run();

	function run() {
		const derivative = surrogates().find((x) => x.name === surrogate)?.derivative as (x: number) => number;
		return teach({ trains, target, decay: Math.exp(-1 / 10), threshold: 1, keep: Math.exp(-1 / 10), surrogate: derivative }, w);
	}

	function update() {
		const lr = 0.04;
		updates++;
		for (let i = 0; i < INPUTS; i++) {
			const g = pass.grad[i];
			m[i] = 0.9 * m[i] + 0.1 * g;
			s[i] = 0.999 * s[i] + 0.001 * g * g;
			w[i] -= (lr * (m[i] / (1 - 0.9 ** updates))) / (Math.sqrt(s[i] / (1 - 0.999 ** updates)) + 1e-8);
		}
		pass = run();
		losses.push(pass.loss);
		if (losses.length > 400) losses.shift();
	}

	let colors: Palette;
	let view = fit(canvas);

	const layout = () => {
		const { width, height } = view;
		const left = 12;
		const right = width - 12;
		const raster = { top: 10, bottom: height * 0.34 };
		const membrane = { top: height * 0.4, bottom: height * 0.74 };
		const marks = { top: height * 0.77, bottom: height * 0.84 };
		const loss = { top: height * 0.88, bottom: height - 6 };
		const x = (t: number) => left + ((t + 0.5) / STEPS) * (right - left);
		return { left, right, raster, membrane, marks, loss, x };
	};

	function draw() {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const L = layout();
		const pitch = (L.raster.bottom - L.raster.top) / INPUTS;
		for (let t = 0; t < STEPS; t++) {
			for (let i = 0; i < INPUTS; i++) {
				if (!trains[t][i]) continue;
				const strength = Math.min(1, Math.abs(w[i]) / 0.8);
				ctx.fillStyle = w[i] > 0 ? alpha(colors.ink, 0.25 + 0.6 * strength) : alpha(colors.membrane, 0.25 + 0.6 * strength);
				ctx.fillRect(L.x(t) - 1, L.raster.top + i * pitch, 2, Math.max(1, pitch - 1));
			}
		}
		const lo = -1.5;
		const hi = 2;
		const y = (v: number) => L.membrane.bottom - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (L.membrane.bottom - L.membrane.top);
		ctx.setLineDash([4, 4]);
		ctx.strokeStyle = alpha(colors.spike, 0.6);
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(L.left, Math.round(y(1)) + 0.5);
		ctx.lineTo(L.right, Math.round(y(1)) + 0.5);
		ctx.stroke();
		ctx.setLineDash([]);
		for (const t of target) {
			ctx.fillStyle = alpha(colors.learn, 0.14);
			ctx.fillRect(L.x(t) - 3, L.membrane.top, 6, L.marks.bottom - L.membrane.top);
		}
		ctx.strokeStyle = colors.membrane;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		for (let t = 0; t < STEPS; t++) {
			if (t === 0) ctx.moveTo(L.x(t), y(pass.v[t]));
			else ctx.lineTo(L.x(t), y(pass.v[t]));
		}
		ctx.stroke();
		ctx.strokeStyle = colors.spike;
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let t = 0; t < STEPS; t++) {
			if (!pass.spikes[t]) continue;
			ctx.moveTo(L.x(t), L.membrane.top - 2);
			ctx.lineTo(L.x(t), L.membrane.top + 10);
		}
		ctx.stroke();
		ctx.fillStyle = alpha(colors.ink, 0.06);
		ctx.fillRect(L.left, L.marks.top, L.right - L.left, L.marks.bottom - L.marks.top);
		ctx.fillStyle = colors.learn;
		for (const t of target) {
			const cx = L.x(t);
			ctx.beginPath();
			ctx.moveTo(cx, L.marks.top + 2);
			ctx.lineTo(cx + 5, L.marks.bottom - 2);
			ctx.lineTo(cx - 5, L.marks.bottom - 2);
			ctx.fill();
		}
		if (losses.length > 1) {
			const top = Math.max(...losses, 1e-6);
			ctx.strokeStyle = colors.learn;
			ctx.lineWidth = 1.25;
			ctx.beginPath();
			for (const [k, value] of losses.entries()) {
				const xx = L.left + (k / 399) * (L.right - L.left);
				const yy = L.loss.bottom - (value / top) * (L.loss.bottom - L.loss.top);
				if (k === 0) ctx.moveTo(xx, yy);
				else ctx.lineTo(xx, yy);
			}
			ctx.stroke();
		}
		const fired = pass.spikes.reduce((a, b) => a + b, 0);
		if (status) status.textContent = `loss ${pass.loss.toFixed(4)} · ${updates} updates · ${fired} spikes for ${target.length} targets`;
	}

	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});

	canvas.addEventListener('pointerdown', (event) => {
		const r = canvas.getBoundingClientRect();
		const L = layout();
		const t = Math.round(((event.clientX - r.left - L.left) / (L.right - L.left)) * STEPS - 0.5);
		if (t < 0 || t >= STEPS) return;
		const near = target.findIndex((x) => Math.abs(x - t) <= 3);
		target = near >= 0 ? target.filter((_, k) => k !== near) : [...target, t].sort((a, b) => a - b);
		pass = run();
		draw();
	});
	trainButton.addEventListener('click', () => {
		training = !training;
		trainButton.textContent = training ? 'Pause' : 'Train';
		trainButton.setAttribute('aria-pressed', String(training));
	});
	root.querySelector('[data-act="reset"]')?.addEventListener('click', () => {
		w = Float64Array.from(start);
		m = new Float64Array(INPUTS);
		s = new Float64Array(INPUTS);
		updates = 0;
		losses.length = 0;
		pass = run();
		draw();
	});
	return {
		setSurrogate(name: string) {
			surrogate = name;
			pass = run();
			draw();
		},
		start: animate(canvas, () => {
			if (!training) return;
			update();
			draw();
		}),
	};
}
