// Chapter 7's figure: two copies of one recurrent spiking network, from the same starting weights, learning
// to draw the same curve, one by e-prop and one by BPTT (src/engines/eprop.ts), one update a frame.
import { generator } from '../engines/brunel';
import { gradients, type Params } from '../engines/eprop';
import { alpha, animate, fit, onTheme, type Palette } from './theme';

const I = 20;
const N = 50;
const T = 200;
const KEYS = ['wIn', 'wRec', 'wOut', 'bOut'] as const;
const SETUP = { beta: Math.exp(-1 / 20), kappa: Math.exp(-1 / 10), threshold: 1, surrogate: (x: number) => 0.3 * Math.max(0, 1 - Math.abs(x)) };

class Learner {
	readonly m: Record<string, Float64Array> = {};
	readonly v: Record<string, Float64Array> = {};
	updates = 0;
	losses: number[] = [];
	y: Float64Array<ArrayBufferLike> = new Float64Array(T);

	constructor(
		readonly params: Params,
		readonly rule: 'eprop' | 'bptt',
	) {
		for (const k of KEYS) {
			this.m[k] = new Float64Array(params[k].length);
			this.v[k] = new Float64Array(params[k].length);
		}
	}

	update(u: number[][], target: number[][]) {
		const pass = gradients(this.params, SETUP, u, target);
		const g = pass[this.rule];
		this.updates++;
		const t = this.updates;
		for (const k of KEYS) {
			const p = this.params[k];
			const m = this.m[k];
			const v = this.v[k];
			for (let i = 0; i < p.length; i++) {
				m[i] = 0.9 * m[i] + 0.1 * g[k][i];
				v[i] = 0.999 * v[i] + 0.001 * g[k][i] ** 2;
				p[i] -= (0.01 * (m[i] / (1 - 0.9 ** t))) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8);
			}
		}
		this.y = pass.y;
		this.losses.push(pass.loss);
		return pass;
	}
}

export function epropFigure(root: HTMLElement) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const button = root.querySelector<HTMLButtonElement>('[data-act="train"]') as HTMLButtonElement;
	const random = generator(3);
	const gauss = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
	const u = Array.from({ length: T }, (_, t) => Array.from({ length: I }, (_, i) => ((t + 10 * i) % 200 < 10 && random() < 0.6 ? 1 : 0)));
	const target = Array.from({ length: T }, (_, t) => [
		0.6 * Math.sin((2 * Math.PI * t) / 100) + 0.3 * Math.sin((2 * Math.PI * t) / 50 + 1) + 0.2 * Math.sin((2 * Math.PI * t) / 33 + 2),
	]);
	const start: Params = {
		inputs: I,
		size: N,
		outputs: 1,
		wIn: Float64Array.from({ length: I * N }, () => gauss()),
		wRec: Float64Array.from({ length: N * N }, () => 0.15 * gauss()),
		wOut: Float64Array.from({ length: N }, () => 0.05 * gauss()),
		bOut: new Float64Array(1),
	};
	const clone = (p: Params): Params => ({ ...p, wIn: p.wIn.slice(), wRec: p.wRec.slice(), wOut: p.wOut.slice(), bOut: p.bOut.slice() });
	let learners = [new Learner(clone(start), 'eprop'), new Learner(clone(start), 'bptt')];
	let cosine = 0;
	let training = false;
	let colors: Palette;
	let view = fit(canvas);

	const first = gradients(start, SETUP, u, target);
	for (const l of learners) l.y = first.y;

	const measure = () => {
		const pass = gradients(learners[0].params, SETUP, u, target);
		let ab = 0;
		let aa = 0;
		let bb = 0;
		for (const k of ['wIn', 'wRec', 'wOut'] as const) {
			for (let i = 0; i < pass.eprop[k].length; i++) {
				ab += pass.eprop[k][i] * pass.bptt[k][i];
				aa += pass.eprop[k][i] ** 2;
				bb += pass.bptt[k][i] ** 2;
			}
		}
		cosine = ab / Math.sqrt(aa * bb);
	};
	measure();

	const draw = () => {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const wide = width > 560;
		const plotW = wide ? width * 0.66 : width;
		const half = wide ? height / 2 : height * 0.36;
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		const names = ['e-prop', 'BPTT'];
		for (const [k, l] of learners.entries()) {
			const top = k * half + 8;
			const bottom = (k + 1) * half - 8;
			const x = (t: number) => 12 + (t / (T - 1)) * (plotW - 24);
			const y = (v: number) => (top + bottom) / 2 - v * ((bottom - top) / 2.6);
			ctx.strokeStyle = alpha(colors.ink, 0.35);
			ctx.lineWidth = 3;
			ctx.beginPath();
			for (let t = 0; t < T; t++) (t ? ctx.lineTo : ctx.moveTo).call(ctx, x(t), y(target[t][0]));
			ctx.stroke();
			ctx.strokeStyle = k ? colors.membrane : colors.learn;
			ctx.lineWidth = 1.75;
			ctx.beginPath();
			for (let t = 0; t < T; t++) (t ? ctx.lineTo : ctx.moveTo).call(ctx, x(t), y(Math.max(-1.3, Math.min(1.3, l.y[t]))));
			ctx.stroke();
			ctx.fillStyle = k ? colors.membrane : colors.learn;
			ctx.fillText(names[k], 14, top + 10);
		}
		const lx = wide ? plotW + 16 : 12;
		const lw = wide ? width - plotW - 28 : width - 24;
		const ltop = wide ? 14 : 2 * half + 10;
		const lbottom = height - 18;
		const all = learners.flatMap((l) => l.losses);
		const hi = Math.log10(Math.max(...all, 10));
		const lo = Math.log10(Math.max(Math.min(...all, 1), 1e-3)) - 0.2;
		const ly = (v: number) => lbottom - ((Math.log10(v) - lo) / (hi - lo)) * (lbottom - ltop);
		ctx.fillStyle = colors.muted;
		ctx.fillText('loss, log scale', lx, ltop - 2);
		const n = Math.max(1, ...learners.map((l) => l.losses.length));
		for (const [k, l] of learners.entries()) {
			ctx.strokeStyle = k ? colors.membrane : colors.learn;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			for (const [i, v] of l.losses.entries()) (i ? ctx.lineTo : ctx.moveTo).call(ctx, lx + (i / Math.max(n - 1, 1)) * lw, ly(v));
			ctx.stroke();
		}
		if (status) {
			const [e, b] = learners;
			const last = (l: Learner) => (l.losses.length ? l.losses.at(-1)!.toFixed(2) : '–');
			status.textContent = `${e.updates} updates · loss: e-prop ${last(e)}, BPTT ${last(b)} · their gradients' cosine similarity ${cosine.toFixed(3)} · e-prop keeps ${(2 * N * (I + N)).toLocaleString('en-US')} numbers whatever the length; BPTT stores ${(2 * T * N).toLocaleString('en-US')}, growing with it`;
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
	button.addEventListener('click', () => {
		training = !training;
		button.textContent = training ? 'Pause' : 'Train both';
	});
	root.querySelector('[data-act="reset"]')?.addEventListener('click', () => {
		learners = [new Learner(clone(start), 'eprop'), new Learner(clone(start), 'bptt')];
		for (const l of learners) l.y = first.y;
		measure();
		draw();
	});
	animate(canvas, () => {
		if (!training) return;
		for (const l of learners) l.update(u, target);
		if (learners[0].updates % 5 === 0) measure();
		draw();
	});
}
