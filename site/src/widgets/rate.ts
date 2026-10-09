// Chapter 3's figure: a neuron on a constant input, on a scope, beside its firing rate against every
// input (the f-I curve) for each of three neurons, with the reader's input marked.
import { ALIF, LIF } from '../engines/neurons';
import { scope } from './scope';
import { alpha, fit, onTheme, type Palette } from './theme';

export type Kind = 'subtract' | 'zero' | 'alif';
const DECAY = Math.exp(-1 / 12);
const make = (kind: Kind) => (kind === 'alif' ? new ALIF(DECAY, Math.exp(-1 / 100), 0.3) : new LIF(DECAY, 1, kind));

/** Spikes per step over `steps` steps of a constant input `x`, after `warm` steps to settle. */
export function rate(kind: Kind, x: number, steps = 2000, warm = 500): number {
	const cell = make(kind);
	let n = 0;
	for (let t = 0; t < warm + steps; t++) {
		const s = cell.step(x);
		if (t >= warm) n += s;
	}
	return n / steps;
}

/** The zero-reset neuron's rate from the closed form: it needs the first `n` with `x (1 - β^n) / (1 - β) ≥ 1`. */
export function zeroRate(x: number, beta = DECAY): number {
	if (x >= 1) return 1;
	if (x <= 1 - beta) return 0;
	return 1 / Math.ceil(Math.log(1 - (1 - beta) / x) / Math.log(beta) - 1e-12);
}

const COLORS: Record<Kind, keyof Palette> = { subtract: 'membrane', zero: 'ink2', alif: 'learn' };
const LABELS: Record<Kind, string> = { subtract: 'subtract', zero: 'to zero', alif: 'adaptive' };

export function rateFigure(root: HTMLElement, read: () => { kind: Kind; x: number }) {
	const scopeCanvas = root.querySelector<HTMLCanvasElement>('canvas[data-scope]') as HTMLCanvasElement;
	const curveCanvas = root.querySelector<HTMLCanvasElement>('canvas[data-curve]') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const xs = Array.from({ length: 121 }, (_, k) => (k / 120) * 1.2);
	const curves = Object.fromEntries((['subtract', 'zero', 'alif'] as Kind[]).map((k) => [k, xs.map((x) => rate(k, x))])) as Record<Kind, number[]>;
	let kind = read().kind;
	let cell = make(kind);
	const view = scope(scopeCanvas, {
		threshold: 1,
		range: [-0.2, 1.8],
		inputRange: [-0.2, 1.4],
		step() {
			const { x } = read();
			const fired = cell.step(x);
			return { v: fired ? cell.peak : cell.v, input: x, fired };
		},
	}, { steps: 240 });
	let colors: Palette;
	let curveView = fit(curveCanvas);
	const draw = () => {
		if (!colors) return;
		const { x } = read();
		const { ctx, width, height } = curveView;
		ctx.clearRect(0, 0, width, height);
		const left = 34;
		const right = width - 10;
		const top = 12;
		const bottom = height - 36;
		const px = (v: number) => left + (v / 1.2) * (right - left);
		const py = (r: number) => bottom - r * (bottom - top);
		ctx.font = `11px ${getComputedStyle(curveCanvas).getPropertyValue('--sx-mono')}`;
		ctx.strokeStyle = alpha(colors.ink, 0.12);
		ctx.beginPath();
		for (const r of [0, 0.5, 1]) {
			ctx.moveTo(left, Math.round(py(r)) + 0.5);
			ctx.lineTo(right, Math.round(py(r)) + 0.5);
		}
		ctx.stroke();
		ctx.fillStyle = colors.muted;
		ctx.textAlign = 'right';
		for (const r of [0, 0.5, 1]) ctx.fillText(String(r), left - 6, py(r) + 4);
		ctx.textAlign = 'center';
		for (const v of [0, 0.4, 0.8, 1.2]) ctx.fillText(String(v), Math.min(px(v), right - 8), bottom + 15);
		ctx.fillText('input per step', (left + right) / 2, height - 5);
		ctx.setLineDash([3, 3]);
		ctx.strokeStyle = alpha(colors.ink, 0.35);
		ctx.beginPath();
		ctx.moveTo(px(1 - DECAY), top);
		ctx.lineTo(px(1 - DECAY), bottom);
		ctx.stroke();
		ctx.setLineDash([]);
		for (const k of ['zero', 'alif', 'subtract'] as Kind[]) {
			const color = colors[COLORS[k]] as string;
			ctx.strokeStyle = k === kind ? color : alpha(color, 0.45);
			ctx.lineWidth = k === kind ? 2.25 : 1.25;
			ctx.beginPath();
			for (const [i, v] of xs.entries()) {
				if (i === 0) ctx.moveTo(px(v), py(curves[k][i]));
				else ctx.lineTo(px(v), py(curves[k][i]));
			}
			ctx.stroke();
		}
		ctx.textAlign = 'left';
		let ly = top + 10;
		for (const k of ['subtract', 'zero', 'alif'] as Kind[]) {
			ctx.fillStyle = colors[COLORS[k]] as string;
			ctx.fillRect(left + 8, ly - 4, 12, 2.5);
			ctx.fillText(LABELS[k], left + 26, ly);
			ly += 15;
		}
		const r = rate(kind, x);
		ctx.fillStyle = colors.spike;
		ctx.beginPath();
		ctx.arc(px(x), py(r), 5, 0, 2 * Math.PI);
		ctx.fill();
		if (status) status.textContent = `input ${x.toFixed(3)} per step: ${(r * 100).toFixed(1)} spikes per 100 steps once settled · rheobase 1 − β = ${(1 - DECAY).toFixed(3)} · LIFCell and ALIFCell, τ = 12 steps`;
	};
	new ResizeObserver(() => {
		curveView = fit(curveCanvas);
		draw();
	}).observe(curveCanvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	return () => {
		const next = read().kind;
		if (next !== kind) {
			kind = next;
			cell = make(kind);
		}
		view.redraw();
		draw();
	};
}
