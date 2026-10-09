// The simulators page's figures: sparx's voltage beside a reference simulator's on the same inputs, with
// their difference on a log scale, and Izhikevich's twenty patterns beside his code's.
import { alpha, fit, onTheme, type Palette } from './theme';

interface Trace {
	v: number[];
	spikes: number[];
}

export interface Case {
	difference: number[];
	name: string;
	title: string;
	reference: string;
	kind: string;
	dt: number;
	note: string;
	test: string;
	sparx: Trace;
	theirs: Trace;
	error: number;
	same_spikes: boolean;
}

export interface Panel {
	panel: string;
	pattern: string;
	dt: number;
	sparx: number[];
	theirs: number[];
	same_spikes: boolean;
	spikes: number;
	error: number;
}

export function overlay(canvas: HTMLCanvasElement) {
	let colors: Palette;
	let view = fit(canvas);
	let shown: Case | null = null;
	let zoom: [number, number] = [0, 1];
	const draw = () => {
		if (!colors || !shown) return;
		const c = shown;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const n = c.sparx.v.length;
		const from = Math.floor(zoom[0] * n);
		const to = Math.max(from + 10, Math.floor(zoom[1] * n));
		const left = 44;
		const right = width - 10;
		const x = (t: number) => left + ((t - from) / (to - from)) * (right - left);
		const top = 10;
		const split = height * 0.68;
		let lo = Infinity;
		let hi = -Infinity;
		for (let t = from; t < to; t++) {
			lo = Math.min(lo, c.sparx.v[t], c.theirs.v[t]);
			hi = Math.max(hi, c.sparx.v[t], c.theirs.v[t]);
		}
		const pad = (hi - lo) * 0.08 + 1e-9;
		const y = (v: number) => split - 8 - ((v - lo + pad) / (hi - lo + 2 * pad)) * (split - 8 - top);
		ctx.font = `10px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.textAlign = 'right';
		ctx.fillText(`${hi.toFixed(0)} mV`, left - 6, y(hi) + 3);
		ctx.fillText(`${lo.toFixed(0)}`, left - 6, y(lo) + 3);
		const line = (v: number[], color: string, width: number, dash: number[]) => {
			ctx.strokeStyle = color;
			ctx.lineWidth = width;
			ctx.setLineDash(dash);
			ctx.beginPath();
			for (let t = from; t < to; t++) {
				if (t === from) ctx.moveTo(x(t), y(v[t]));
				else ctx.lineTo(x(t), y(v[t]));
			}
			ctx.stroke();
			ctx.setLineDash([]);
		};
		line(c.theirs.v, alpha(colors.spike, 0.9), 3.5, []);
		line(c.sparx.v, colors.membrane, 1.25, []);
		const bottom = height - 18;
		const ly = (e: number) => bottom - ((Math.log10(Math.max(e, 1e-16)) + 16) / 16) * (bottom - split - 6);
		ctx.strokeStyle = alpha(colors.ink, 0.1);
		ctx.lineWidth = 1;
		ctx.beginPath();
		for (const e of [1e-12, 1e-8, 1e-4, 1]) {
			ctx.moveTo(left, Math.round(ly(e)) + 0.5);
			ctx.lineTo(right, Math.round(ly(e)) + 0.5);
		}
		ctx.stroke();
		ctx.fillStyle = colors.muted;
		for (const [e, label] of [[1e-12, '1e-12'], [1e-8, '1e-8'], [1e-4, '1e-4'], [1, '1 mV']] as const) ctx.fillText(label, left - 6, ly(e) + 3);
		ctx.fillStyle = colors.learn;
		let exact = 0;
		for (let t = from; t < to; t++) {
			const e = c.difference[t];
			if (e === 0) {
				exact++;
				continue;
			}
			ctx.fillRect(x(t) - 0.75, ly(e) - 0.75, 1.5, 1.5);
		}
		if (exact === to - from) {
			ctx.fillStyle = colors.bio;
			ctx.fillText('identical at every step: a difference of exactly 0', left + 8, ly(1e-8) + 3);
		}
		ctx.textAlign = 'left';
		ctx.fillStyle = colors.muted;
		ctx.fillText(`|difference|, log scale · ${((to - from) * c.dt).toFixed(0)} ms shown`, left, height - 4);
	};
	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	return {
		show(c: Case) {
			shown = c;
			draw();
		},
		zoom(span: [number, number]) {
			zoom = span;
			draw();
		},
	};
}

export function panels(root: HTMLElement, data: Panel[]) {
	let colors: Palette;
	const canvases = [...root.querySelectorAll<HTMLCanvasElement>('canvas[data-panel]')];
	const draw = () => {
		if (!colors) return;
		for (const canvas of canvases) {
			if (!canvas.getBoundingClientRect().width) continue;
			const p = data[Number(canvas.dataset.panel)];
			const { ctx, width, height } = fit(canvas);
			ctx.clearRect(0, 0, width, height);
			const n = p.sparx.length;
			const x = (t: number) => 2 + (t / (n - 1)) * (width - 4);
			const y = (v: number) => height - 3 - ((Math.min(Math.max(v, -90), 35) + 90) / 125) * (height - 6);
			for (const [trace, color, w] of [[p.theirs, alpha(colors.spike, 0.9), 3], [p.sparx, colors.membrane, 1.1]] as const) {
				ctx.strokeStyle = color;
				ctx.lineWidth = w;
				ctx.beginPath();
				for (let t = 0; t < n; t++) {
					if (t === 0) ctx.moveTo(x(t), y(trace[t]));
					else ctx.lineTo(x(t), y(trace[t]));
				}
				ctx.stroke();
			}
		}
	};
	onTheme((p) => {
		colors = p;
		draw();
	});
	new ResizeObserver(draw).observe(root);
}
