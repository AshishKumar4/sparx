// The plasticity page's figures: PairSTDP's window, and one neuron finding a pattern hidden in noise
// (src/engines/stdp.ts, src/engines/pattern.ts).
import { Pattern } from '../engines/pattern';
import { PairSTDP, type Rule } from '../engines/stdp';
import { alpha, animate, fit, onTheme, type Palette } from './theme';

/** The weight change of one pair, a presynaptic spike `lag` ms before a postsynaptic one, from w = 0.5. */
export function change(rule: Rule, lag: number, dt = 0.1): number {
	const stdp = new PairSTDP(rule, 1, 1);
	const w = Float64Array.of(0.5 * rule.w_max);
	const pre = new Int32Array(1);
	const steps = Math.round(Math.abs(lag) / dt) + 2;
	const first = 1;
	for (let t = 0; t < steps; t++) {
		const preAt = lag >= 0 ? first : first + Math.round(-lag / dt);
		const postAt = lag >= 0 ? first + Math.round(lag / dt) : first;
		stdp.step(w, Uint8Array.of(t === preAt ? 1 : 0), Uint8Array.of(t === postAt ? 1 : 0), pre, pre, dt);
	}
	return w[0] - 0.5 * rule.w_max;
}

export function windowFigure(root: HTMLElement, read: () => Rule) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	let colors: Palette;
	let view = fit(canvas);
	const draw = () => {
		if (!colors) return;
		const rule = read();
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const lags = Array.from({ length: 81 }, (_, k) => -80 + 2 * k).filter((l) => l !== 0);
		const changes = lags.map((l) => change(rule, l));
		const top = Math.max(1e-9, ...changes.map(Math.abs));
		const cx = (l: number) => 30 + ((l + 80) / 160) * (width - 50);
		const cy = (c: number) => height / 2 - (c / top) * (height / 2 - 26);
		ctx.strokeStyle = alpha(colors.ink, 0.15);
		ctx.beginPath();
		ctx.moveTo(30, Math.round(height / 2) + 0.5);
		ctx.lineTo(width - 20, Math.round(height / 2) + 0.5);
		ctx.moveTo(Math.round(cx(0)) + 0.5, 14);
		ctx.lineTo(Math.round(cx(0)) + 0.5, height - 14);
		ctx.stroke();
		for (const [k, l] of lags.entries()) {
			ctx.fillStyle = changes[k] > 0 ? colors.learn : colors.membrane;
			ctx.beginPath();
			ctx.arc(cx(l), cy(changes[k]), 3, 0, 2 * Math.PI);
			ctx.fill();
		}
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.textAlign = 'center';
		ctx.fillText('post after pre (ms) →', cx(40), height - 6);
		ctx.fillText('← pre after post', cx(-40), height - 6);
		ctx.textAlign = 'left';
		ctx.fillStyle = colors.learn;
		ctx.fillText('stronger', cx(4), 16);
		ctx.fillStyle = colors.membrane;
		ctx.fillText('weaker', cx(-78), height - 22);
	};
	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	return draw;
}

export function patternFigure(root: HTMLElement) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const span = 600;
	const shown = 120;
	let sim = new Pattern();
	let speed = 10;
	const raster = new Uint8Array(span * shown);
	const voltage = new Float64Array(span);
	const fired = new Uint8Array(span);
	const phase = new Int16Array(span);
	const recent: { t: number; inside: boolean }[] = [];
	let windows: number[] = [];
	let colors: Palette;
	let view = fit(canvas);
	const rows = Array.from({ length: shown }, (_, k) => (k < shown / 2 ? k : sim.setup.afferents / 2 + (k - shown / 2)));

	const step = () => {
		const f = sim.step();
		const at = (sim.t - 1) % span;
		for (let k = 0; k < shown; k++) raster[at * shown + k] = sim.spikes[rows[k]];
		voltage[at] = f ? sim.setup.threshold * 1.1 : sim.neuron.v;
		fired[at] = f;
		phase[at] = sim.phase;
		if (sim.phase === 0) windows.push(sim.t);
		if (f) recent.push({ t: sim.t, inside: sim.phase >= 0 });
		while (recent.length && recent[0].t < sim.t - 10000) recent.shift();
		windows = windows.filter((t) => t > sim.t - 10000);
	};

	const draw = () => {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const left = 8;
		const plotW = width * (width > 640 ? 0.68 : 1) - left;
		const rasterBottom = height * (width > 640 ? 0.62 : 0.45);
		const pitch = (rasterBottom - 8) / shown;
		const x = (c: number) => left + (c / span) * plotW;
		const first = Math.max(0, sim.t - span);
		for (let t = first; t < sim.t; t++) {
			const at = t % span;
			const c = t - (sim.t - span);
			if (phase[at] >= 0) {
				ctx.fillStyle = alpha(colors.learn, 0.1);
				ctx.fillRect(x(c), 4, plotW / span + 0.5, height - 8);
			}
		}
		for (let t = first; t < sim.t; t++) {
			const at = t % span;
			const c = t - (sim.t - span);
			for (let k = 0; k < shown; k++) {
				if (!raster[at * shown + k]) continue;
				ctx.fillStyle = k < shown / 2 ? alpha(colors.ink, 0.75) : alpha(colors.ink, 0.4);
				ctx.fillRect(x(c), 8 + k * pitch, 1.6, Math.max(1, pitch * 0.9));
			}
		}
		const vTop = rasterBottom + 14;
		const vBottom = width > 640 ? height - 10 : height * 0.72;
		const vmax = sim.setup.threshold * 1.15;
		const y = (v: number) => vBottom - (Math.max(0, Math.min(v, vmax)) / vmax) * (vBottom - vTop);
		ctx.setLineDash([4, 4]);
		ctx.strokeStyle = alpha(colors.spike, 0.6);
		ctx.beginPath();
		ctx.moveTo(left, y(sim.setup.threshold));
		ctx.lineTo(left + plotW, y(sim.setup.threshold));
		ctx.stroke();
		ctx.setLineDash([]);
		ctx.strokeStyle = colors.membrane;
		ctx.lineWidth = 1.25;
		ctx.beginPath();
		for (let t = first; t < sim.t; t++) {
			const c = t - (sim.t - span);
			if (t === first) ctx.moveTo(x(c), y(voltage[t % span]));
			else ctx.lineTo(x(c), y(voltage[t % span]));
		}
		ctx.stroke();
		ctx.strokeStyle = colors.spike;
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let t = first; t < sim.t; t++) {
			if (!fired[t % span]) continue;
			const c = t - (sim.t - span);
			ctx.moveTo(x(c), vTop - 10);
			ctx.lineTo(x(c), vTop);
		}
		ctx.stroke();
		const wide = width > 640;
		const wl = wide ? left + plotW + 24 : left;
		const wt = wide ? 8 : vBottom + 18;
		const ww = wide ? width - wl - 8 : width - 16;
		const wh = wide ? height - 16 : height - wt - 8;
		const n = sim.setup.afferents;
		const bar = ww / n;
		ctx.fillStyle = alpha(colors.ink, 0.04);
		ctx.fillRect(wl, wt, ww, wh);
		for (let i = 0; i < n; i++) {
			const w = sim.weights[i];
			ctx.fillStyle = i < n / 2 ? colors.learn : alpha(colors.ink, 0.45);
			ctx.fillRect(wl + i * bar, wt + wh - w * wh, Math.max(1, bar - 0.2), w * wh);
		}
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.fillText('weights: pattern afferents | others', wl + 4, wt + 14);
		const inside = recent.filter((r) => r.inside).length;
		const hit = windows.filter((w0) => recent.some((r) => r.inside && r.t >= w0 && r.t < w0 + sim.setup.length)).length;
		if (status) status.textContent = `${(sim.t / 1000).toFixed(1)} s simulated · last 10 s: ${inside} spikes in the pattern, ${recent.length - inside} outside · fired in ${hit} of ${windows.length} patterns`;
	};

	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	for (let k = 0; k < span; k++) step();
	animate(canvas, (seconds) => {
		const steps = Math.min(4000, Math.round(seconds * 1000 * speed));
		for (let k = 0; k < steps; k++) step();
		draw();
	});
	return {
		setSpeed(value: number) {
			speed = value;
		},
		reset() {
			sim = new Pattern();
			recent.length = 0;
			windows = [];
			for (let k = 0; k < span; k++) step();
			draw();
		},
		setLearning(on: boolean) {
			sim.learning = on;
		},
	};
}
