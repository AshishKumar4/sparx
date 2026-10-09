// A live raster of a Brunel network stepped in a worker, with the excitatory population's rate below.
import { alpha, fit, onTheme, type Palette } from '../theme';
import type { Batch, Setup } from './worker';

export interface Settings {
	order: number;
	g: number;
	eta: number;
	/** Each excitatory synapse's jump, mV. */
	j: number;
}

export interface Raster {
	set(settings: Settings): void;
	/** The excitatory rate over the window shown (Hz), and the median CV and Fano factor since the first 100 ms. */
	stats(): { rate: number; cv: number; fano: number };
}

export function raster(canvas: HTMLCanvasElement, initial: Settings, { window = 400, rows = 200, speed = 120 } = {}): Raster {
	let excitatory = 4 * initial.order;
	const shownE = Math.round(rows * 0.8);
	const shownOf = () => [...Array.from({ length: shownE }, (_, k) => k), ...Array.from({ length: rows - shownE }, (_, k) => excitatory + k)];
	const dt = 0.1;
	const capacity = Math.round(window / dt);
	const times: number[] = [];
	const ids: number[] = [];
	const rate = new Float32Array(capacity);
	let now = 0;
	let cv = Number.NaN;
	let fano = Number.NaN;
	let colors: Palette;
	let view = fit(canvas);
	let dirty = true;
	new ResizeObserver(() => {
		view = fit(canvas);
		dirty = true;
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		dirty = true;
	});

	const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
	const start = (settings: Settings) => {
		excitatory = 4 * settings.order;
		times.length = 0;
		ids.length = 0;
		rate.fill(0);
		now = 0;
		cv = fano = Number.NaN;
		const setup: Setup = { ...settings, seed: 7, shown: shownOf(), speed };
		worker.postMessage(setup);
	};
	worker.onmessage = (event: MessageEvent<Batch>) => {
		const batch = event.data;
		let at = 0;
		const t0 = batch.time - batch.steps * dt;
		for (let s = 0; s < batch.steps; s++) {
			const t = t0 + (s + 1) * dt;
			const n = batch.spikes[at++];
			for (let k = 0; k < n; k++) {
				times.push(t);
				ids.push(batch.spikes[at++]);
			}
			const step = Math.round(t / dt);
			rate[step % capacity] = batch.excitatory[s] / excitatory / (dt / 1000);
		}
		now = batch.time;
		cv = batch.cv;
		fano = batch.fano;
		let drop = 0;
		while (drop < times.length && times[drop] < now - window) drop++;
		if (drop) {
			times.splice(0, drop);
			ids.splice(0, drop);
		}
		dirty = true;
	};
	start(initial);

	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	let visible = false;
	new IntersectionObserver(([entry]) => {
		visible = entry.isIntersecting;
		worker.postMessage(visible && !document.hidden ? 'resume' : 'pause');
	}).observe(canvas);
	document.addEventListener('visibilitychange', () => worker.postMessage(visible && !document.hidden ? 'resume' : 'pause'));
	if (reduced) setTimeout(() => worker.postMessage('pause'), window * 1000 / speed + 500);

	const draw = () => {
		if (dirty && colors) {
			dirty = false;
			const { ctx, width, height } = view;
			ctx.clearRect(0, 0, width, height);
			const band = Math.max(36, height * 0.2);
			const top = 6;
			const bottom = height - band - 10;
			const pitch = (bottom - top) / rows;
			const x = (t: number) => width - ((now - t) / window) * width;
			ctx.fillStyle = alpha(colors.ink, 0.035);
			ctx.fillRect(0, top + shownE * pitch, width, (rows - shownE) * pitch);
			const size = Math.max(1.4, Math.min(2.4, pitch * 0.9));
			for (let k = 0; k < times.length; k++) {
				ctx.fillStyle = ids[k] < shownE ? colors.spike : colors.learn;
				ctx.fillRect(x(times[k]) - size / 2, top + ids[k] * pitch, size, size);
			}
			const peak = 200;
			const base = height - 6;
			ctx.strokeStyle = colors.membrane;
			ctx.lineWidth = 1.25;
			ctx.beginPath();
			const bin = 5;
			let first = true;
			for (let c = capacity - 1; c >= 0; c -= bin) {
				const step = Math.round(now / dt) - (capacity - 1 - c);
				if (step < 0) continue;
				let sum = 0;
				for (let b = 0; b < bin; b++) sum += rate[(step - b + capacity * 4) % capacity];
				const value = sum / bin;
				const xx = width - ((capacity - 1 - c) / capacity) * width;
				const yy = base - Math.min(1, value / peak) * (band - 6);
				if (first) ctx.moveTo(xx, yy);
				else ctx.lineTo(xx, yy);
				first = false;
			}
			ctx.stroke();
		}
		requestAnimationFrame(draw);
	};
	requestAnimationFrame(draw);

	return {
		set: start,
		stats() {
			const filled = Math.min(capacity, Math.round(now / dt));
			let sum = 0;
			for (const r of rate) sum += r;
			return { rate: filled ? sum / filled : 0, cv, fano };
		},
	};
}
