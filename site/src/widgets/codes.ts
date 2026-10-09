// Chapter 4's figures: twelve values put into spikes by a rate code and a latency code, and read back;
// and a signal over time put into change events two ways (src/engines/encode.ts).
import { generator } from '../engines/brunel';
import { delta, latency, rate, sendOnDelta } from '../engines/encode';
import { alpha, fit, onTheme, type Palette } from './theme';

export function staticCodes(root: HTMLElement, read: () => { steps: number; jitter: number }) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const values = [0.95, 0.8, 0.62, 0.45, 0.3, 0.15, 0.05, 0.2, 0.5, 0.7, 0.9, 0.35];
	let seed = 1;
	let colors: Palette;
	let view = fit(canvas);
	let layout = { left: 0, bar: 0, rows: 0, row: 0, top: 0 };

	const draw = () => {
		if (!colors) return;
		const { steps, jitter } = read();
		const random = generator(seed);
		const rated = rate(values, steps, random);
		const timed = latency(values, steps).map((ts) => ts.map((t) => Math.min(Math.max(t + Math.round((2 * random() - 1) * jitter), 0), steps - 1)));
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const top = 26;
		const n = values.length;
		const row = (height - top - 8) / n;
		const label = 18;
		const bar = Math.min(80, width * 0.14);
		const decode = Math.min(60, width * 0.1);
		const raster = (width - label - bar - 2 * decode - 5 * 10) / 2;
		const x0 = label;
		const xRate = x0 + bar + 10;
		const xRateOut = xRate + raster + 10;
		const xLat = xRateOut + decode + 10;
		const xLatOut = xLat + raster + 10;
		layout = { left: x0, bar, rows: n, row, top };
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.textAlign = 'center';
		ctx.fillText('value', x0 + bar / 2, 14);
		ctx.fillText(`rate, ${steps} steps`, xRate + raster / 2, 14);
		ctx.fillText('read', xRateOut + decode / 2, 14);
		ctx.fillText(`latency, ${steps} steps`, xLat + raster / 2, 14);
		ctx.fillText('read', xLatOut + decode / 2, 14);
		let rateCount = 0;
		let latCount = 0;
		let rateErr = 0;
		let latErr = 0;
		const cell = raster / steps;
		for (let i = 0; i < n; i++) {
			const y = top + i * row;
			const h = Math.max(2, row - 3);
			ctx.fillStyle = alpha(colors.ink, 0.06);
			ctx.fillRect(x0, y, bar, h);
			ctx.fillRect(xRateOut, y, decode, h);
			ctx.fillRect(xLatOut, y, decode, h);
			ctx.fillStyle = colors.ink2;
			ctx.fillRect(x0, y, bar * values[i], h);
			ctx.fillStyle = alpha(colors.ink, 0.04);
			ctx.fillRect(xRate, y, raster, h);
			ctx.fillRect(xLat, y, raster, h);
			ctx.fillStyle = colors.spike;
			for (const t of rated[i]) ctx.fillRect(xRate + t * cell + 0.5, y, Math.max(1, cell - 1), h);
			for (const t of timed[i]) ctx.fillRect(xLat + t * cell + 0.5, y, Math.max(1, cell - 1), h);
			const fromRate = rated[i].length / steps;
			const fromLat = timed[i].length ? 1 - timed[i][0] / (steps - 1) : 0;
			ctx.fillStyle = colors.membrane;
			ctx.fillRect(xRateOut, y, decode * fromRate, h);
			ctx.fillRect(xLatOut, y, decode * fromLat, h);
			ctx.fillStyle = alpha(colors.ink, 0.5);
			ctx.fillRect(xRateOut + decode * values[i] - 0.5, y - 1, 1.5, h + 2);
			ctx.fillRect(xLatOut + decode * values[i] - 0.5, y - 1, 1.5, h + 2);
			rateCount += rated[i].length;
			latCount += timed[i].length;
			rateErr += Math.abs(fromRate - values[i]);
			latErr += Math.abs(fromLat - values[i]);
		}
		if (status)
			status.textContent = `rate: ${rateCount} spikes, read back within ${(rateErr / n).toFixed(3)} on average · latency: ${latCount} spikes, within ${(latErr / n).toFixed(3)}`;
	};

	const setValue = (event: PointerEvent) => {
		const r = canvas.getBoundingClientRect();
		const x = event.clientX - r.left;
		const y = event.clientY - r.top;
		const i = Math.floor((y - layout.top) / layout.row);
		if (i < 0 || i >= values.length || x < layout.left - 8 || x > layout.left + layout.bar + 8) return false;
		values[i] = Math.min(Math.max((x - layout.left) / layout.bar, 0), 1);
		draw();
		return true;
	};
	let dragging = false;
	canvas.addEventListener('pointerdown', (event) => {
		dragging = setValue(event);
		if (dragging) canvas.setPointerCapture(event.pointerId);
	});
	canvas.addEventListener('pointermove', (event) => dragging && setValue(event));
	canvas.addEventListener('pointerup', () => (dragging = false));
	root.querySelector('[data-act="resample"]')?.addEventListener('click', () => {
		seed++;
		draw();
	});
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

export function changeCodes(root: HTMLElement, read: () => { threshold: number }) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const N = 160;
	const signal = Array.from({ length: N }, (_, t) => {
		const ramp = t < 70 ? (t / 70) * 0.5 : 0.5;
		const bump = 0.35 * Math.exp(-(((t - 95) / 6) ** 2));
		const drop = t > 125 ? -0.3 : 0;
		return 0.2 + ramp + bump + drop;
	});
	let colors: Palette;
	let view = fit(canvas);
	let geometry = { left: 0, right: 0, top: 0, bottom: 0 };

	const draw = () => {
		if (!colors) return;
		const { threshold } = read();
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const left = 12;
		const right = width - 12;
		const top = 10;
		const bottom = height * 0.62;
		geometry = { left, right, top, bottom };
		const x = (t: number) => left + (t / (N - 1)) * (right - left);
		const y = (v: number) => bottom - v * (bottom - top);
		const sod = sendOnDelta(signal, threshold);
		const step = delta(signal, threshold);
		ctx.strokeStyle = alpha(colors.ink, 0.1);
		ctx.beginPath();
		ctx.moveTo(left, Math.round(y(0)) + 0.5);
		ctx.lineTo(right, Math.round(y(0)) + 0.5);
		ctx.stroke();
		ctx.strokeStyle = colors.ink2;
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let t = 0; t < N; t++) (t ? ctx.lineTo : ctx.moveTo).call(ctx, x(t), y(signal[t]));
		ctx.stroke();
		ctx.strokeStyle = colors.learn;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		for (let t = 0; t < N; t++) {
			if (t === 0) ctx.moveTo(x(t), y(sod.level[t]));
			else {
				ctx.lineTo(x(t), y(sod.level[t - 1]));
				ctx.lineTo(x(t), y(sod.level[t]));
			}
		}
		ctx.stroke();
		const lane = (bottom + 14);
		const laneH = (height - lane - 8) / 2;
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		const events = (values: number[], y0: number, label: string, color: string) => {
			ctx.fillStyle = colors.muted;
			ctx.fillText(label, left, y0 + 10);
			const mid = y0 + laneH / 2 + 4;
			ctx.strokeStyle = alpha(colors.ink, 0.1);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(left, mid);
			ctx.lineTo(right, mid);
			ctx.stroke();
			ctx.strokeStyle = color;
			ctx.lineWidth = 2;
			ctx.beginPath();
			for (let t = 0; t < N; t++) {
				if (!values[t]) continue;
				const h = Math.min(Math.abs(values[t]), 3) * (laneH / 7);
				ctx.moveTo(x(t), mid);
				ctx.lineTo(x(t), mid - Math.sign(values[t]) * h);
			}
			ctx.stroke();
		};
		events(sod.events, lane, 'send-on-delta', colors.learn);
		events(step, lane + laneH, 'DeltaEncoder (step to step)', colors.spike);
		const count = (v: number[]) => v.reduce((a, b) => a + Math.abs(b), 0);
		const err = signal.reduce((a, v, t) => Math.max(a, Math.abs(v - sod.level[t])), 0);
		if (status)
			status.textContent = `send-on-delta: ${count(sod.events)} events, the violet level never more than ${err.toFixed(2)} from the signal · step to step: ${count(step)} events · a rate code of the same signal: about ${Math.round(signal.reduce((a, b) => a + b, 0))} spikes`;
	};

	let drawing = false;
	let last = -1;
	const paint = (event: PointerEvent) => {
		const r = canvas.getBoundingClientRect();
		const px = event.clientX - r.left;
		const py = event.clientY - r.top;
		const t = Math.round(((px - geometry.left) / (geometry.right - geometry.left)) * (N - 1));
		if (t < 0 || t >= N || py > geometry.bottom + 10) return;
		const v = Math.min(Math.max((geometry.bottom - py) / (geometry.bottom - geometry.top), 0), 1);
		const from = last < 0 ? t : last;
		for (let k = Math.min(from, t); k <= Math.max(from, t); k++) signal[k] = v;
		last = t;
		draw();
	};
	canvas.addEventListener('pointerdown', (event) => {
		drawing = true;
		last = -1;
		canvas.setPointerCapture(event.pointerId);
		paint(event);
	});
	canvas.addEventListener('pointermove', (event) => drawing && paint(event));
	canvas.addEventListener('pointerup', () => (drawing = false));
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
