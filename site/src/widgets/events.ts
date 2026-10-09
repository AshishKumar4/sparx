// A fan turning in front of two cameras: one that sends every pixel's brightness each frame, and an event
// camera (src/engines/events.ts) that sends a pixel only when its log brightness changes.
import { EventCamera } from '../engines/events';
import { animate, onTheme, type Palette } from './theme';

export const WIDTH = 72;
export const HEIGHT = 40;
const pixels = WIDTH * HEIGHT;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const rgb = (hex: string) => {
	const n = Number.parseInt(hex.slice(1), 16);
	return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

/** The log brightness each pixel sees: a three-bladed fan turned `turn` radians, in front of a still, striped
 * wall. */
export function scene(turn: number, out: Float64Array): Float64Array {
	const cx = WIDTH / 2;
	const cy = HEIGHT / 2;
	for (let j = 0; j < HEIGHT; j++) {
		for (let i = 0; i < WIDTH; i++) {
			const x = i + 0.5 - cx;
			const y = j + 0.5 - cy;
			const r = Math.hypot(x, y);
			const sector = (2 * Math.PI) / 3;
			let a = (Math.atan2(y, x) - turn) % sector;
			if (a < 0) a += sector;
			const across = Math.abs(a - sector / 2) * r;
			const blade = sigmoid((0.32 * r - across) / 0.5) * sigmoid((17 - r) / 0.5) * sigmoid((r - 2) / 0.5);
			const wall = 0.16 + 0.08 * (0.5 + 0.5 * Math.sin(i * 0.9)) + 0.04 * (j / HEIGHT);
			out[j * WIDTH + i] = Math.log(wall * (1 - blade) + 0.85 * blade);
		}
	}
	return out;
}

export interface Settings {
	speed: number;
	threshold: number;
}

export function eventFigure(canvas: HTMLCanvasElement, settings: () => Settings, report: (perSecond: number) => void) {
	const camera = new EventCamera(pixels);
	const seen = new Float64Array(pixels);
	const age = new Float64Array(pixels).fill(Infinity);
	const sign = new Int8Array(pixels);
	const counts = new Uint32Array(1000);
	let ms = 0;
	let turn = 0;
	camera.reset(scene(0, seen));
	const frame = document.createElement('canvas');
	frame.width = WIDTH;
	frame.height = HEIGHT;
	const events = document.createElement('canvas');
	events.width = WIDTH;
	events.height = HEIGHT;
	let colors: Palette;
	onTheme((p) => {
		colors = p;
		draw();
	});
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

	function advance(steps: number) {
		const { speed, threshold } = settings();
		camera.threshold = threshold;
		for (let k = 0; k < steps; k++) {
			turn += 2 * Math.PI * speed * 0.001;
			const sent = camera.step(scene(turn, seen));
			counts[ms % 1000] = sent;
			ms++;
			for (let p = 0; p < pixels; p++) {
				age[p] += 1;
				if (camera.on[p] || camera.off[p]) {
					age[p] = 0;
					sign[p] = camera.on[p] ? 1 : -1;
				}
			}
		}
		let total = 0;
		for (let k = 0; k < Math.min(ms, 1000); k++) total += counts[k];
		report(ms >= 1000 ? total : (total * 1000) / Math.max(ms, 1));
	}

	function paint(target: HTMLCanvasElement, fill: (p: number, rgba: Uint8ClampedArray, at: number) => void) {
		const ctx = target.getContext('2d') as CanvasRenderingContext2D;
		const image = ctx.createImageData(WIDTH, HEIGHT);
		for (let p = 0; p < pixels; p++) fill(p, image.data, 4 * p);
		ctx.putImageData(image, 0, 0);
	}

	function draw() {
		if (!colors) return;
		const page = rgb(colors.page);
		const on = rgb(colors.spike);
		const off = rgb(colors.membrane);
		paint(frame, (p, d, at) => {
			const v = Math.round(255 * Math.min(Math.exp(seen[p]), 1));
			d.set([v, v, v, 255], at);
		});
		paint(events, (p, d, at) => {
			const w = Math.exp(-age[p] / 25);
			const c = sign[p] > 0 ? on : off;
			d.set([0, 1, 2].map((k) => Math.round(page[k] + (c[k] - page[k]) * w)).concat(255), at);
		});
		const ratio = Math.min(devicePixelRatio || 1, 2);
		const { width, height } = canvas.getBoundingClientRect();
		canvas.width = Math.max(1, Math.round(width * ratio));
		canvas.height = Math.max(1, Math.round(height * ratio));
		const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
		ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		ctx.imageSmoothingEnabled = false;
		const gap = 12;
		const label = 22;
		// Side by side where they fit at a useful size, else one above the other.
		const stacked = width < 520;
		const scale = stacked
			? Math.min((width - 24) / WIDTH, (height - 2 * label - gap - 12) / (2 * HEIGHT))
			: Math.min((width - gap - 24) / (2 * WIDTH), (height - label - 12) / HEIGHT);
		const w = WIDTH * scale;
		const h = HEIGHT * scale;
		const left = stacked ? (width - w) / 2 : (width - 2 * w - gap) / 2;
		const top = stacked ? label + (height - 2 * (h + label) - gap) / 2 + 6 : label + (height - label - h) / 2;
		const second = stacked ? { x: left, y: top + h + gap + label } : { x: left + w + gap, y: top };
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.fillText('every pixel, every frame', left, top - 8);
		ctx.fillText('events: ON and OFF', second.x, second.y - 8);
		ctx.drawImage(frame, left, top, w, h);
		ctx.drawImage(events, second.x, second.y, w, h);
		ctx.strokeStyle = colors.line;
		ctx.strokeRect(second.x + 0.5, second.y + 0.5, w - 1, h - 1);
	}

	new ResizeObserver(() => draw()).observe(canvas);
	// 300 ms before the first paint, so the figure shows events before it moves.
	advance(300);
	draw();
	animate(canvas, (dt) => {
		if (reduced) return;
		advance(Math.max(1, Math.round(dt * 1000)));
		draw();
	});
	return {
		redraw() {
			if (reduced) advance(300);
			draw();
		},
	};
}
