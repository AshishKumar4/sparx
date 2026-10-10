// A scrolling plot of several traces in stacked bands, each band on its own axis, with events marked along
// the top: what the biology chapter's figures draw, and the neuron scopes (scope.ts).
import { alpha, animate, fit, onTheme, type Palette } from './theme';

export type Role = 'spike' | 'membrane' | 'bio' | 'learn' | 'ink';

export interface Band {
	/** The band's share of the height. */
	weight: number;
	range: [number, number];
	lines: Role[];
	/** Dashed levels with their labels. */
	guides?: { at: number; label: string; role?: Role }[];
	/** Faint lines dividing the band into this many parts. */
	grid?: number;
	/** Shade each line's area down to zero, or the band's floor, instead of stroking it. */
	fill?: boolean;
}

export interface Sample {
	/** One value per line, in the order the bands list them. */
	values: number[];
	event?: boolean;
}

export function traces(
	canvas: HTMLCanvasElement,
	bands: Band[],
	step: () => Sample,
	{ steps = 400, perFrame = 1, events = 'ink' as Role } = {},
) {
	const width = bands.reduce((n, band) => n + band.lines.length, 0);
	const values = new Float64Array(steps * width);
	const events = new Uint8Array(steps);
	let head = 0;
	let colors: Palette;
	let view = fit(canvas);
	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

	function advance(n: number) {
		for (let k = 0; k < n; k++) {
			const sample = step();
			values.set(sample.values, (head % steps) * width);
			events[head % steps] = sample.event ? 1 : 0;
			head++;
		}
	}

	function draw() {
		if (!colors) return;
		const { ctx, width: w, height: h } = view;
		ctx.clearRect(0, 0, w, h);
		const marks = 12;
		const gap = 12;
		const total = bands.reduce((n, band) => n + band.weight, 0);
		const usable = h - marks - 6 - gap * (bands.length - 1) - 4;
		const first = Math.max(0, head - steps);
		const x = (t: number) => ((steps - (head - t)) / (steps - 1)) * w;

		ctx.strokeStyle = events === 'ink' ? alpha(colors.ink, 0.5) : colors[events];
		ctx.lineWidth = events === 'ink' ? 1.5 : 2;
		ctx.beginPath();
		for (let t = first; t < head; t++) {
			if (!events[t % steps]) continue;
			ctx.moveTo(Math.round(x(t)) + 0.5, 2);
			ctx.lineTo(Math.round(x(t)) + 0.5, marks);
		}
		ctx.stroke();

		let top = marks + 6;
		let column = 0;
		ctx.font = `10px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		for (const band of bands) {
			const height = (band.weight / total) * usable;
			const [lo, hi] = band.range;
			const y = (v: number) => top + height - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * height;
			if (band.grid) {
				ctx.strokeStyle = alpha(colors.ink, 0.07);
				ctx.lineWidth = 1;
				ctx.beginPath();
				for (let k = 1; k < band.grid; k++) {
					const yy = Math.round(top + (k * height) / band.grid) + 0.5;
					ctx.moveTo(0, yy);
					ctx.lineTo(w, yy);
				}
				ctx.stroke();
			}
			for (const guide of band.guides ?? []) {
				const yy = Math.round(y(guide.at)) + 0.5;
				ctx.setLineDash([4, 4]);
				ctx.strokeStyle = alpha(colors[guide.role ?? 'ink'], 0.35);
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(0, yy);
				ctx.lineTo(w, yy);
				ctx.stroke();
				ctx.setLineDash([]);
				ctx.fillStyle = colors.muted;
				if (guide.label) ctx.fillText(guide.label, 6, yy - 4);
			}
			const floor = y(Math.min(Math.max(0, lo), hi));
			for (const role of band.lines) {
				ctx.lineWidth = 1.75;
				ctx.lineJoin = 'round';
				ctx.beginPath();
				if (band.fill) ctx.moveTo(x(first), floor);
				for (let t = first; t < head; t++) {
					const yy = y(values[(t % steps) * width + column]);
					if (t === first && !band.fill) ctx.moveTo(x(t), yy);
					else ctx.lineTo(x(t), yy);
				}
				if (band.fill) {
					ctx.lineTo(x(head - 1), floor);
					ctx.fillStyle = alpha(colors[role], 0.1);
					ctx.fill();
				} else {
					ctx.strokeStyle = colors[role];
					ctx.stroke();
				}
				column++;
			}
			top += height + gap;
		}
	}

	// The window starts full, so the figure reads as running from its first frame.
	advance(steps);
	draw();
	animate(canvas, () => {
		if (reduced) return;
		advance(perFrame);
		draw();
	});
	return {
		redraw() {
			if (reduced) advance(steps);
			draw();
		},
	};
}
