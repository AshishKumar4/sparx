// The front page's racer: the network site/lab/racer.py trained (src/engines/racer.ts) driving a car around
// a track the visitor draws, seeing only its event camera's output, drawn beside what the camera sends and
// the network's spikes.
import { Racer, type RacerModel, Track } from '../../engines/racer';
import { alpha, animate, fit, onTheme, type Palette } from '../theme';

/** A closed track like those the racer trained on (site/lab/racer.py, `Tracks`): a radius of 6 to 10 m bent
 * by four harmonics, stretched, run either way round. */
export function randomTrack(points = 128, random: () => number = Math.random): number[][] {
	const amplitude = [2, 3, 4, 5].map((k) => (random() * 0.9) / k ** 1.5);
	const phase = [0, 1, 2, 3].map(() => random() * 2 * Math.PI);
	const radius = 6 + 4 * random();
	const stretch = 0.7 + 0.6 * random();
	const sign = random() < 0.5 ? 1 : -1;
	const rotate = random() * 2 * Math.PI;
	return Array.from({ length: points }, (_, i) => {
		const theta = (2 * Math.PI * i) / points;
		const r = 1 + amplitude.reduce((sum, a, j) => sum + a * Math.cos((j + 2) * theta + phase[j]), 0);
		const x = radius * stretch * r * Math.cos(theta);
		const y = (sign * radius * r * Math.sin(theta)) / stretch;
		return [Math.cos(rotate) * x - Math.sin(rotate) * y, Math.sin(rotate) * x + Math.cos(rotate) * y];
	});
}

/** A drawn loop, in screen pixels, as a track: closed, resampled to `points` at equal spacing, smoothed, and
 * scaled to the mean radius of the tracks it trained on, 8 m. Null for a stroke too short to be a loop. */
export function fromStroke(stroke: [number, number][], points = 128): number[][] | null {
	if (stroke.length < 8) return null;
	const closed = [...stroke, stroke[0]];
	const along = [0];
	for (let i = 1; i < closed.length; i++) along.push(along[i - 1] + Math.hypot(closed[i][0] - closed[i - 1][0], closed[i][1] - closed[i - 1][1]));
	const total = along.at(-1) as number;
	if (total < 200) return null;
	let at = 0;
	let track = Array.from({ length: points }, (_, k) => {
		const s = (total * k) / points;
		while (along[at + 1] < s) at++;
		const f = (s - along[at]) / Math.max(along[at + 1] - along[at], 1e-9);
		return [closed[at][0] + f * (closed[at + 1][0] - closed[at][0]), -(closed[at][1] + f * (closed[at + 1][1] - closed[at][1]))];
	});
	for (let pass = 0; pass < 3; pass++) {
		track = track.map((_, i) => {
			const sum = [0, 0];
			for (let d = -3; d <= 3; d++) {
				const p = track[(i + d + points) % points];
				sum[0] += p[0] / 7;
				sum[1] += p[1] / 7;
			}
			return sum;
		});
	}
	const cx = track.reduce((s, p) => s + p[0], 0) / points;
	const cy = track.reduce((s, p) => s + p[1], 0) / points;
	const mean = track.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / points;
	if (mean < 1e-6) return null;
	return track.map(([x, y]) => [((x - cx) * 8) / mean, ((y - cy) * 8) / mean]);
}

export interface RacerHero {
	newTrack(): void;
	draw(on: boolean): void;
	setPaused(value: boolean): void;
	readonly paused: boolean;
}

export function racerHero(root: HTMLElement, model: RacerModel, report: (state: { lap: number | null; laps: number; events: number; spikes: number; crashed: boolean }) => void): RacerHero {
	const canvas = root.querySelector<HTMLCanvasElement>('[data-racer-stage]') as HTMLCanvasElement;
	const netBox = root.querySelector<HTMLElement>('[data-racer-net]') as HTMLElement;
	// On the front page the track keeps clear of the copy and the controls; in a figure it fills what the
	// network's drawing leaves.
	const copy = root.querySelector<HTMLElement>('.copy');
	const panel = root.querySelector<HTMLElement>('[data-mode-panel="racer"]');
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	const desk = matchMedia('(min-width: 64rem)');
	const world = model.world;
	let points = randomTrack(world.points, seeded(11));
	let track = new Track(points, world);
	const racer = new Racer(model, track);
	const pixels = racer.pixels;
	const hidden = racer.network.neurons;
	const [first] = model.layers.flatMap((l) => (l.kind === 'dense' ? [l.outputs] : []));
	const seen = new Float32Array(2 * pixels);
	const flash = new Float32Array(hidden);
	let colors: Palette;
	let view = fit(canvas);
	let region = { x: 0, y: 0, w: 0, h: 0 };
	let net = { x: 0, y: 0, w: 0, h: 0 };
	let scale = 1;
	let origin = [0, 0];
	let paused = reduced;
	let drawing = false;
	let stroke: [number, number][] = [];
	let carry = 0;
	let covered = 0;
	let along = 0;
	let clock = 0;
	let started = 0;
	let laps = 0;
	let last: number | null = null;
	let crashedAt = -1;
	let events = 0;
	let spikes = 0;

	function start() {
		const d = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
		racer.track = track;
		racer.reset([points[0][0], points[0][1], Math.atan2(d[1], d[0]), 0]);
		track.locate(world, points[0][0], points[0][1], 0, racer.place);
		along = racer.place[1];
		covered = 0;
		started = clock;
		crashedAt = -1;
		seen.fill(0);
		flash.fill(0);
		fitTrack();
	}

	function layout() {
		const c = canvas.getBoundingClientRect();
		// Hidden on the drone's tab: keep the last layout until the canvas has a size again.
		if (!c.width || !c.height) return;
		view = fit(canvas);
		const box = (el: HTMLElement) => {
			const r = el.getBoundingClientRect();
			return { x: r.left - c.left, y: r.top - c.top, w: r.width, h: r.height };
		};
		net = box(netBox);
		if (!copy || !panel) {
			region = { x: 12, y: 12, w: view.width - 24, h: Math.max(120, net.y - 24) };
			fitTrack();
			return;
		}
		const words = box(copy);
		const controls = box(panel);
		if (desk.matches) {
			const left = Math.max(net.x + net.w, words.x + Math.min(words.w, 620)) + 32;
			region = { x: left, y: 24, w: view.width - left - 32, h: Math.max(160, controls.y - 40) };
		} else {
			const top = words.y + words.h + 12;
			region = { x: 16, y: top, w: view.width - 32, h: Math.max(120, net.y - top - 12) };
		}
		fitTrack();
	}

	function fitTrack() {
		let x0 = Infinity;
		let x1 = -Infinity;
		let y0 = Infinity;
		let y1 = -Infinity;
		for (const [x, y] of points) {
			x0 = Math.min(x0, x);
			x1 = Math.max(x1, x);
			y0 = Math.min(y0, y);
			y1 = Math.max(y1, y);
		}
		const margin = world.half_width + 0.6;
		scale = Math.min(region.w / (x1 - x0 + 2 * margin), region.h / (y1 - y0 + 2 * margin));
		origin = [region.x + region.w / 2 - ((x0 + x1) / 2) * scale, region.y + region.h / 2 + ((y0 + y1) / 2) * scale];
	}

	const sx = (x: number) => origin[0] + x * scale;
	const sy = (y: number) => origin[1] - y * scale;

	function step() {
		clock += world.dt;
		if (crashedAt >= 0) {
			if (clock - crashedAt > 1.2) start();
			return;
		}
		events = racer.step();
		spikes = 0;
		for (let i = 0; i < hidden; i++) {
			flash[i] = racer.network.spikes[i] ? 1 : flash[i] * 0.8;
			spikes += racer.network.spikes[i];
		}
		for (let p = 0; p < 2 * pixels; p++) seen[p] = racer.input[p] ? 1 : seen[p] * 0.75;
		let d = racer.place[1] - along;
		if (d < -track.total / 2) d += track.total;
		if (d > track.total / 2) d -= track.total;
		along = racer.place[1];
		covered += d;
		if (covered >= track.total) {
			covered -= track.total;
			laps++;
			last = clock - started;
			started = clock;
		}
		if (racer.place[0] > world.half_width + 1) crashedAt = clock;
	}

	function drawTrack(ctx: CanvasRenderingContext2D) {
		const loop = () => {
			ctx.beginPath();
			for (const [k, [x, y]] of points.entries()) {
				if (k) ctx.lineTo(sx(x), sy(y));
				else ctx.moveTo(sx(x), sy(y));
			}
			ctx.closePath();
		};
		ctx.lineJoin = 'round';
		loop();
		ctx.strokeStyle = alpha(colors.ink, 0.22);
		ctx.lineWidth = 2 * (world.half_width + 0.04) * scale;
		ctx.stroke();
		loop();
		ctx.strokeStyle = colors.page;
		ctx.lineWidth = 2 * (world.half_width - 0.04) * scale;
		ctx.stroke();
		loop();
		ctx.strokeStyle = alpha(colors.ink, 0.06);
		ctx.stroke();
		loop();
		ctx.setLineDash([0.75 * scale, 0.75 * scale]);
		ctx.strokeStyle = alpha(colors.ink, 0.35);
		ctx.lineWidth = Math.max(1, 0.08 * scale);
		ctx.stroke();
		ctx.setLineDash([]);
	}

	function drawCar(ctx: CanvasRenderingContext2D) {
		const [x, y, psi] = racer.car;
		const c = Math.cos(psi);
		const s = Math.sin(psi);
		for (let p = 0; p < pixels; p++) {
			const on = seen[p];
			const off = seen[pixels + p];
			const wx = x + racer.forward[p] * c - racer.side[p] * s;
			const wy = y + racer.forward[p] * s + racer.side[p] * c;
			ctx.fillStyle = on > 0.05 ? alpha(colors.spike, Math.min(1, on)) : off > 0.05 ? alpha(colors.membrane, Math.min(1, off)) : alpha(colors.ink, 0.12);
			ctx.fillRect(sx(wx) - 1.5, sy(wy) - 1.5, 3, 3);
		}
		ctx.save();
		ctx.translate(sx(x), sy(y));
		ctx.rotate(-psi);
		const length = Math.max(10, 0.7 * scale);
		const width = Math.max(6, 0.36 * scale);
		ctx.fillStyle = crashedAt >= 0 ? colors.faint : colors.spike;
		ctx.beginPath();
		ctx.roundRect(-length / 2, -width / 2, length, width, width / 3);
		ctx.fill();
		ctx.restore();
	}

	function drawNet(ctx: CanvasRenderingContext2D) {
		const gap = 18;
		const labelH = 16;
		const camW = Math.min(net.w * 0.42, (net.h - labelH) * 2);
		const cell = camW / world.columns;
		const camH = cell * world.rows;
		const top = net.y + labelH + Math.max(0, (net.h - labelH - camH) / 2);
		ctx.font = `500 10px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.fillText('what the camera sends', net.x, top - 6);
		for (let r = 0; r < world.rows; r++) {
			for (let col = 0; col < world.columns; col++) {
				const p = r * world.columns + col;
				const on = seen[p];
				const off = seen[pixels + p];
				// The far rows are the top of the picture, as a camera sees the road ahead.
				const yy = top + (world.rows - 1 - r) * cell;
				ctx.fillStyle = on > 0.05 ? alpha(colors.spike, Math.min(1, on)) : off > 0.05 ? alpha(colors.membrane, Math.min(1, off)) : alpha(colors.ink, 0.06);
				ctx.fillRect(net.x + col * cell + 0.5, yy + 0.5, cell - 1, cell - 1);
			}
		}
		const left = net.x + camW + gap;
		const room = net.x + net.w - left;
		const second = hidden - first;
		const columns = [12, 8];
		const sizes = [first, second];
		const unit = Math.min((room - gap - 40) / (columns[0] + columns[1]), camH / Math.ceil(first / columns[0]));
		let x = left;
		ctx.fillStyle = colors.muted;
		ctx.fillText(`${hidden} spiking neurons`, left, top - 6);
		let offset = 0;
		for (let layer = 0; layer < 2; layer++) {
			for (let k = 0; k < sizes[layer]; k++) {
				const f = flash[offset + k];
				const cx = x + (k % columns[layer]) * unit + unit / 2;
				const cy = top + Math.floor(k / columns[layer]) * unit + unit / 2;
				ctx.fillStyle = f > 0.05 ? alpha(colors.spike, 0.25 + 0.75 * f) : alpha(colors.ink, 0.1);
				ctx.beginPath();
				ctx.arc(cx, cy, Math.max(0.5, unit * 0.32), 0, 2 * Math.PI);
				ctx.fill();
			}
			offset += sizes[layer];
			x += columns[layer] * unit + gap / 2;
		}
		const bars = x + 6;
		const steer = Math.tanh(racer.network.readout[0]);
		const speed = racer.car[3] / world.max_speed;
		ctx.fillStyle = colors.muted;
		ctx.fillText('steer', bars, top + 8);
		ctx.fillText('speed', bars, top + camH / 2 + 8);
		const barW = Math.max(24, net.x + net.w - bars);
		ctx.fillStyle = alpha(colors.ink, 0.1);
		ctx.fillRect(bars, top + 14, barW, 4);
		ctx.fillRect(bars, top + camH / 2 + 14, barW, 4);
		ctx.fillStyle = colors.learn;
		ctx.fillRect(bars + barW / 2 + Math.min(0, steer) * (barW / 2), top + 14, Math.abs(steer) * (barW / 2), 4);
		ctx.fillRect(bars, top + camH / 2 + 14, Math.min(1, Math.max(0, speed)) * barW, 4);
	}

	function draw() {
		if (!colors || !view.width || net.w <= 0) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		drawTrack(ctx);
		drawCar(ctx);
		if (stroke.length > 1) {
			ctx.strokeStyle = colors.bio;
			ctx.lineWidth = 3;
			ctx.lineCap = 'round';
			ctx.beginPath();
			for (const [k, [x, y]] of stroke.entries()) {
				if (k) ctx.lineTo(x, y);
				else ctx.moveTo(x, y);
			}
			ctx.stroke();
		}
		drawNet(ctx);
		report({ lap: last, laps, events, spikes, crashed: crashedAt >= 0 });
	}

	const local = (event: PointerEvent): [number, number] => {
		const r = canvas.getBoundingClientRect();
		return [event.clientX - r.left, event.clientY - r.top];
	};
	const inRegion = ([x, y]: [number, number]) => x >= region.x && x <= region.x + region.w && y >= region.y && y <= region.y + region.h;
	canvas.addEventListener('pointerdown', (event) => {
		const at = local(event);
		if (!inRegion(at) || (event.pointerType !== 'mouse' && !drawing)) return;
		canvas.setPointerCapture(event.pointerId);
		stroke = [at];
	});
	canvas.addEventListener('pointermove', (event) => {
		if (!stroke.length) {
			if (event.pointerType === 'mouse') canvas.style.cursor = inRegion(local(event)) ? 'crosshair' : 'default';
			return;
		}
		const at = local(event);
		const end = stroke[stroke.length - 1];
		if (Math.hypot(at[0] - end[0], at[1] - end[1]) > 3) stroke.push(at);
		if (paused) draw();
	});
	const finish = () => {
		if (!stroke.length) return;
		const made = fromStroke(stroke);
		stroke = [];
		if (made) {
			points = made;
			track = new Track(points, world);
			laps = 0;
			last = null;
			start();
		}
		hero.draw(false);
		draw();
	};
	canvas.addEventListener('pointerup', finish);
	canvas.addEventListener('pointercancel', () => {
		stroke = [];
		draw();
	});

	// Fitting the canvas clears it, so every new layout is drawn at once: an off-screen figure would
	// otherwise stay blank until it next animates, and a paused one until it is played.
	const resize = () => {
		layout();
		draw();
	};
	layout();
	new ResizeObserver(resize).observe(canvas);
	desk.addEventListener('change', resize);
	onTheme((p) => {
		colors = p;
		draw();
	});
	start();
	// A second and a half of driving before the first paint, so the car is under way.
	for (let t = 0; t < 75; t++) step();
	draw();
	animate(canvas, (dt) => {
		if (paused) return;
		carry += dt;
		let n = 0;
		while (carry >= world.dt && n < 6) {
			step();
			carry -= world.dt;
			n++;
		}
		if (n === 6) carry = 0;
		draw();
	});

	const hero: RacerHero = {
		newTrack() {
			points = randomTrack(world.points);
			track = new Track(points, world);
			laps = 0;
			last = null;
			start();
			draw();
		},
		draw(on: boolean) {
			drawing = on;
			root.dataset.drawing = String(on);
			canvas.style.touchAction = on ? 'none' : '';
			draw();
		},
		setPaused(value: boolean) {
			paused = value;
			draw();
		},
		get paused() {
			return paused;
		},
	};
	return hero;
}

/** A racer driving inside `root`, bound to the controls and readouts there: buttons data-racer-act="track",
 * "draw" and "pause", and readouts data-racer-events, -spikes and -lap. `waiting` is the lap readout before a
 * lap is done. */
export function mountRacer(root: HTMLElement, model: RacerModel, waiting: (laps: number) => string): RacerHero {
	const field = (name: string) => root.querySelector<HTMLElement>(`[data-racer-${name}]`) as HTMLElement;
	const button = (act: string) => root.querySelector<HTMLButtonElement>(`[data-racer-act="${act}"]`) as HTMLButtonElement;
	const [draw, pause] = [button('draw'), button('pause')];
	const car = racerHero(root, model, (state) => {
		field('events').textContent = String(state.events);
		field('spikes').textContent = String(state.spikes);
		field('lap').textContent = state.crashed ? 'off the road: starting again' : state.lap === null ? waiting(state.laps) : `last lap ${state.lap.toFixed(1)} s`;
		draw.setAttribute('aria-pressed', String(root.dataset.drawing === 'true'));
	});
	const paused = (value: boolean) => {
		car.setPaused(value);
		root.dataset.racerPaused = String(value);
		pause.setAttribute('aria-pressed', String(value));
		pause.textContent = value ? 'Play' : 'Pause';
	};
	paused(car.paused);
	button('track').addEventListener('click', () => car.newTrack());
	draw.addEventListener('click', () => car.draw(root.dataset.drawing !== 'true'));
	pause.addEventListener('click', () => paused(!car.paused));
	return car;
}

/** Uniform numbers from a seed (mulberry32), so the first track is the same on every visit. */
function seeded(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
