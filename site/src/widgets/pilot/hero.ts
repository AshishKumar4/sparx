// The front page's hero: the trained pilot (src/engines/pilot.ts) flying its drone after the
// visitor's pointer, with its spikes on a raster beside it.
import { Pilot, type PilotModel } from '../../engines/pilot';
import { alpha, animate, fit, onTheme, type Palette } from '../theme';

const HEIGHT = 1.5;
const WIDTHS = [0.9, 4.0];
const HISTORY = 240;

interface Spark {
	x: number;
	y: number;
	vx: number;
	vy: number;
	life: number;
	kind: 0 | 1;
}

export function hero(root: HTMLElement, model: PilotModel): void {
	const world = root.querySelector<HTMLCanvasElement>('[data-world]') as HTMLCanvasElement;
	const raster = root.querySelector<HTMLCanvasElement>('[data-raster]') as HTMLCanvasElement;
	const rate = root.querySelector<HTMLElement>('[data-rate]');
	const silencedCount = root.querySelector<HTMLElement>('[data-silenced]');
	const pauseButton = root.querySelector<HTMLButtonElement>('[data-act="pause"]');
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	const pilot = new Pilot(model);
	const neurons = pilot.neurons;
	const dt = pilot.drone.dt;

	let colors: Palette;
	let view = fit(world);
	let panel = fit(raster);
	let scale = 1;
	let box = [2, HEIGHT];
	let cx = 0;
	let cy = 0;
	const layout = () => {
		view = fit(world);
		panel = fit(raster);
		scale = view.height / (2 * HEIGHT);
		box = [Math.min(Math.max(view.width / (2 * scale), WIDTHS[0]), WIDTHS[1]), HEIGHT];
		cx = view.width / 2;
		cy = view.height / 2;
	};
	layout();
	new ResizeObserver(layout).observe(world);
	onTheme((p) => (colors = p));

	// On a wide screen the copy sits on the left and the panel at the bottom right, so the drone
	// idles in the space between, above the panel.
	const desk = matchMedia('(min-width: 64rem)');
	const home = () => (desk.matches ? [0.32 * box[0], 0.3] : [0, 0]);
	pilot.state.set([home()[0], home()[1], 0, 0, 0, 0]);
	let target = home();
	let pointer: [number, number] | null = null;
	let pointerAt = -1e9;
	let grabbed = false;
	let fling: [number, number] = [0, 0];
	let paused = reduced;
	let clock = 0;
	let carry = 0;

	const spikes = new Uint8Array(HISTORY * neurons);
	const readouts = new Float32Array(HISTORY * 2);
	let head = 0;
	let recent = 0;
	const trail: number[] = [];
	const sparks: Spark[] = [];

	const toWorld = (event: PointerEvent): [number, number] => {
		const r = world.getBoundingClientRect();
		return [(event.clientX - r.left - cx) / scale, -(event.clientY - r.top - cy) / scale];
	};
	const clampTarget = (x: number, y: number) => [
		Math.min(Math.max(x, -0.88 * box[0]), 0.88 * box[0]),
		Math.min(Math.max(y, -0.85 * box[1]), 0.85 * box[1]),
	];

	const kick = (vx: number, vy: number, spin: number) => {
		pilot.state[2] += vx;
		pilot.state[3] += vy;
		pilot.state[5] += spin;
	};
	const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

	const panelBox = root.querySelector('.pilot');
	root.addEventListener('pointermove', (event) => {
		if (event.pointerType !== 'mouse' && !grabbed) return;
		if (!grabbed && panelBox?.contains(event.target as Node)) {
			pointer = null;
			return;
		}
		const [x, y] = toWorld(event);
		if (grabbed) {
			fling = [0.6 * fling[0] + 0.4 * ((x - pilot.state[0]) / Math.max(dt, 1 / 120)), 0.6 * fling[1] + 0.4 * ((y - pilot.state[1]) / Math.max(dt, 1 / 120))];
			pilot.state[0] = Math.min(Math.max(x, -box[0]), box[0]);
			pilot.state[1] = Math.min(Math.max(y, -box[1]), box[1]);
		}
		pointer = [x, y];
		pointerAt = clock;
	});
	root.addEventListener('pointerleave', () => (pointer = null));
	world.addEventListener('pointerdown', (event) => {
		const [x, y] = toWorld(event);
		const near = Math.hypot(x - pilot.state[0], y - pilot.state[1]) < 0.4;
		if (event.pointerType === 'mouse' && near) {
			grabbed = true;
			fling = [0, 0];
			world.setPointerCapture(event.pointerId);
			world.classList.add('grabbing');
		} else if (event.pointerType === 'mouse') {
			const dx = pilot.state[0] - x;
			const dy = pilot.state[1] - y;
			const d = Math.max(Math.hypot(dx, dy), 0.3);
			const push = 4.5 * Math.exp(-d / 1.6);
			kick((push * dx) / d, (push * dy) / d, (Math.random() < 0.5 ? -1 : 1) * 10 * push);
			burst(x, y);
		}
		if (event.pointerType !== 'mouse') {
			pointer = [x, y];
			pointerAt = clock + 1e6;
		}
		if (paused && !reduced) setPaused(false);
	});
	const release = () => {
		if (!grabbed) return;
		grabbed = false;
		world.classList.remove('grabbing');
		const speed = Math.hypot(...fling);
		const cap = speed > 9 ? 9 / speed : 1;
		pilot.state[2] = fling[0] * cap;
		pilot.state[3] = fling[1] * cap;
	};
	world.addEventListener('pointerup', release);
	world.addEventListener('pointercancel', release);

	const shown = () => {
		const n = pilot.silenced.reduce((a, b) => a + b, 0);
		if (silencedCount) silencedCount.textContent = n ? `${n} silenced` : '';
		root.dataset.silenced = String(n);
	};
	const act = (name: string) => {
		if (name === 'gust') kick(2.2 * gauss(), 2.2 * gauss(), 9 * gauss());
		else if (name === 'flip') kick(0, 1.5, (Math.random() < 0.5 ? -1 : 1) * 26);
		else if (name === 'silence') {
			const alive = [...pilot.silenced.keys()].filter((i) => !pilot.silenced[i]);
			for (let k = 0; k < 16 && alive.length; k++) pilot.silenced[alive.splice(Math.floor(Math.random() * alive.length), 1)[0]] = 1;
			shown();
		} else if (name === 'restore') {
			pilot.silenced.fill(0);
			shown();
		} else if (name === 'pause') setPaused(!paused);
		if (name !== 'pause' && paused && !reduced) setPaused(false);
	};
	for (const button of root.querySelectorAll<HTMLButtonElement>('[data-act]')) button.addEventListener('click', () => act(button.dataset.act as string));

	function setPaused(value: boolean) {
		paused = value;
		root.dataset.paused = String(value);
		if (pauseButton) {
			pauseButton.setAttribute('aria-pressed', String(value));
			pauseButton.textContent = value ? 'Play' : 'Pause';
		}
	}
	setPaused(paused);

	raster.addEventListener('click', (event) => {
		const r = raster.getBoundingClientRect();
		const row = rowAt(event.clientY - r.top);
		if (row === null) return;
		pilot.silenced[row] ^= 1;
		shown();
	});
	raster.addEventListener('pointermove', (event) => {
		const row = rowAt(event.clientY - raster.getBoundingClientRect().top);
		raster.title = row === null ? '' : `Neuron ${(row % 64) + 1} of layer ${row < 64 ? 1 : 2}: click to ${pilot.silenced[row] ? 'restore' : 'silence'} it`;
	});

	const rows = () => {
		const top = 6;
		const gap = 6;
		const trace = Math.max(26, panel.height * 0.18);
		const pitch = (panel.height - top - 2 * gap - trace - 4) / neurons;
		return { top, gap, trace, pitch };
	};
	function rowAt(y: number): number | null {
		const { top, gap, pitch } = rows();
		const half = neurons / 2;
		let row = Math.floor((y - top) / pitch);
		if (row >= half) row = Math.floor((y - top - gap) / pitch);
		return row >= 0 && row < neurons ? row : null;
	}

	function burst(x: number, y: number) {
		for (let k = 0; k < 14; k++) {
			const a = Math.random() * 2 * Math.PI;
			const v = 1 + 2 * Math.random();
			sparks.push({ x, y, vx: v * Math.cos(a), vy: v * Math.sin(a), life: 0.5, kind: 1 });
		}
	}

	function step() {
		const idle = pointer === null && clock - pointerAt > 1.2;
		if (!idle && pointer) target = clampTarget(pointer[0], pointer[1]);
		else if (idle) {
			const t = clock;
			const [hx, hy] = home();
			const reach = desk.matches ? [0.22 * box[0], 0.42] : [0.6 * box[0], 0.8];
			target = clampTarget(hx + reach[0] * Math.sin(0.23 * t), hy + reach[1] * Math.sin(0.41 * t + 0.7));
		}
		if (grabbed) {
			pilot.think(pilot.observe(target[0], target[1]));
			pilot.state[2] = fling[0];
			pilot.state[3] = fling[1];
		} else {
			pilot.step(target[0], target[1], box[0], box[1]);
		}
		const s = pilot.state;
		const row = (head % HISTORY) * neurons;
		let fired = 0;
		for (let i = 0; i < neurons; i++) {
			spikes[row + i] = pilot.spikes[i];
			fired += pilot.spikes[i];
		}
		readouts[(head % HISTORY) * 2] = pilot.readout[0];
		readouts[(head % HISTORY) * 2 + 1] = pilot.readout[1];
		head++;
		recent = 0.97 * recent + 0.03 * fired;
		const sin = Math.sin(s[4]);
		const cos = Math.cos(s[4]);
		for (let k = 0; k < 2; k++) {
			const side = k ? 1 : -1;
			if (Math.random() < pilot.thrust[k] / pilot.drone.max_thrust) {
				const bx = s[0] + side * pilot.drone.arm * cos;
				const by = s[1] + side * pilot.drone.arm * sin;
				const v = 1.5 + 2.5 * (pilot.thrust[k] / pilot.drone.max_thrust);
				sparks.push({ x: bx, y: by, vx: s[2] + v * sin + 0.3 * gauss(), vy: s[3] - v * cos + 0.3 * gauss(), life: 0.35, kind: 0 });
			}
		}
		for (let k = 0; k < fired; k += 6) {
			const a = Math.random() * 2 * Math.PI;
			sparks.push({ x: s[0], y: s[1], vx: s[2] + 1.4 * Math.cos(a), vy: s[3] + 1.4 * Math.sin(a), life: 0.45, kind: 1 });
		}
		trail.push(s[0], s[1]);
		if (trail.length > 160) trail.splice(0, 2);
		clock += dt;
	}

	function drawWorld() {
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const px = (x: number) => cx + x * scale;
		const py = (y: number) => cy - y * scale;

		ctx.strokeStyle = alpha(colors.ink, colors.dark ? 0.045 : 0.06);
		ctx.lineWidth = 1;
		ctx.beginPath();
		for (let x = Math.ceil(-box[0] * 4) / 4; x <= box[0]; x += 0.25) {
			ctx.moveTo(Math.round(px(x)) + 0.5, 0);
			ctx.lineTo(Math.round(px(x)) + 0.5, height);
		}
		for (let y = -HEIGHT; y <= HEIGHT; y += 0.25) {
			ctx.moveTo(0, Math.round(py(y)) + 0.5);
			ctx.lineTo(width, Math.round(py(y)) + 0.5);
		}
		ctx.stroke();

		const [tx, ty] = target;
		const pulse = 0.5 + 0.5 * Math.sin(clock * 4);
		ctx.strokeStyle = alpha(colors.spike, 0.75);
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.arc(px(tx), py(ty), 9 + 3 * pulse, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
			ctx.moveTo(px(tx) + dx * 15, py(ty) + dy * 15);
			ctx.lineTo(px(tx) + dx * 21, py(ty) + dy * 21);
		}
		ctx.stroke();

		const s = pilot.state;
		ctx.setLineDash([3, 5]);
		ctx.strokeStyle = alpha(colors.ink, 0.16);
		ctx.beginPath();
		ctx.moveTo(px(s[0]), py(s[1]));
		ctx.lineTo(px(tx), py(ty));
		ctx.stroke();
		ctx.setLineDash([]);

		for (let k = 2; k < trail.length; k += 2) {
			ctx.strokeStyle = alpha(colors.membrane, (0.35 * k) / trail.length);
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(px(trail[k - 2]), py(trail[k - 1]));
			ctx.lineTo(px(trail[k]), py(trail[k + 1]));
			ctx.stroke();
		}

		for (const p of sparks) {
			const life = p.kind ? p.life / 0.45 : p.life / 0.35;
			ctx.fillStyle = alpha(p.kind ? colors.spike : colors.membrane, Math.max(0, life) * (p.kind ? 0.95 : 0.4));
			const r = p.kind ? 1.6 : 2.2;
			ctx.fillRect(px(p.x) - r, py(p.y) - r, 2 * r, 2 * r);
		}

		ctx.save();
		ctx.translate(px(s[0]), py(s[1]));
		ctx.rotate(-s[4]);
		const arm = pilot.drone.arm * scale;
		const unit = scale / 100;
		for (let k = 0; k < 2; k++) {
			const side = k ? 1 : -1;
			const power = pilot.thrust[k] / pilot.drone.max_thrust;
			const plume = ctx.createLinearGradient(0, 4 * unit, 0, (10 + 34 * power) * unit);
			plume.addColorStop(0, alpha(colors.membrane, 0.45 * power + 0.1));
			plume.addColorStop(1, alpha(colors.membrane, 0));
			ctx.fillStyle = plume;
			ctx.beginPath();
			ctx.moveTo(side * arm - 6 * unit, 3 * unit);
			ctx.lineTo(side * arm + 6 * unit, 3 * unit);
			ctx.lineTo(side * arm + 2 * unit, (10 + 34 * power) * unit);
			ctx.lineTo(side * arm - 2 * unit, (10 + 34 * power) * unit);
			ctx.fill();
		}
		ctx.strokeStyle = colors.ink;
		ctx.lineCap = 'round';
		ctx.lineWidth = Math.max(2, 3.2 * unit);
		ctx.beginPath();
		ctx.moveTo(-arm, 0);
		ctx.lineTo(arm, 0);
		ctx.stroke();
		ctx.lineWidth = Math.max(1.5, 2.2 * unit);
		ctx.beginPath();
		for (const side of [-1, 1]) {
			ctx.moveTo(side * arm, 0);
			ctx.lineTo(side * arm, -5 * unit);
			ctx.moveTo(side * 5 * unit, 4 * unit);
			ctx.lineTo(side * 8 * unit, 10 * unit);
		}
		ctx.stroke();
		for (let k = 0; k < 2; k++) {
			const side = k ? 1 : -1;
			const power = pilot.thrust[k] / pilot.drone.max_thrust;
			ctx.fillStyle = alpha(colors.ink, 0.25 + 0.5 * power);
			ctx.beginPath();
			ctx.ellipse(side * arm, -6 * unit, 11 * unit, 1.8 * unit + 0.6, 0, 0, 2 * Math.PI);
			ctx.fill();
		}
		ctx.fillStyle = colors.ink;
		ctx.beginPath();
		ctx.roundRect(-8 * unit, -4 * unit, 16 * unit, 8 * unit, 3 * unit);
		ctx.fill();
		const glow = Math.min(1, recent / 18);
		ctx.fillStyle = alpha(colors.spike, 0.35 + 0.65 * glow);
		ctx.shadowColor = colors.spike;
		ctx.shadowBlur = 12 * glow;
		ctx.beginPath();
		ctx.arc(0, 0, 2.2 * unit + 1, 0, 2 * Math.PI);
		ctx.fill();
		ctx.restore();
	}

	function drawRaster() {
		const { ctx, width, height } = panel;
		ctx.clearRect(0, 0, width, height);
		const { top, gap, trace, pitch } = rows();
		const columns = Math.min(HISTORY, Math.floor(width / 2));
		const col = width / columns;
		const half = neurons / 2;
		const y = (i: number) => top + i * pitch + (i >= half ? gap : 0);
		ctx.fillStyle = alpha(colors.ink, 0.06);
		for (let i = 0; i < neurons; i++) {
			if (pilot.silenced[i]) ctx.fillRect(0, y(i), width, Math.max(1, pitch));
		}
		ctx.fillStyle = colors.spike;
		const size = Math.max(1, pitch * 0.82);
		for (let c = 0; c < columns; c++) {
			const t = head - columns + c;
			if (t < 0) continue;
			const row = (t % HISTORY) * neurons;
			const x = c * col;
			for (let i = 0; i < neurons; i++) if (spikes[row + i]) ctx.fillRect(x, y(i), Math.max(1.2, col * 0.8), size);
		}
		ctx.font = `500 10px ${getComputedStyle(raster).getPropertyValue('--sx-mono') || 'monospace'}`;
		ctx.textBaseline = 'top';
		const labels: [string, number][] = [['layer 1 · 64 LIF', y(0)], ['layer 2 · 64 LIF', y(half)], ['rotors · 2 LI', top + neurons * pitch + gap * 2]];
		for (const [text, at] of labels) {
			const w = ctx.measureText(text).width + 8;
			ctx.fillStyle = alpha(colors.page, 0.85);
			ctx.fillRect(0, at, w, 13);
			ctx.fillStyle = colors.muted;
			ctx.fillText(text, 4, at + 2);
		}
		const base = top + neurons * pitch + gap * 2 + trace / 2;
		ctx.strokeStyle = alpha(colors.ink, 0.12);
		ctx.beginPath();
		ctx.moveTo(0, base);
		ctx.lineTo(width, base);
		ctx.stroke();
		for (let k = 0; k < 2; k++) {
			ctx.strokeStyle = k ? colors.membrane : alpha(colors.membrane, 0.55);
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			for (let c = 0; c < columns; c++) {
				const t = head - columns + c;
				if (t < 0) continue;
				const v = readouts[(t % HISTORY) * 2 + k];
				const yy = base - Math.tanh(v / 3) * (trace / 2 - 2);
				if (c === 0 || t === 0) ctx.moveTo(c * col, yy);
				else ctx.lineTo(c * col, yy);
			}
			ctx.stroke();
		}
	}

	const frame = (seconds: number) => {
		if (!paused) {
			carry += seconds;
			let n = 0;
			while (carry >= dt && n < 6) {
				step();
				carry -= dt;
				n++;
			}
			if (n === 6) carry = 0;
			for (const p of sparks) {
				p.x += p.vx * seconds;
				p.y += p.vy * seconds;
				p.vy -= (p.kind ? 2 : 0) * seconds;
				p.life -= seconds;
			}
			for (let k = sparks.length - 1; k >= 0; k--) if (sparks[k].life <= 0) sparks.splice(k, 1);
			if (rate) rate.textContent = `${Math.round(recent / neurons / dt)} Hz`;
		}
		drawWorld();
		drawRaster();
	};

	if (reduced) {
		for (let k = 0; k < 150; k++) step();
		frame(0);
	}
	const loop = animate(root, frame);
	onTheme(() => !loop.running() && frame(0));
	shown();
}
