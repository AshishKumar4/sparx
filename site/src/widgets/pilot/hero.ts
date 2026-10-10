// The front page's hero: the trained pilot (src/engines/pilot.ts) flying its drone after the
// visitor's pointer, drawn beside the network itself: its readings, its 128 LIF neurons lighting as
// they spike, and the two readout neurons wired to the rotors they drive.
import { INPUTS, Pilot, type PilotModel } from '../../engines/pilot';
import { alpha, animate, fit, onTheme, type Palette } from '../theme';

const HEIGHT = 1.5;
const SHORT = ['Δx', 'Δy', 'vx', 'vy', 'sin', 'cos', 'ω'];
const WIDTHS = [0.9, 4.0];

interface Spark {
	x: number;
	y: number;
	vx: number;
	vy: number;
	life: number;
	kind: 0 | 1;
}

interface Node {
	x: number;
	y: number;
}

interface Edge {
	from: number;
	to: number;
	strength: number;
}

const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

/** The strongest `keep` connections out of each of `inputs` units, by absolute weight. */
function strongest(kernel: Float64Array, inputs: number, outputs: number, keep: number, from: number, to: number): Edge[] {
	let largest = 0;
	for (const w of kernel) largest = Math.max(largest, Math.abs(w));
	const edges: Edge[] = [];
	for (let i = 0; i < inputs; i++) {
		const order = Array.from({ length: outputs }, (_, j) => j).sort((a, b) => Math.abs(kernel[i * outputs + b]) - Math.abs(kernel[i * outputs + a]));
		for (const j of order.slice(0, keep)) edges.push({ from: from + i, to: to + j, strength: kernel[i * outputs + j] / largest });
	}
	return edges;
}

export function hero(root: HTMLElement, model: PilotModel): void {
	const canvas = root.querySelector<HTMLCanvasElement>('[data-stage]') as HTMLCanvasElement;
	const netBox = root.querySelector<HTMLElement>('[data-net]') as HTMLElement;
	const rate = root.querySelector<HTMLElement>('[data-rate]');
	const silencedCount = root.querySelector<HTMLElement>('[data-silenced]');
	const pauseButton = root.querySelector<HTMLButtonElement>('[data-act="pause"]');
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	const desk = matchMedia('(min-width: 64rem)');
	const pilot = new Pilot(model);
	const hidden = pilot.neurons;
	const half = hidden / 2;
	const dt = pilot.drone.dt;
	const kernels = pilot.network.kernels();

	// Nodes: 7 readings, 128 hidden neurons, 2 readouts; edges: the strongest of each layer.
	const I = INPUTS.length;
	const nodes: Node[] = Array.from({ length: I + hidden + 2 }, () => ({ x: 0, y: 0 }));
	const H0 = I;
	const O0 = I + hidden;
	const edges = [
		...strongest(kernels[0].kernel, I, half, 5, 0, H0),
		...strongest(kernels[1].kernel, half, half, 3, H0, H0 + half),
		...strongest(kernels[2].kernel, half, 2, 2, H0 + half, O0),
	];
	const flash = new Float32Array(nodes.length);

	let colors: Palette;
	let view = fit(canvas);
	let scale = 1;
	let box = [2, HEIGHT];
	let origin = [0, 0];
	let net = { x: 0, y: 0, w: 0, h: 0 };
	let cell = 10;
	let narrow = false;

	const layout = () => {
		view = fit(canvas);
		const c = canvas.getBoundingClientRect();
		const n = netBox.getBoundingClientRect();
		net = { x: n.left - c.left, y: n.top - c.top, w: n.width, h: n.height };
		const worldHeight = desk.matches ? view.height : Math.max(120, net.y - 12);
		scale = worldHeight / (2 * HEIGHT);
		box = [Math.min(Math.max(view.width / (2 * scale), WIDTHS[0]), WIDTHS[1]), HEIGHT];
		origin = [view.width / 2, worldHeight / 2];
		narrow = net.w < 480;
		const labelW = narrow ? 54 : Math.min(86, net.w * 0.2);
		const outW = Math.min(64, net.w * 0.14);
		const grid = Math.min((net.w - labelW - outW - 3 * 18) / 2, net.h - 30);
		cell = grid / 8;
		const top = net.y + 24 + (net.h - 24 - grid) / 2;
		const left1 = net.x + labelW + 18;
		const left2 = left1 + grid + 18;
		for (let i = 0; i < I; i++) nodes[i] = { x: net.x + labelW, y: top + ((i + 0.5) * grid) / I };
		for (let k = 0; k < hidden; k++) {
			const layer = k < half ? 0 : 1;
			const local = k % half;
			nodes[H0 + k] = { x: (layer ? left2 : left1) + (local % 8 + 0.5) * cell, y: top + (Math.floor(local / 8) + 0.5) * cell };
		}
		for (let k = 0; k < 2; k++) nodes[O0 + k] = { x: left2 + grid + 18 + outW / 2, y: top + grid * (k ? 0.72 : 0.28) };
	};
	layout();
	// Fitting the canvas clears it, so a new layout is drawn at once, paused or not.
	const relayout = () => {
		layout();
		frame(0);
	};
	new ResizeObserver(relayout).observe(canvas);
	desk.addEventListener('change', relayout);
	onTheme((p) => (colors = p));

	// Wide, the drone idles right of the copy and above the panel; narrow, below the copy.
	const home = (): [number, number] => (desk.matches ? [0.42 * box[0], 0.35] : [0, -0.95]);
	pilot.state.set([home()[0], home()[1], 0, 0, 0, 0]);
	let target = home();
	let pointer: [number, number] | null = null;
	let pointerAt = -1e9;
	let grabbed = false;
	let fling: [number, number] = [0, 0];
	let paused = reduced;
	let clock = 0;
	let carry = 0;
	let recent = 0;
	const trail: number[] = [];
	const sparks: Spark[] = [];

	const px = (x: number) => origin[0] + x * scale;
	const py = (y: number) => origin[1] - y * scale;
	const local = (event: PointerEvent): [number, number] => {
		const r = canvas.getBoundingClientRect();
		return [event.clientX - r.left, event.clientY - r.top];
	};
	const inNet = ([x, y]: [number, number]) => x >= net.x - 8 && x <= net.x + net.w + 8 && y >= net.y - 8 && y <= net.y + net.h + 8;
	const toWorld = ([x, y]: [number, number]): [number, number] => [(x - origin[0]) / scale, -(y - origin[1]) / scale];
	const clampTarget = (x: number, y: number): [number, number] => [
		Math.min(Math.max(x, -0.88 * box[0]), 0.88 * box[0]),
		Math.min(Math.max(y, -0.85 * box[1]), 0.85 * box[1]),
	];
	const neuronAt = ([x, y]: [number, number]) => {
		for (let k = 0; k < hidden; k++) {
			const n = nodes[H0 + k];
			if (Math.abs(n.x - x) < cell / 2 && Math.abs(n.y - y) < cell / 2) return k;
		}
		return -1;
	};
	const kick = (vx: number, vy: number, spin: number) => {
		pilot.state[2] += vx;
		pilot.state[3] += vy;
		pilot.state[5] += spin;
	};
	const shown = () => {
		const n = pilot.silenced.reduce((a, b) => a + b, 0);
		if (silencedCount) silencedCount.textContent = n ? `${n} silenced` : '';
		root.dataset.silenced = String(n);
	};
	function setPaused(value: boolean) {
		paused = value;
		root.dataset.paused = String(value);
		if (pauseButton) {
			pauseButton.setAttribute('aria-pressed', String(value));
			pauseButton.textContent = value ? 'Play' : 'Pause';
		}
	}
	setPaused(paused);

	canvas.addEventListener('pointermove', (event) => {
		const at = local(event);
		if (grabbed) {
			const [x, y] = toWorld(at);
			fling = [0.6 * fling[0] + 0.4 * ((x - pilot.state[0]) / 0.016), 0.6 * fling[1] + 0.4 * ((y - pilot.state[1]) / 0.016)];
			pilot.state[0] = Math.min(Math.max(x, -box[0]), box[0]);
			pilot.state[1] = Math.min(Math.max(y, -box[1]), box[1]);
			pointer = [x, y];
			pointerAt = clock;
			return;
		}
		if (event.pointerType !== 'mouse') return;
		const overNet = inNet(at);
		canvas.style.cursor = overNet ? (neuronAt(at) >= 0 ? 'pointer' : 'default') : 'crosshair';
		if (overNet || (!desk.matches && at[1] > net.y - 12)) {
			pointer = null;
			return;
		}
		pointer = toWorld(at);
		pointerAt = clock;
	});
	canvas.addEventListener('pointerleave', () => (pointer = null));
	canvas.addEventListener('pointerdown', (event) => {
		const at = local(event);
		if (paused && !reduced) setPaused(false);
		const neuron = inNet(at) ? neuronAt(at) : -1;
		if (neuron >= 0) {
			pilot.silenced[neuron] ^= 1;
			shown();
			return;
		}
		if (inNet(at)) return;
		const [x, y] = toWorld(at);
		const near = Math.hypot(x - pilot.state[0], y - pilot.state[1]) < 0.45;
		if (near && event.pointerType === 'mouse') {
			grabbed = true;
			fling = [0, 0];
			canvas.setPointerCapture(event.pointerId);
			root.classList.add('grabbing');
		} else if (event.pointerType === 'mouse') {
			const dx = pilot.state[0] - x;
			const dy = pilot.state[1] - y;
			const d = Math.max(Math.hypot(dx, dy), 0.3);
			const push = 4.5 * Math.exp(-d / 1.6);
			kick((push * dx) / d, (push * dy) / d, (Math.random() < 0.5 ? -1 : 1) * 10 * push);
			burst(x, y);
		} else {
			pointer = clampTarget(x, y);
			pointerAt = clock + 1e6;
		}
	});
	const release = () => {
		if (!grabbed) return;
		grabbed = false;
		root.classList.remove('grabbing');
		const speed = Math.hypot(...fling);
		const cap = speed > 9 ? 9 / speed : 1;
		pilot.state[2] = fling[0] * cap;
		pilot.state[3] = fling[1] * cap;
	};
	canvas.addEventListener('pointerup', release);
	canvas.addEventListener('pointercancel', release);

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
			const [hx, hy] = home();
			const reach = desk.matches ? [0.3 * box[0], 0.55] : [0.55 * box[0], 0.3];
			target = clampTarget(hx + reach[0] * Math.sin(0.23 * clock), hy + reach[1] * Math.sin(0.41 * clock + 0.7));
		}
		if (grabbed) {
			pilot.think(pilot.observe(target[0], target[1]));
			for (let k = 0; k < 2; k++) pilot.thrust[k] = pilot.drone.max_thrust / (1 + Math.exp(-(pilot.readout[k] + pilot.drone.hover_logit)));
		} else pilot.step(target[0], target[1], box[0], box[1]);
		let fired = 0;
		for (let i = 0; i < hidden; i++) {
			if (pilot.spikes[i]) {
				flash[H0 + i] = 1;
				fired++;
			}
		}
		recent = 0.97 * recent + 0.03 * fired;
		const s = pilot.state;
		const sin = Math.sin(s[4]);
		const cos = Math.cos(s[4]);
		for (let k = 0; k < 2; k++) {
			const power = pilot.thrust[k] / pilot.drone.max_thrust;
			if (Math.random() < power) {
				const side = k ? 1 : -1;
				const v = 1.5 + 2.5 * power;
				sparks.push({ x: s[0] + side * pilot.drone.arm * cos, y: s[1] + side * pilot.drone.arm * sin,
					vx: s[2] + v * sin + 0.3 * gauss(), vy: s[3] - v * cos + 0.3 * gauss(), life: 0.35, kind: 0 });
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

	const mono = () => getComputedStyle(root).getPropertyValue('--sx-mono') || 'monospace';

	function drawWorld(ctx: CanvasRenderingContext2D) {
		const { width } = view;
		const worldBottom = desk.matches ? view.height : 2 * origin[1];
		ctx.strokeStyle = alpha(colors.ink, colors.dark ? 0.05 : 0.065);
		ctx.lineWidth = 1;
		ctx.beginPath();
		for (let x = Math.ceil(-box[0] * 4) / 4; x <= box[0]; x += 0.25) {
			ctx.moveTo(Math.round(px(x)) + 0.5, 0);
			ctx.lineTo(Math.round(px(x)) + 0.5, worldBottom);
		}
		for (let y = -HEIGHT; y <= HEIGHT + 1e-9; y += 0.25) {
			ctx.moveTo(0, Math.round(py(y)) + 0.5);
			ctx.lineTo(width, Math.round(py(y)) + 0.5);
		}
		ctx.stroke();

		const [tx, ty] = target;
		const pulse = 0.5 + 0.5 * Math.sin(clock * 4);
		ctx.strokeStyle = alpha(colors.spike, 0.85);
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.arc(px(tx), py(ty), 10 + 3 * pulse, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
			ctx.moveTo(px(tx) + dx * 16, py(ty) + dy * 16);
			ctx.lineTo(px(tx) + dx * 23, py(ty) + dy * 23);
		}
		ctx.stroke();

		const s = pilot.state;
		ctx.setLineDash([3, 5]);
		ctx.strokeStyle = alpha(colors.ink, 0.18);
		ctx.beginPath();
		ctx.moveTo(px(s[0]), py(s[1]));
		ctx.lineTo(px(tx), py(ty));
		ctx.stroke();
		ctx.setLineDash([]);

		for (let k = 2; k < trail.length; k += 2) {
			ctx.strokeStyle = alpha(colors.membrane, (0.4 * k) / trail.length);
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(px(trail[k - 2]), py(trail[k - 1]));
			ctx.lineTo(px(trail[k]), py(trail[k + 1]));
			ctx.stroke();
		}
		for (const p of sparks) {
			const life = p.kind ? p.life / 0.45 : p.life / 0.35;
			ctx.fillStyle = alpha(p.kind ? colors.spike : colors.membrane, Math.max(0, life) * (p.kind ? 0.95 : 0.4));
			const r = p.kind ? 1.7 : 2.3;
			ctx.fillRect(px(p.x) - r, py(p.y) - r, 2 * r, 2 * r);
		}
	}

	function rotors(): [number, number][] {
		const s = pilot.state;
		const arm = pilot.drone.arm;
		const unit = scale / 100;
		return [-1, 1].map((side) => {
			const bx = side * arm;
			const by = -6 * unit / scale;
			return [px(s[0] + bx * Math.cos(s[4]) - by * Math.sin(s[4])), py(s[1] + bx * Math.sin(s[4]) + by * Math.cos(s[4]))];
		});
	}

	function drawWires(ctx: CanvasRenderingContext2D) {
		const ends = rotors();
		for (let k = 0; k < 2; k++) {
			const from = nodes[O0 + k];
			const [x2, y2] = ends[k];
			const power = pilot.thrust[k] / pilot.drone.max_thrust;
			const bend = Math.max(80, Math.abs(x2 - from.x) * 0.45);
			const c1 = desk.matches ? [from.x + bend, from.y] : [from.x, from.y - bend];
			const c2 = desk.matches ? [x2 - bend * 0.3, y2 + bend * 0.6] : [x2, y2 + bend * 0.6];
			ctx.strokeStyle = alpha(colors.membrane, 0.12 + 0.35 * power);
			ctx.lineWidth = 1 + 2.5 * power;
			ctx.beginPath();
			ctx.moveTo(from.x, from.y);
			ctx.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], x2, y2);
			ctx.stroke();
			ctx.fillStyle = colors.membrane;
			for (let q = 0; q < 4; q++) {
				const t = (clock * (0.35 + 0.8 * power) + q / 4 + k * 0.13) % 1;
				const u = 1 - t;
				const bx = u * u * u * from.x + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * x2;
				const by = u * u * u * from.y + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * y2;
				ctx.globalAlpha = 0.3 + 0.7 * power;
				ctx.beginPath();
				ctx.arc(bx, by, 1.6 + 1.6 * power, 0, 2 * Math.PI);
				ctx.fill();
			}
			ctx.globalAlpha = 1;
		}
	}

	function drawDrone(ctx: CanvasRenderingContext2D) {
		const s = pilot.state;
		ctx.save();
		ctx.translate(px(s[0]), py(s[1]));
		ctx.rotate(-s[4]);
		const arm = pilot.drone.arm * scale;
		const unit = scale / 100;
		for (let k = 0; k < 2; k++) {
			const side = k ? 1 : -1;
			const power = pilot.thrust[k] / pilot.drone.max_thrust;
			const plume = ctx.createLinearGradient(0, 4 * unit, 0, (10 + 34 * power) * unit);
			plume.addColorStop(0, alpha(colors.membrane, 0.5 * power + 0.1));
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

	function drawNetwork(ctx: CanvasRenderingContext2D) {
		ctx.lineWidth = 1;
		for (const e of edges) {
			const a = nodes[e.from];
			const b = nodes[e.to];
			const lit = e.from >= H0 && e.from < O0 ? flash[e.from] : 0;
			const base = colors.dark ? 0.05 : 0.07;
			ctx.strokeStyle = lit > 0.25 ? alpha(colors.spike, Math.min(0.7, lit * (0.1 + 0.6 * Math.abs(e.strength)))) : alpha(colors.ink, base * Math.abs(e.strength) + 0.015);
			ctx.beginPath();
			ctx.moveTo(a.x, a.y);
			ctx.lineTo(b.x, b.y);
			ctx.stroke();
		}
		ctx.font = `500 10px ${mono()}`;
		ctx.textBaseline = 'middle';
		ctx.textAlign = 'right';
		for (let i = 0; i < I; i++) {
			const n = nodes[i];
			const value = Math.max(-1, Math.min(1, pilot.obs[i]));
			ctx.fillStyle = colors.muted;
			ctx.fillText(narrow ? SHORT[i] : INPUTS[i], n.x - 30, n.y);
			ctx.fillStyle = alpha(colors.ink, 0.12);
			ctx.fillRect(n.x - 24, n.y - 2, 20, 4);
			ctx.fillStyle = colors.membrane;
			ctx.fillRect(n.x - 14, n.y - 2, 10 * value, 4);
			ctx.beginPath();
			ctx.arc(n.x, n.y, 2.5, 0, 2 * Math.PI);
			ctx.fill();
		}
		const r = Math.max(2, cell * 0.3);
		for (let k = 0; k < hidden; k++) {
			const n = nodes[H0 + k];
			const f = flash[H0 + k];
			if (pilot.silenced[k]) {
				ctx.strokeStyle = alpha(colors.ink, 0.35);
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(n.x - r, n.y - r);
				ctx.lineTo(n.x + r, n.y + r);
				ctx.moveTo(n.x + r, n.y - r);
				ctx.lineTo(n.x - r, n.y + r);
				ctx.stroke();
				continue;
			}
			if (f > 0.3) {
				ctx.fillStyle = alpha(colors.spike, 0.16 * f);
				ctx.beginPath();
				ctx.arc(n.x, n.y, r * 2.2, 0, 2 * Math.PI);
				ctx.fill();
			}
			ctx.fillStyle = f > 0.05 ? alpha(colors.spike, 0.35 + 0.65 * f) : alpha(colors.ink, 0.22);
			ctx.beginPath();
			ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
			ctx.fill();
		}
		ctx.textAlign = 'center';
		ctx.textBaseline = 'alphabetic';
		for (let k = 0; k < 2; k++) {
			const n = nodes[O0 + k];
			const power = pilot.thrust[k] / pilot.drone.max_thrust;
			const R = Math.max(9, cell * 0.9);
			ctx.fillStyle = alpha(colors.membrane, 0.12);
			ctx.beginPath();
			ctx.arc(n.x, n.y, R, 0, 2 * Math.PI);
			ctx.fill();
			ctx.fillStyle = colors.membrane;
			ctx.beginPath();
			ctx.moveTo(n.x, n.y);
			ctx.arc(n.x, n.y, R, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * power);
			ctx.fill();
			ctx.fillStyle = colors.muted;
			ctx.fillText(k ? 'right' : 'left', n.x, n.y + R + 13);
		}
		const grid = cell * 8;
		const top = nodes[H0].y - cell / 2 - 10;
		ctx.fillStyle = colors.muted;
		ctx.fillText(narrow ? '64 LIF' : 'layer 1 · 64 LIF', nodes[H0].x - cell / 2 + grid / 2, top);
		ctx.fillText(narrow ? '64 LIF' : 'layer 2 · 64 LIF', nodes[H0 + half].x - cell / 2 + grid / 2, top);
		ctx.fillText('rotors', nodes[O0].x, top);
		ctx.textAlign = 'right';
		ctx.fillText(narrow ? 'in' : 'readings', nodes[0].x - 4, top);
	}

	const frame = (seconds: number) => {
		if (!colors) return;
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
			const fade = Math.exp(-seconds / 0.03);
			for (let k = 0; k < flash.length; k++) flash[k] *= fade;
			if (rate) rate.textContent = `${Math.round(recent / hidden / dt)} Hz`;
		}
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		drawWorld(ctx);
		drawNetwork(ctx);
		drawWires(ctx);
		drawDrone(ctx);
	};

	if (reduced) {
		for (let k = 0; k < 150; k++) step();
	}
	// The canvas, not the section: on the racer's tab it is hidden and the drone rests.
	const loop = animate(canvas, frame);
	onTheme(() => !loop.running() && frame(0));
	shown();
}
