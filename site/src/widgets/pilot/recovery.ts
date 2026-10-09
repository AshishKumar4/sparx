// One flight of the trained pilot (src/engines/pilot.ts) from a start the reader picks, replayed in real time:
// the drone's path in its box, every neuron's spikes, and the thrust of each rotor.
import { Pilot, type PilotModel } from '../../engines/pilot';
import { alpha, animate, fit, onTheme, type Palette } from '../theme';

export const STEPS = 300;
export const BOX: [number, number] = [2.4, 1.5];
export const TARGET: [number, number] = [1.2, 0.6];

/** x, y, vx, vy, θ, ω for each start the figure offers. */
export const STARTS: Record<string, number[]> = {
	upside: [-1.6, -0.6, 0, 0, Math.PI, 0],
	thrown: [-1.8, -1.0, 4, 3, 0.3, 12],
	falling: [0, 1.2, 2, -3, Math.PI / 2, -6],
};

export interface Flight {
	states: Float64Array[];
	spikes: Uint8Array[];
	thrust: [number, number][];
}

export function fly(model: PilotModel, start: number[]): Flight {
	const pilot = new Pilot(model);
	pilot.state.set(start);
	const flight: Flight = { states: [Float64Array.from(pilot.state)], spikes: [], thrust: [] };
	for (let t = 0; t < STEPS; t++) {
		pilot.step(TARGET[0], TARGET[1], BOX[0], BOX[1]);
		flight.states.push(Float64Array.from(pilot.state));
		flight.spikes.push(Uint8Array.from(pilot.spikes));
		flight.thrust.push([pilot.thrust[0], pilot.thrust[1]]);
	}
	return flight;
}

/** The first step from which the drone stays within 15 cm of the target, or null. */
export function arrival(flight: Flight): number | null {
	let from: number | null = null;
	for (let t = 0; t < flight.states.length; t++) {
		const s = flight.states[t];
		const near = Math.hypot(s[0] - TARGET[0], s[1] - TARGET[1]) < 0.15;
		if (near && from === null) from = t;
		if (!near) from = null;
	}
	return from;
}

export function recovery(canvas: HTMLCanvasElement, model: PilotModel, report: (flight: Flight) => void) {
	const hover = (model.drone.mass * model.drone.gravity) / 2;
	let flight = fly(model, STARTS.upside);
	let cursor = STEPS;
	report(flight);
	let colors: Palette;
	let view = fit(canvas);
	const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});

	function draw() {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const narrow = width < 560;
		const pad = 12;
		const sky = narrow ? { x: pad, y: pad, w: width - 2 * pad, h: height * 0.42 } : { x: pad, y: pad, w: width * 0.44, h: height - 2 * pad };
		const right = narrow
			? { x: pad, y: sky.y + sky.h + 12, w: width - 2 * pad, h: height - sky.h - 3 * pad }
			: { x: sky.x + sky.w + 16, y: pad, w: width - sky.w - 16 - 2 * pad, h: height - 2 * pad };
		const scale = Math.min(sky.w / (2 * BOX[0]), sky.h / (2 * BOX[1]));
		const cx = sky.x + sky.w / 2;
		const cy = sky.y + sky.h / 2;
		const px = (x: number) => cx + x * scale;
		const py = (y: number) => cy - y * scale;
		const now = Math.min(cursor, STEPS);

		ctx.strokeStyle = colors.line;
		ctx.lineWidth = 1;
		ctx.strokeRect(px(-BOX[0]) + 0.5, py(BOX[1]) + 0.5, 2 * BOX[0] * scale - 1, 2 * BOX[1] * scale - 1);
		ctx.strokeStyle = colors.bio;
		ctx.beginPath();
		ctx.arc(px(TARGET[0]), py(TARGET[1]), 0.15 * scale, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.strokeStyle = alpha(colors.membrane, 0.6);
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		for (let t = 0; t <= now; t++) {
			const s = flight.states[t];
			if (t === 0) ctx.moveTo(px(s[0]), py(s[1]));
			else ctx.lineTo(px(s[0]), py(s[1]));
		}
		ctx.stroke();
		const s = flight.states[now];
		ctx.save();
		ctx.translate(px(s[0]), py(s[1]));
		ctx.rotate(-s[4]);
		ctx.strokeStyle = colors.ink;
		ctx.lineWidth = 3;
		ctx.lineCap = 'round';
		ctx.beginPath();
		ctx.moveTo(-0.25 * scale, 0);
		ctx.lineTo(0.25 * scale, 0);
		ctx.stroke();
		ctx.fillStyle = colors.spike;
		for (const side of [-1, 1]) ctx.fillRect(side * 0.25 * scale - 4, -5, 8, 3);
		ctx.restore();

		const rasterH = right.h * 0.62;
		const n = flight.spikes[0].length;
		const dx = right.w / STEPS;
		const dy = rasterH / n;
		ctx.fillStyle = colors.spike;
		for (let t = 0; t < now; t++) {
			const row = flight.spikes[t];
			for (let i = 0; i < n; i++) if (row[i]) ctx.fillRect(right.x + t * dx, right.y + i * dy, Math.max(dx, 1), Math.max(dy, 1));
		}
		ctx.strokeStyle = colors.line;
		ctx.strokeRect(right.x + 0.5, right.y + 0.5, right.w - 1, rasterH - 1);
		const top = right.y + rasterH + 10;
		const h = right.h - rasterH - 10;
		const y = (f: number) => top + h - (f / 12) * h;
		ctx.setLineDash([4, 4]);
		ctx.strokeStyle = alpha(colors.ink, 0.35);
		ctx.beginPath();
		ctx.moveTo(right.x, y(hover));
		ctx.lineTo(right.x + right.w, y(hover));
		ctx.stroke();
		ctx.setLineDash([]);
		for (const [k, role] of [
			[0, 'membrane'],
			[1, 'learn'],
		] as const) {
			ctx.strokeStyle = colors[role];
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			for (let t = 0; t < now; t++) {
				const f = flight.thrust[t][k];
				if (t === 0) ctx.moveTo(right.x, y(f));
				else ctx.lineTo(right.x + t * dx, y(f));
			}
			ctx.stroke();
		}
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = colors.muted;
		ctx.fillText(`${(now / 100).toFixed(2)} s`, right.x + right.w - 44, top + 12);
	}

	// Steps of 10 ms not yet shown, so the replay runs in real time at any frame rate.
	let owed = 0;
	animate(canvas, (dt) => {
		if (reduced || cursor >= STEPS) return;
		owed += dt * 100;
		const steps = Math.floor(owed);
		if (!steps) return;
		owed -= steps;
		cursor = Math.min(STEPS, cursor + steps);
		draw();
	});
	return {
		start(name: string) {
			flight = fly(model, STARTS[name]);
			cursor = reduced ? STEPS : 0;
			owed = 0;
			report(flight);
			draw();
		},

	};
}
