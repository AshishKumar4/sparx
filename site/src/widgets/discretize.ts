// Chapter 2's figure: a membrane charged by a current step, solved exactly (the curve), stepped exactly
// at dt (sparx's `decay ** dt`, the dots) and stepped by forward Euler at dt (the squares).
import { alpha, fit, onTheme, type Palette } from './theme';

export interface Membrane {
	tau: number;
	/** The current times the resistance: where the membrane settles, in units of the threshold. */
	drive: number;
	/** When the current switches on and off, ms. */
	on: number;
	off: number;
	span: number;
}

export function exact(m: Membrane, t: number): number {
	const rise = (s: number) => m.drive * (1 - Math.exp(-s / m.tau));
	if (t <= m.on) return 0;
	if (t <= m.off) return rise(t - m.on);
	return rise(m.off - m.on) * Math.exp(-(t - m.off) / m.tau);
}

/** The membrane at each step of `dt`, by the exact update `v = e^(-dt/tau) v + (1 - e^(-dt/tau)) R I` or by
 * Euler's `v = v + dt / tau (R I - v)`, the current read at the start of each step. */
export function stepped(m: Membrane, dt: number, euler: boolean): [number, number][] {
	const out: [number, number][] = [[0, 0]];
	let v = 0;
	const keep = euler ? 1 - dt / m.tau : Math.exp(-dt / m.tau);
	for (let t = 0; t < m.span - 1e-9; t += dt) {
		const drive = t >= m.on && t < m.off ? m.drive : 0;
		v = keep * v + (1 - keep) * drive;
		out.push([t + dt, v]);
	}
	return out;
}

export function discretize(root: HTMLElement, read: () => { m: Membrane; dt: number }) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	let colors: Palette;
	let view = fit(canvas);
	const draw = () => {
		if (!colors) return;
		const { m, dt } = read();
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const left = 40;
		const right = width - 12;
		const top = 14;
		const bottom = height - 26;
		const lo = -0.6;
		const hi = 1.6;
		const x = (t: number) => left + (t / m.span) * (right - left);
		const y = (v: number) => bottom - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (bottom - top);
		ctx.font = `11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.fillStyle = alpha(colors.ink, 0.05);
		ctx.fillRect(x(m.on), top, x(m.off) - x(m.on), bottom - top);
		ctx.strokeStyle = alpha(colors.ink, 0.15);
		ctx.beginPath();
		ctx.moveTo(left, Math.round(y(0)) + 0.5);
		ctx.lineTo(right, Math.round(y(0)) + 0.5);
		ctx.stroke();
		ctx.fillStyle = colors.muted;
		ctx.textAlign = 'right';
		for (const v of [0, 1]) ctx.fillText(String(v), left - 6, y(v) + 4);
		ctx.textAlign = 'center';
		for (let t = 0; t <= m.span; t += 20) ctx.fillText(`${t}`, x(t), height - 8);
		ctx.textAlign = 'left';
		ctx.fillText('ms', right - 16, height - 8);
		ctx.strokeStyle = colors.ink2;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		for (let k = 0; k <= 600; k++) {
			const t = (k / 600) * m.span;
			if (k === 0) ctx.moveTo(x(t), y(exact(m, t)));
			else ctx.lineTo(x(t), y(exact(m, t)));
		}
		ctx.stroke();
		const series = (points: [number, number][], color: string, square: boolean) => {
			ctx.strokeStyle = alpha(color, 0.5);
			ctx.lineWidth = 1;
			ctx.beginPath();
			for (const [k, [t, v]] of points.entries()) {
				if (k === 0) ctx.moveTo(x(t), y(v));
				else ctx.lineTo(x(t), y(v));
			}
			ctx.stroke();
			ctx.fillStyle = color;
			for (const [t, v] of points) {
				if (square) ctx.fillRect(x(t) - 3.5, y(v) - 3.5, 7, 7);
				else {
					ctx.beginPath();
					ctx.arc(x(t), y(v), 3.5, 0, 2 * Math.PI);
					ctx.fill();
				}
			}
		};
		const eu = stepped(m, dt, true);
		const ex = stepped(m, dt, false);
		series(eu, colors.spike, true);
		series(ex, colors.membrane, false);
		const err = (pts: [number, number][]) => Math.max(...pts.map(([t, v]) => Math.abs(v - exact(m, t))));
		if (status) {
			const keep = 1 - dt / m.tau;
			status.textContent = `dt / tau = ${(dt / m.tau).toFixed(2)} · largest error at the steps: exact ${err(ex).toExponential(1)}, Euler ${err(eu).toFixed(3)}${keep < 0 ? ` · Euler keeps ${keep.toFixed(2)} of v each step: it overshoots and flips sign` : ''}`;
		}
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
