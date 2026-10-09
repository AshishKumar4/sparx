// sparx's neuron models, one step at a time, in double precision. Each mirrors the `step` of the
// model it is named after (sparx.dynamics), and test/neurons.test.ts holds it to sparx's run.

export type Reset = 'subtract' | 'zero' | 'none';

/** `sparx.dynamics.LIFCell`: `v = decay v + x`, a spike at `v >= threshold`, then the reset. */
export class LIF {
	v = 0;
	/** The membrane before this step's reset, which crossed the threshold if it fired. */
	peak = 0;
	constructor(
		public decay: number,
		public threshold = 1,
		public reset: Reset = 'subtract',
	) {}

	step(x: number): number {
		const v = this.decay * this.v + x;
		const s = v >= this.threshold ? 1 : 0;
		this.peak = v;
		this.v = !s || this.reset === 'none' ? v : this.reset === 'zero' ? 0 : v - this.threshold;
		return s;
	}
}

/** `sparx.dynamics.ALIFCell` without refractoriness: the threshold is `threshold + beta * a`, a spike
 * subtracts the baseline `threshold`, and `a` decays by `adaptDecay` a step and gains 1 per spike. */
export class ALIF {
	v = 0;
	a = 0;
	peak = 0;
	constructor(
		public decay: number,
		public adaptDecay: number,
		public beta = 1.8,
		public threshold = 1,
	) {}

	step(x: number): number {
		const theta = this.threshold + this.beta * this.a;
		const v = this.decay * this.v + x;
		const s = v >= theta ? 1 : 0;
		this.peak = v;
		this.v = v - s * this.threshold;
		this.a = this.adaptDecay * this.a + s;
		return s;
	}
}

/** `sparx.dynamics.Serial(LICell(synapse), LIFCell(decay))`, the current-based LIF: each input jumps a
 * synaptic current that decays by `synapse` a step, and the membrane integrates the current. */
export class CurrentLIF {
	i = 0;
	readonly cell: LIF;
	constructor(
		public synapse: number,
		decay: number,
		threshold = 1,
		reset: Reset = 'subtract',
	) {
		this.cell = new LIF(decay, threshold, reset);
	}

	step(x: number): number {
		this.i = this.synapse * this.i + x;
		return this.cell.step(this.i);
	}
}

export { LeakyIntegrateAndFire } from './biology';

/** `sparx.dynamics.Izhikevich` with `scheme="published"`: two half-steps of `v`, then `u` from the new
 * `v`; a spike at 30 mV sets `v` to `c` and adds `d` to `u`. */
export class Izhikevich {
	v: number;
	u: number;
	constructor(
		public a = 0.02,
		public b = 0.2,
		public c = -65,
		public d = 8,
		public v_th = 30,
		v_init = -65,
	) {
		this.v = v_init;
		this.u = b * v_init;
	}

	step(i: number, dt: number): number {
		const dv = (v: number, u: number) => 0.04 * v ** 2 + 5 * v + 140 - u + i;
		let v = this.v + (dt / 2) * dv(this.v, this.u);
		v = v + (dt / 2) * dv(v, this.u);
		const u = this.u + dt * this.a * (this.b * v - this.u);
		const fired = v >= this.v_th;
		this.v = fired ? this.c : v;
		this.u = fired ? u + this.d : u;
		return fired ? 1 : 0;
	}
}

/** Izhikevich's (2003) cortical and thalamic classes, `(a, b, c, d)`, as `sparx.dynamics.IZHIKEVICH_2003`. */
export const IZHIKEVICH_2003: Record<string, [number, number, number, number]> = {
	regular_spiking: [0.02, 0.2, -65, 8],
	intrinsically_bursting: [0.02, 0.2, -55, 4],
	chattering: [0.02, 0.2, -50, 2],
	fast_spiking: [0.1, 0.2, -65, 2],
	low_threshold_spiking: [0.02, 0.25, -65, 2],
	thalamo_cortical: [0.02, 0.25, -65, 0.05],
	resonator: [0.1, 0.26, -65, 2],
};
