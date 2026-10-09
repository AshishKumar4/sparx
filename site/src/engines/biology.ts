// sparx.dynamics in physical units with synapses that open conductances (src/sparx/dynamics/core.py,
// neurons.py, synapses.py), op for op in float64: what the biology chapter's figures step.

/** sparx's reversal potentials (mV), `RECEPTORS`, from Brette et al.'s benchmarks (2007). */
export const RECEPTORS: Record<string, number> = { ampa: 0, nmda: 0, gaba_a: -80, gaba_b: -95 };

const factorial = (n: number): number => (n < 2 ? 1 : n * factorial(n - 1));
const PHI1 = Array.from({ length: 14 }, (_, n) => 1 / factorial(n + 1));
const PSI = Array.from({ length: 14 }, (_, n) => 1 / (factorial(n) * (n + 2)));

function series(x: number, coefficients: number[]): number {
	let out = 0;
	for (let k = coefficients.length - 1; k >= 0; k--) out = out * x + coefficients[k];
	return out;
}

/** `(e^x - 1) / x`, by series near 0. */
const phi1 = (x: number) => (Math.abs(x) < 0.5 ? series(x, PHI1) : Math.expm1(x) / x);
/** `(e^x (x - 1) + 1) / x^2`, by series near 0. */
const psi = (x: number) => (Math.abs(x) < 0.5 ? series(x, PSI) : (Math.exp(x) * (x - 1) + 1) / x ** 2);

/** A current or conductance `(amplitude + slope s) exp(-s / tau)` over a step, `s` ms from its start. */
export interface Term {
	amplitude: number;
	slope: number;
	tau: number;
}

/** The term's exact average over the step. */
export function mean(term: Term, dt: number): number {
	const x = -dt / term.tau;
	return term.amplitude * phi1(x) + term.slope * dt * psi(x);
}

/** What a membrane with time constant `tau` gains from the current `term` over the step, times its capacitance. */
export function response(term: Term, tau: number, dt: number): number {
	const x = -dt * (1 / term.tau - 1 / tau);
	return Math.exp(-dt / tau) * dt * (term.amplitude * phi1(x) + term.slope * dt * psi(x));
}

/** NMDA's magnesium block at `v` (mV): the fraction of receptors open to current (Jahr and Stevens 1990). */
export const mgBlock = (v: number, mg = 1) => 1 / (1 + (mg / 3.57) * Math.exp(-0.062 * v));

/** One step's input: a held current (pA), current waveforms, and conductances (nS) by receptor, in order. */
export interface Input {
	current: number;
	currents: Term[];
	conductance: [string, number][];
	jump: number;
}

/** `sparx.dynamics.LeakyIntegrateAndFire`: each step solved exactly with the conductances held, NMDA gated. */
export class LeakyIntegrateAndFire {
	v: number;
	refractory = 0;
	constructor(
		public tau_m = 20,
		public c_m = 200,
		public e_l = -60,
		public v_th = -50,
		public v_reset = -60,
		public t_ref = 5,
		public i_e = 0,
		public reversal: Record<string, number> = RECEPTORS,
	) {
		this.v = e_l;
	}

	step(input: Input, dt: number): number {
		const g_l = this.c_m / this.tau_m;
		let g_syn = 0;
		let syn_drive = 0;
		for (const [name, g] of input.conductance) {
			const gated = name === 'nmda' ? g * mgBlock(this.v) : g;
			g_syn += gated;
			syn_drive += gated * this.reversal[name];
		}
		const g_total = g_l + g_syn;
		const drive = g_l * this.e_l + this.i_e + input.current + syn_drive;
		const tau = this.c_m / g_total;
		const target = drive / g_total;
		let waves = 0;
		for (const term of input.currents) waves += response(term, tau, dt);
		const integrated = target + (this.v - target) * Math.exp(-dt / tau) + waves / this.c_m + input.jump;
		const held = this.refractory > dt / 2;
		const v = held ? this.v_reset : integrated;
		const fired = !held && v >= this.v_th;
		this.v = fired ? this.v_reset : v;
		this.refractory = fired ? this.t_ref : Math.max(this.refractory - dt, 0);
		return fired ? 1 : 0;
	}
}

/** A synapse onto the neuron: exponential decay with `tau` (ms), its output a current or a conductance. */
export interface Receptor {
	tau: number;
	kind: 'current' | 'conductance';
}

/** `sparx.dynamics.PointNeuron` of a `LeakyIntegrateAndFire` and exponential synapses, by receptor name, with
 * each conductance held over the step at its mean (`hold="mean"`, second order) or at its value at the step's
 * start (`"start"`, Brian2's `exponential_euler`). `held` adds conductances of its own, as a `SynapticInput`
 * takes them. */
export class PointNeuron {
	readonly synapses: Record<string, number> = {};
	constructor(
		readonly cell: LeakyIntegrateAndFire,
		readonly receptors: Record<string, Receptor>,
		readonly hold: 'mean' | 'start' = 'mean',
	) {
		for (const name in receptors) this.synapses[name] = 0;
	}

	step(current: number, arriving: Record<string, number>, dt: number, held: [string, number][] = []): number {
		const currents: Term[] = [];
		const conductance: [string, number][] = [];
		for (const [name, { tau, kind }] of Object.entries(this.receptors)) {
			const term = { amplitude: this.synapses[name], slope: 0, tau };
			if (kind === 'current') currents.push(term);
			else conductance.push([name, 0 + (this.hold === 'start' ? term.amplitude : mean(term, dt))]);
		}
		const fired = this.cell.step({ current, currents, conductance: [...conductance, ...held], jump: 0 }, dt);
		for (const [name, { tau }] of Object.entries(this.receptors))
			this.synapses[name] = this.synapses[name] * Math.exp(-dt / tau) + (arriving[name] ?? 0);
		return fired;
	}
}

/** `x / (1 - exp(-x / y))`, continuous through `x = 0`. */
function vtrap(x: number, y: number): number {
	if (Math.abs(x / y) < 1e-6) return y + x / 2;
	return x / -Math.expm1(-x / y);
}

/** `(alpha, beta)` of m, h and n at `v`, 1/ms. */
export function rates(v: number): [number, number][] {
	return [
		[0.1 * vtrap(v + 40, 10), 4 * Math.exp(-(v + 65) / 18)],
		[0.07 * Math.exp(-(v + 65) / 20), 1 / (1 + Math.exp(-(v + 35) / 10))],
		[0.01 * vtrap(v + 55, 10), 0.125 * Math.exp(-(v + 65) / 80)],
	];
}

const exact = (x: number, target: number, tau: number, dt: number) => target + (x - target) * Math.exp(-dt / tau);

/** `sparx.dynamics.HodgkinHuxley` with its default scheme, `"strang"`: in substeps of 0.01 ms, half a substep
 * of each gate with the voltage held, a substep of the voltage with the gates held, and half a substep of the
 * gates again, each solved exactly. A spike is reported where the voltage, at or above 0 mV, has begun to fall. */
export class HodgkinHuxley {
	v: number;
	m: number;
	h: number;
	n: number;
	refractory = 0;
	constructor(
		public c_m = 100,
		public g_na = 12000,
		public g_k = 3600,
		public g_l = 30,
		public e_na = 50,
		public e_k = -77,
		public e_l = -54.402,
		v_init = -65,
		public v_spike = 0,
		public t_ref = 2,
		public substep = 0.01,
	) {
		this.v = v_init;
		[this.m, this.h, this.n] = rates(v_init).map(([alpha, beta]) => alpha / (alpha + beta));
	}

	/** Sodium and potassium conductances (nS) at the current state. */
	get conductances(): [number, number] {
		// As XLA multiplies out m ** 3 and n ** 4.
		const m3 = this.m * (this.m * this.m);
		const n2 = this.n * this.n;
		return [this.g_na * m3 * this.h, this.g_k * (n2 * n2)];
	}

	private gates(v: number, length: number) {
		const [[am, bm], [ah, bh], [an, bn]] = rates(v);
		this.m = exact(this.m, am / (am + bm), 1 / (am + bm), length);
		this.h = exact(this.h, ah / (ah + bh), 1 / (ah + bh), length);
		this.n = exact(this.n, an / (an + bn), 1 / (an + bn), length);
	}

	step(current: number, dt: number): number {
		const before = this.v;
		const count = Math.max(1, Math.ceil(dt / this.substep - 1e-9));
		const h = dt / count;
		let v = this.v;
		for (let i = 0; i < count; i++) {
			this.gates(v, h / 2);
			const [g_na, g_k] = this.conductances;
			const g_total = g_na + g_k + this.g_l + 0;
			const drive = g_na * this.e_na + g_k * this.e_k + this.g_l * this.e_l + 0 + current;
			v = exact(v, drive / g_total, this.c_m / g_total, h);
			this.gates(v, h / 2);
		}
		this.v = v;
		const fired = v >= this.v_spike && v < before && this.refractory <= dt / 2;
		this.refractory = fired ? this.t_ref : Math.max(this.refractory - dt, 0);
		return fired ? 1 : 0;
	}
}
