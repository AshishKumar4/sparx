// Brunel's (2000) network as `sparx.graph.models.brunel` builds it and `sparx.graph.Network` steps it:
// LIF membranes in mV solved exactly over each step, delta synapses that land before the threshold
// test `delay` steps after the spike, external Poisson input landing in the step it is drawn, and a
// reset held for `t_ref`. test/brunel.test.ts holds it to sparx's run on the same edges and input.

export interface Edges {
	pre: Int32Array;
	post: Int32Array;
	weight: Float64Array;
}

export interface NetworkSpec {
	excitatory: number;
	inhibitory: number;
	edges: Edges;
	delay: number;
	dt: number;
	tau_m: number;
	v_th: number;
	v_reset: number;
	t_ref: number;
	e_l: number;
}

export class DeltaNetwork {
	readonly size: number;
	readonly v: Float64Array;
	readonly refractory: Float64Array;
	readonly fired: Uint8Array;
	private readonly start: Int32Array;
	private readonly targets: Int32Array;
	private readonly weights: Float64Array;
	private readonly ring: Float64Array[];
	private readonly leak: number;
	t = 0;

	constructor(readonly spec: NetworkSpec) {
		const n = spec.excitatory + spec.inhibitory;
		this.size = n;
		this.v = new Float64Array(n).fill(spec.e_l);
		this.refractory = new Float64Array(n);
		this.fired = new Uint8Array(n);
		// Each neuron's out-edges, laid end to end in the order they were given: a counting sort by `pre`.
		const { pre, post, weight } = spec.edges;
		this.start = new Int32Array(n + 1);
		for (const i of pre) this.start[i + 1]++;
		for (let i = 0; i < n; i++) this.start[i + 1] += this.start[i];
		const next = this.start.slice(0, n);
		this.targets = new Int32Array(pre.length);
		this.weights = new Float64Array(pre.length);
		for (let e = 0; e < pre.length; e++) {
			const k = next[pre[e]]++;
			this.targets[k] = post[e];
			this.weights[k] = weight[e];
		}
		this.ring = Array.from({ length: spec.delay + 1 }, () => new Float64Array(n));
		this.leak = Math.exp(-spec.dt / spec.tau_m);
	}

	/** One step with `external` jumps (mV) per neuron; returns how many neurons fired. */
	step(external: Float64Array): number {
		const { dt, v_th, v_reset, t_ref, e_l, delay } = this.spec;
		const due = this.ring[this.t % (delay + 1)];
		let count = 0;
		for (let i = 0; i < this.size; i++) {
			const jump = external[i] + due[i];
			const integrated = e_l + (this.v[i] - e_l) * this.leak + jump;
			const held = this.refractory[i] > dt / 2;
			const v = held ? v_reset : integrated;
			const fired = !held && v >= v_th;
			this.v[i] = fired ? v_reset : v;
			this.refractory[i] = fired ? t_ref : Math.max(this.refractory[i] - dt, 0);
			this.fired[i] = fired ? 1 : 0;
			count += this.fired[i];
		}
		due.fill(0);
		const ahead = this.ring[(this.t + delay) % (delay + 1)];
		for (let i = 0; i < this.size; i++) {
			if (!this.fired[i]) continue;
			for (let k = this.start[i]; k < this.start[i + 1]; k++) ahead[this.targets[k]] += this.weights[k];
		}
		this.t++;
		return count;
	}
}

/** Uniform numbers in [0, 1) from sfc32, seeded by splitmix32, so a run replays from its seed. */
export function generator(seed: number): () => number {
	let state = seed >>> 0;
	const mix = () => {
		state = (state + 0x9e3779b9) | 0;
		let z = state;
		z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
		z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
		return (z ^ (z >>> 15)) >>> 0;
	};
	let a = mix();
	let b = mix();
	let c = mix();
	let d = mix();
	return () => {
		const t = (((a + b) | 0) + d) | 0;
		d = (d + 1) | 0;
		a = b ^ (b >>> 9);
		b = (c + (c << 3)) | 0;
		c = (c << 21) | (c >>> 11);
		c = (c + t) | 0;
		return (t >>> 0) / 4294967296;
	};
}

export interface Brunel {
	order: number;
	g: number;
	eta: number;
	j?: number;
	delay?: number;
	epsilon?: number;
	dt?: number;
}

/** A Brunel network and the source of its external input: each call draws one step's jumps (mV) per neuron. */
export interface BrunelRun {
	network: DeltaNetwork;
	draw: () => Float64Array;
}

/** `sparx.graph.models.brunel`'s network with its own draws: each neuron takes `C_E` excitatory and `C_I`
 * inhibitory inputs with replacement, as NEST's `fixed_indegree` does, and Poisson input from `C_E`
 * sources at `eta` times the threshold rate. Returns the network and a source of its external input. */
export function brunel({ order, g, eta, j = 0.1, delay = 1.5, epsilon = 0.1, dt = 0.1 }: Brunel, seed = 1): BrunelRun {
	const random = generator(seed);
	const excitatory = 4 * order;
	const inhibitory = order;
	const n = excitatory + inhibitory;
	const ce = Math.round(epsilon * excitatory);
	const ci = Math.round(epsilon * inhibitory);
	const edges: Edges = { pre: new Int32Array(n * (ce + ci)), post: new Int32Array(n * (ce + ci)), weight: new Float64Array(n * (ce + ci)) };
	let e = 0;
	for (let post = 0; post < n; post++) {
		for (let k = 0; k < ce + ci; k++, e++) {
			const inhibitoryInput = k >= ce;
			edges.pre[e] = inhibitoryInput ? excitatory + Math.floor(random() * inhibitory) : Math.floor(random() * excitatory);
			edges.post[e] = post;
			edges.weight[e] = inhibitoryInput ? -g * j : j;
		}
	}
	const network = new DeltaNetwork({
		excitatory, inhibitory, edges, delay: Math.round(delay / dt), dt, tau_m: 20, v_th: 20, v_reset: 10, t_ref: 2, e_l: 0,
	});
	const mean = ((eta * (20 / (j * ce * 20)) * 1000 * ce) * dt) / 1000;
	const table: number[] = [];
	let p = Math.exp(-mean);
	let total = p;
	for (let k = 1; total < 1 - 1e-12 && k < 200; k++) {
		table.push(total);
		p *= mean / k;
		total += p;
	}
	const external = new Float64Array(n);
	const draw = () => {
		for (let i = 0; i < n; i++) {
			const u = random();
			let k = 0;
			while (k < table.length && u > table[k]) k++;
			external[i] = j * k;
		}
		return external;
	};
	return { network, draw };
}
