// `sparx.dynamics.PairSTDP`, NEST's stdp_synapse: traces on the neurons, weights on the edges.

export interface Rule {
	tau_plus: number;
	tau_minus: number;
	lambda: number;
	alpha: number;
	mu_plus: number;
	mu_minus: number;
	w_max: number;
}

export class PairSTDP {
	readonly pre: Float64Array;
	readonly post: Float64Array;

	constructor(
		public rule: Rule,
		pres: number,
		posts: number,
	) {
		this.pre = new Float64Array(pres);
		this.post = new Float64Array(posts);
	}

	/** One step of `dt` ms: traces decay, a postsynaptic arrival potentiates, then a presynaptic spike depresses. */
	step(weights: Float64Array, preSpikes: Uint8Array, postArrivals: Uint8Array, pre: Int32Array, post: Int32Array, dt: number): void {
		const r = this.rule;
		const keepPre = Math.exp(-dt / r.tau_plus);
		const keepPost = Math.exp(-dt / r.tau_minus);
		for (let i = 0; i < this.pre.length; i++) this.pre[i] *= keepPre;
		for (let j = 0; j < this.post.length; j++) this.post[j] *= keepPost;
		for (let e = 0; e < weights.length; e++) {
			let w = weights[e] / r.w_max;
			if (postArrivals[post[e]]) w = Math.min(w + r.lambda * (1 - w) ** r.mu_plus * this.pre[pre[e]], 1);
			if (preSpikes[pre[e]]) w = Math.max(w - r.alpha * r.lambda * w ** r.mu_minus * this.post[post[e]], 0);
			weights[e] = w * r.w_max;
		}
		for (let i = 0; i < this.pre.length; i++) this.pre[i] += preSpikes[i];
		for (let j = 0; j < this.post.length; j++) this.post[j] += postArrivals[j];
	}
}
