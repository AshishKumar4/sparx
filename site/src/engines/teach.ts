// One `sparx.dynamics.LIFCell` on weighted input spike trains, and the gradient of how far its spikes
// are from a target's, by backpropagation through time with a surrogate's slope standing in for the
// spike's. It is what `jax.grad` computes through `sparx.spike` (test/learning.test.ts).

export interface Problem {
	/** `[T][N]` input spikes, 0 or 1. */
	trains: number[][];
	/** Target spike times, in steps. */
	target: number[];
	decay: number;
	threshold: number;
	/** What an exponential filter of the spikes keeps each step: `exp(-1 / tau)`. */
	keep: number;
	surrogate: (x: number) => number;
}

export interface Pass {
	loss: number;
	grad: Float64Array;
	spikes: Uint8Array;
	v: Float64Array;
}

/** The loss, the mean squared difference of the filtered output and target spikes, and its gradient in `w`. */
export function teach({ trains, target, decay, threshold, keep, surrogate }: Problem, w: Float64Array): Pass {
	const T = trains.length;
	const N = w.length;
	const u = new Float64Array(T);
	const v = new Float64Array(T);
	const o = new Uint8Array(T);
	const f = new Float64Array(T);
	const g = new Float64Array(T);
	const wanted = new Uint8Array(T);
	for (const t of target) if (t < T) wanted[t] = 1;
	let before = 0;
	let fo = 0;
	let fg = 0;
	let loss = 0;
	for (let t = 0; t < T; t++) {
		let x = 0;
		for (let i = 0; i < N; i++) x += trains[t][i] * w[i];
		u[t] = decay * before + x;
		o[t] = u[t] - threshold >= 0 ? 1 : 0;
		v[t] = u[t] - threshold * o[t];
		before = v[t];
		fo = keep * fo + o[t];
		fg = keep * fg + wanted[t];
		f[t] = fo;
		g[t] = fg;
		loss += (fo - fg) ** 2;
	}
	loss /= T;
	const grad = new Float64Array(N);
	let df = 0;
	let dv = 0;
	for (let t = T - 1; t >= 0; t--) {
		df = (2 * (f[t] - g[t])) / T + keep * df;
		const dout = df - threshold * dv;
		const du = dout * surrogate(u[t] - threshold) + dv;
		for (let i = 0; i < N; i++) grad[i] += du * trains[t][i];
		dv = du * decay;
	}
	return { loss, grad, spikes: o, v: u };
}
