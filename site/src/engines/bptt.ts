// Chapter 6's neuron: a LIFCell whose own spike feeds back to it with weight `w` (an autapse), run forward,
// then backpropagated by hand from its last membrane to every step's input. It is what jax.grad computes
// through sparx's LIFCell and sparx.spike (test/learning.test.ts).

export interface Unrolled {
	/** The membrane before each step's reset, and the spikes. */
	u: Float64Array;
	s: Uint8Array;
	/** d v_T / d x_t for every step t: how much each step's input moves the last membrane. */
	grad: Float64Array;
}

export function unroll(x: ArrayLike<number>, decay: number, w: number, surrogate: (x: number) => number, detach: boolean, threshold = 1): Unrolled {
	const T = x.length;
	const u = new Float64Array(T);
	const s = new Uint8Array(T);
	let v = 0;
	let spike = 0;
	for (let t = 0; t < T; t++) {
		u[t] = decay * v + x[t] + w * spike;
		spike = u[t] - threshold >= 0 ? 1 : 0;
		s[t] = spike;
		v = u[t] - threshold * spike;
	}
	const grad = new Float64Array(T);
	let gv = 1;
	let gs = 0;
	for (let t = T - 1; t >= 0; t--) {
		const slope = surrogate(u[t] - threshold);
		const gu = gv * (detach ? 1 : 1 - threshold * slope) + gs * slope;
		grad[t] = gu;
		gv = gu * decay;
		gs = gu * w;
	}
	return { u, s, grad };
}
