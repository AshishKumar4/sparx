// sparx.learn's recurrent network for e-prop: a layer of LIF neurons with a detached reset,
// `x_t = u_t w_in + z_{t-1} w_rec`, and a leaky readout `y_t = κ y_{t-1} + z_t w_out + b_out`, with the
// loss `Σ_t ½ |y_t - y*_t|²`. `gradients` computes e-prop's gradient online (sparx.learn.eprop) and BPTT's
// by the backward pass (jax.grad of sparx.learn.bptt_loss); test/learning.test.ts holds both to sparx.

export interface Params {
	inputs: number;
	size: number;
	outputs: number;
	/** `[inputs][size]`, `[size][size]`, `[size][outputs]`, `[outputs]`, row-major. */
	wIn: Float64Array;
	wRec: Float64Array;
	wOut: Float64Array;
	bOut: Float64Array;
}

export interface Setup {
	/** The membrane's and the readout's decay per step. */
	beta: number;
	kappa: number;
	threshold: number;
	surrogate: (x: number) => number;
}

export interface Pass {
	loss: number;
	y: Float64Array<ArrayBufferLike>;
	z: Uint8Array;
	eprop: Params;
	bptt: Params;
}

const zeros = (p: Params): Params => ({
	...p,
	wIn: new Float64Array(p.wIn.length),
	wRec: new Float64Array(p.wRec.length),
	wOut: new Float64Array(p.wOut.length),
	bOut: new Float64Array(p.bOut.length),
});

/** One pass over `u` `[T][inputs]` toward `target` `[T][outputs]`: the loss, the readout `[T * outputs]`, the
 * spikes `[T * size]`, and both gradients. */
export function gradients(p: Params, setup: Setup, u: ArrayLike<number>[], target: ArrayLike<number>[]): Pass {
	const { inputs: I, size: N, outputs: O } = p;
	const { beta, kappa, threshold, surrogate } = setup;
	const T = u.length;
	const pre = I + N;
	const uu = new Float64Array(T * N);
	const z = new Uint8Array(T * N);
	const y = new Float64Array(T * O);
	const dy = new Float64Array(T * O);
	const v = new Float64Array(N);
	const yNow = new Float64Array(O);
	// e-prop's state: each synapse's filtered presynaptic activity, and its eligibility trace filtered by κ.
	const presyn = new Float64Array(N * pre);
	const filtered = new Float64Array(N * pre);
	const zBar = new Float64Array(N);
	const ep = zeros(p);
	let leak = 0;
	let loss = 0;
	for (let t = 0; t < T; t++) {
		const zPrev = t ? z.subarray((t - 1) * N, t * N) : new Uint8Array(N);
		const zNow = z.subarray(t * N, (t + 1) * N);
		for (let j = 0; j < N; j++) {
			let x = 0;
			for (let i = 0; i < I; i++) x += u[t][i] * p.wIn[i * N + j];
			for (let k = 0; k < N; k++) x += zPrev[k] * p.wRec[k * N + j];
			const now = beta * v[j] + x;
			uu[t * N + j] = now;
			zNow[j] = now - threshold >= 0 ? 1 : 0;
			v[j] = now - threshold * zNow[j];
		}
		for (let o = 0; o < O; o++) {
			let jump = p.bOut[o];
			for (let j = 0; j < N; j++) jump += zNow[j] * p.wOut[j * O + o];
			yNow[o] = kappa * yNow[o] + jump;
			y[t * O + o] = yNow[o];
			const e = yNow[o] - target[t][o];
			dy[t * O + o] = e;
			loss += 0.5 * e * e;
		}
		leak = kappa * leak + 1;
		for (let j = 0; j < N; j++) {
			zBar[j] = kappa * zBar[j] + zNow[j];
			const slope = surrogate(uu[t * N + j] - threshold);
			let signal = 0;
			for (let o = 0; o < O; o++) signal += dy[t * O + o] * p.wOut[j * O + o];
			for (let q = 0; q < pre; q++) {
				const a = q < I ? u[t][q] : zPrev[q - I];
				const s = j * pre + q;
				presyn[s] = beta * presyn[s] + a;
				filtered[s] = kappa * filtered[s] + slope * presyn[s];
				const g = signal * filtered[s];
				if (q < I) ep.wIn[q * N + j] += g;
				else ep.wRec[(q - I) * N + j] += g;
			}
			for (let o = 0; o < O; o++) ep.wOut[j * O + o] += zBar[j] * dy[t * O + o];
		}
		for (let o = 0; o < O; o++) ep.bOut[o] += leak * dy[t * O + o];
	}
	const bp = zeros(p);
	const gy = new Float64Array(O);
	const gu = new Float64Array(N);
	const guNext = new Float64Array(N);
	for (let t = T - 1; t >= 0; t--) {
		for (let o = 0; o < O; o++) {
			gy[o] = dy[t * O + o] + (t < T - 1 ? kappa * gy[o] : 0);
			bp.bOut[o] += gy[o];
		}
		const zNow = z.subarray(t * N, (t + 1) * N);
		for (let j = 0; j < N; j++) {
			let gz = 0;
			for (let o = 0; o < O; o++) gz += gy[o] * p.wOut[j * O + o];
			if (t < T - 1) for (let k = 0; k < N; k++) gz += guNext[k] * p.wRec[j * N + k];
			const gv = t < T - 1 ? beta * guNext[j] : 0;
			gu[j] = gv + gz * surrogate(uu[t * N + j] - threshold);
			for (let o = 0; o < O; o++) bp.wOut[j * O + o] += zNow[j] * gy[o];
		}
		for (let j = 0; j < N; j++) {
			for (let i = 0; i < I; i++) bp.wIn[i * N + j] += u[t][i] * gu[j];
			if (t) for (let k = 0; k < N; k++) bp.wRec[k * N + j] += z[(t - 1) * N + k] * gu[j];
		}
		guNext.set(gu);
	}
	return { loss, y, z, eprop: ep, bptt: bp };
}
