// `sparx.nn.delay_kernel` and the delays page's toy: inputs that each fire once, delayed through a
// `DelayedDense` onto one leaky integrator, and the gradient of its peak in each delay.

/** Each lag's weight for a synapse delayed by `delay` steps: a softmax of the Gaussian's log-density. */
export function kernel(delay: number, maxDelay: number, sigma: number): Float64Array {
	const center = Math.min(Math.max(delay, 0), maxDelay);
	const out = new Float64Array(maxDelay + 1);
	if (sigma === 0) {
		out[Math.round(center)] = 1;
		return out;
	}
	let top = -Infinity;
	for (let k = 0; k <= maxDelay; k++) {
		out[k] = -0.5 * ((k - center) / sigma) ** 2;
		top = Math.max(top, out[k]);
	}
	let sum = 0;
	for (let k = 0; k <= maxDelay; k++) {
		out[k] = Math.exp(out[k] - top);
		sum += out[k];
	}
	for (let k = 0; k <= maxDelay; k++) out[k] /= sum;
	return out;
}

export interface Toy {
	/** When each input fires, in steps. */
	times: number[];
	weight: number;
	maxDelay: number;
	steps: number;
	/** The readout's decay per step. */
	decay: number;
}

/** The readout's membrane over time, its peak's step, and the gradient of minus the peak in each delay. */
export function peak(toy: Toy, delays: number[], sigma: number) {
	const { times, weight, maxDelay, steps, decay } = toy;
	const kernels = delays.map((d) => kernel(d, maxDelay, sigma));
	const y = new Float64Array(steps);
	for (const [i, s] of times.entries()) {
		for (let k = 0; k <= maxDelay && s + k < steps; k++) y[s + k] += weight * kernels[i][k];
	}
	const v = new Float64Array(steps);
	let at = 0;
	for (let t = 0; t < steps; t++) {
		v[t] = (t ? decay * v[t - 1] : 0) + y[t];
		if (v[t] > v[at]) at = t;
	}
	const dy = new Float64Array(steps);
	for (let t = at, back = -1; t >= 0; t--, back *= decay) dy[t] = back;
	const grad = delays.map((d, i) => {
		if (sigma === 0) return 0;
		const center = Math.min(Math.max(d, 0), maxDelay);
		const g = kernels[i];
		const dg = Array.from(g, (_, k) => (times[i] + k < steps ? weight * dy[times[i] + k] : 0));
		let mean = 0;
		for (let k = 0; k <= maxDelay; k++) mean += g[k] * ((k - center) / sigma ** 2);
		let sum = 0;
		for (let k = 0; k <= maxDelay; k++) sum += dg[k] * g[k] * ((k - center) / sigma ** 2 - mean);
		return sum;
	});
	return { v, at, loss: -v[at], grad, kernels };
}
