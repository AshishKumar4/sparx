// The browser's learning engines against sparx in float64 (site/lab/fixtures.py): the surrogate
// gradient of the teach page's neuron, the delays page's kernels and gradient, and PairSTDP's weights.
import { expect, test } from 'bun:test';
import { kernel, peak } from '../src/engines/delays';
import { delta, latency } from '../src/engines/encode';
import { PairSTDP } from '../src/engines/stdp';
import { surrogates } from '../src/engines/surrogate';
import { teach } from '../src/engines/teach';

const fixtures = await Bun.file(new URL('fixtures/sparx.json', import.meta.url)).json();
const largest = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.max(...Array.from(a, (x, i) => Math.abs(x - b[i])));

for (const [name, expected] of Object.entries<{ loss: number; grad: number[] }>(fixtures.teach.cases)) {
	test(`surrogate gradient through a LIF neuron, ${name}`, () => {
		const t = fixtures.teach;
		const surrogate = surrogates().find((s) => s.name === name)?.derivative as (x: number) => number;
		const pass = teach({ trains: t.trains, target: t.target, decay: Math.exp(-1 / t.tau), threshold: 1, keep: Math.exp(-1 / t.filter_tau), surrogate }, Float64Array.from(t.w));
		const worst = largest(pass.grad, expected.grad);
		const scale = Math.max(...expected.grad.map(Math.abs));
		console.log(`${name}: loss ${pass.loss.toFixed(6)} (sparx ${expected.loss.toFixed(6)}), gradient within ${worst.toExponential(1)} of a largest entry ${scale.toExponential(1)}`);
		expect(Math.abs(pass.loss - expected.loss)).toBeLessThan(1e-12);
		expect(worst).toBeLessThan(1e-10 * scale);
		expect(scale).toBeGreaterThan(0);
	});
}

for (const c of fixtures.delays.cases) {
	test(`DelayedDense delays, read ${c.read_at === null ? 'at the peak' : `at step ${c.read_at}`}, sigma ${c.sigma}`, () => {
		const d = fixtures.delays;
		const toy = { times: d.x_times, weight: Math.fround(d.weight), maxDelay: c.max_delay, steps: d.steps, decay: Math.exp(-1 / d.tau) };
		for (const [i, delay] of c.delay.entries()) expect(largest(kernel(delay, c.max_delay, c.sigma), c.kernel[i])).toBeLessThan(1e-14);
		const out = peak(toy, c.delay, c.sigma, c.read_at ?? undefined);
		const worst = largest(out.grad, c.grad);
		console.log(`read ${c.read_at ?? 'at peak'}, sigma ${c.sigma}: loss ${out.loss.toFixed(6)} (sparx ${c.loss.toFixed(6)}), gradient within ${worst.toExponential(1)}`);
		expect(Math.abs(out.loss - c.loss)).toBeLessThan(1e-12);
		expect(worst).toBeLessThan(1e-12);
	});
}

for (const [name, c] of Object.entries<{ mu: number; weights: number[][]; final: number[] }>(fixtures.stdp.cases)) {
	test(`PairSTDP, ${name}`, () => {
		const s = fixtures.stdp;
		const rule = { tau_plus: s.rule.tau_plus, tau_minus: s.rule.tau_minus, lambda: s.rule.lambda, alpha: s.rule.alpha, mu_plus: c.mu, mu_minus: c.mu, w_max: s.rule.w_max };
		const pre = Int32Array.from(s.pre);
		const post = Int32Array.from(s.post);
		const stdp = new PairSTDP(rule, 6, 2);
		const w = new Float64Array(pre.length).fill(0.5);
		let worst = 0;
		for (let t = 0; t < s.pre_spikes.length; t++) {
			const ps = new Uint8Array(6);
			const qs = new Uint8Array(2);
			for (const i of s.pre_spikes[t]) ps[i] = 1;
			for (const j of s.post_spikes[t]) qs[j] = 1;
			stdp.step(w, ps, qs, pre, post, s.dt);
			if (t % 50 === 0) worst = Math.max(worst, largest(w, c.weights[t / 50]));
		}
		worst = Math.max(worst, largest(w, c.final));
		console.log(`PairSTDP ${name}: 12 weights over ${s.pre_spikes.length} steps within ${worst.toExponential(1)}`);
		expect(worst).toBeLessThan(1e-12);
	});
}

test('LatencyEncoder and DeltaEncoder', () => {
	if (!fixtures.encoders) throw new Error('fixtures.encoders is missing: run site/lab/fixtures.py --only encoders');
	const e = fixtures.encoders;
	for (const [steps, expected] of Object.entries<number[][]>(e.latency)) expect(latency(e.values, Number(steps))).toEqual(expected);
	for (const [threshold, expected] of Object.entries<number[]>(e.delta)) expect(delta(e.signal, Number(threshold))).toEqual(expected);
});
