// The browser's learning engines against sparx in float64 (site/lab/fixtures.py): the surrogate
// gradient of the teach page's neuron, the delays page's kernels and gradient, and PairSTDP's weights.
import { expect, test } from 'bun:test';
import { unroll } from '../src/engines/bptt';
import { kernel, peak } from '../src/engines/delays';
import { gradients, type Params } from '../src/engines/eprop';
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

test('backpropagation through an autapse LIFCell', () => {
	if (!fixtures.bptt) throw new Error('fixtures.bptt is missing: run site/lab/fixtures.py --only bptt');
	const b = fixtures.bptt;
	for (const c of b.cases) {
		const derivative = surrogates().find((s) => s.name === c.surrogate)?.derivative as (x: number) => number;
		const out = unroll(b.x, Math.exp(-1 / b.tau), c.w, derivative, c.detach);
		expect([...out.s.keys()].filter((t) => out.s[t])).toEqual(c.spikes);
		const scale = Math.max(...c.grad.map(Math.abs));
		const worst = Math.max(...c.grad.map((g: number, t: number) => Math.abs(g - out.grad[t])));
		console.log(`autapse w=${c.w} ${c.surrogate}${c.detach ? ' detached' : ''}: ${c.spikes.length} spikes, gradient within ${worst.toExponential(1)} of a largest ${scale.toExponential(1)}`);
		expect(worst).toBeLessThan(1e-10 * Math.max(scale, 1));
	}
});

test('e-prop and BPTT through a recurrent LIF layer', () => {
	if (!fixtures.eprop) throw new Error('fixtures.eprop is missing: run site/lab/fixtures.py --only eprop');
	const e = fixtures.eprop;
	const flat = (a: number[] | number[][]) => Float64Array.from((a as number[][]).flat ? (a as number[][]).flat() : (a as number[]));
	const params: Params = {
		inputs: e.params.w_in.length,
		size: e.params.w_rec.length,
		outputs: e.params.b_out.length,
		wIn: flat(e.params.w_in),
		wRec: flat(e.params.w_rec),
		wOut: flat(e.params.w_out),
		bOut: flat(e.params.b_out),
	};
	const triangle = surrogates().find((s) => s.name === 'Triangle')?.derivative as (x: number) => number;
	const pass = gradients(params, { beta: Math.exp(-1 / e.tau), kappa: Math.exp(-1 / e.readout_tau), threshold: 1, surrogate: (x) => 0.3 * triangle(x) }, e.inputs, e.target);
	expect(Math.abs(pass.loss - e.loss)).toBeLessThan(1e-10);
	for (const [rule, mine] of [['eprop', pass.eprop], ['bptt', pass.bptt]] as const) {
		let worst = 0;
		for (const [key, name] of [['wIn', 'w_in'], ['wRec', 'w_rec'], ['wOut', 'w_out'], ['bOut', 'b_out']] as const) {
			const expected = flat(e[rule][name]);
			for (let k = 0; k < expected.length; k++) worst = Math.max(worst, Math.abs(mine[key][k] - expected[k]));
		}
		console.log(`${rule}: every gradient within ${worst.toExponential(1)} of sparx's`);
		expect(worst).toBeLessThan(1e-10);
	}
});
