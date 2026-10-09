// The browser's neurons and Brunel network against sparx's float64 runs (site/lab/fixtures.py).
import { expect, test } from 'bun:test';
import { DeltaNetwork } from '../src/engines/brunel';
import { IZHIKEVICH_2003, Izhikevich, LeakyIntegrateAndFire, LIF, type Reset } from '../src/engines/neurons';

const fixtures = await Bun.file(new URL('fixtures/sparx.json', import.meta.url)).json();

function compare(v: number[], fired: number[], expected: { v: number[]; spikes: number[] }) {
	const spikes = fired.flatMap((s, t) => (s ? [t] : []));
	expect(spikes).toEqual(expected.spikes);
	return Math.max(...v.map((x, t) => Math.abs(x - expected.v[t])));
}

for (const reset of ['subtract', 'zero', 'none'] as Reset[]) {
	test(`LIFCell with reset ${reset}`, () => {
		const cell = new LIF(Math.exp(-1 / 12), 1, reset);
		const v: number[] = [];
		const fired = fixtures.neurons.drive.map((x: number) => {
			const s = cell.step(x);
			v.push(cell.v);
			return s;
		});
		const worst = compare(v, fired, fixtures.neurons.lif[reset]);
		console.log(`LIFCell ${reset}: ${fixtures.neurons.lif[reset].spikes.length} spikes matched, membrane within ${worst.toExponential(1)}`);
		expect(worst).toBeLessThan(1e-12);
	});
}

test('LeakyIntegrateAndFire on a current', () => {
	const { current, dt } = fixtures.neurons.physical;
	const cell = new LeakyIntegrateAndFire();
	const v: number[] = [];
	const fired = current.map((i: number) => {
		const s = cell.step(i, dt);
		v.push(cell.v);
		return s;
	});
	const worst = compare(v, fired, fixtures.neurons.physical);
	console.log(`LeakyIntegrateAndFire: ${fixtures.neurons.physical.spikes.length} spikes matched, within ${worst.toExponential(1)} mV`);
	expect(worst).toBeLessThan(1e-9);
});

for (const [name, [a, b, c, d]] of Object.entries(IZHIKEVICH_2003)) {
	test(`Izhikevich, ${name}`, () => {
		const { after, amplitude, dt, steps } = fixtures.neurons.izhikevich_current;
		const cell = new Izhikevich(a, b, c, d);
		const v: number[] = [];
		const fired = Array.from({ length: steps }, (_, t) => {
			const s = cell.step(t > after ? amplitude : 0, dt);
			v.push(cell.v);
			return s;
		});
		const worst = compare(v, fired, fixtures.neurons.izhikevich[name]);
		console.log(`Izhikevich ${name}: ${fixtures.neurons.izhikevich[name].spikes.length} spikes matched, within ${worst.toExponential(1)} mV`);
		expect(worst).toBeLessThan(1e-9);
	});
}

test("Brunel's network on sparx's edges and external input", () => {
	const b = fixtures.brunel;
	const delays = new Set(b.edges.delay);
	expect(delays.size).toBe(1);
	const network = new DeltaNetwork({
		excitatory: b.sizes.e,
		inhibitory: b.sizes.i,
		edges: { pre: Int32Array.from(b.edges.pre), post: Int32Array.from(b.edges.post), weight: Float64Array.from(b.edges.weight) },
		delay: Math.round(b.edges.delay[0] / b.dt),
		dt: b.dt,
		tau_m: 20,
		v_th: 20,
		v_reset: 10,
		t_ref: 2,
		e_l: 0,
	});
	let total = 0;
	for (let t = 0; t < b.steps; t++) {
		network.step(Float64Array.from(b.external[t]));
		const fired = [...network.fired.keys()].filter((i) => network.fired[i]);
		expect(fired).toEqual(b.spikes[t]);
		total += fired.length;
	}
	console.log(`Brunel, ${network.size} neurons, ${b.edges.pre.length} synapses, ${b.steps} steps: ${total} spikes matched`);
	expect(total).toBeGreaterThan(100);
});
