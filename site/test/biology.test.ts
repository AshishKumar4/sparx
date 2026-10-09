// The browser's conductance neurons and Hodgkin-Huxley against sparx's float64 runs (site/lab/fixtures.py).
import { expect, test } from 'bun:test';
import { HodgkinHuxley, LeakyIntegrateAndFire, PointNeuron, type Receptor } from '../src/engines/biology';

const { biology } = await Bun.file(new URL('fixtures/sparx.json', import.meta.url)).json();
const { dt } = biology;

function worst(got: number[], expected: number[]): number {
	return Math.max(...got.map((x, t) => Math.abs(x - expected[t])));
}

const times = (fired: number[]) => fired.flatMap((s, t) => (s ? [t] : []));

for (const hold of ['mean', 'start'] as const) {
	test(`PointNeuron with a current synapse and AMPA, GABA-A and NMDA conductances, hold="${hold}"`, () => {
		const { receptors, current, arrivals } = biology.point;
		const { v: expected, spikes } = hold === 'mean' ? biology.point : biology.point.start;
		const kinds = Object.fromEntries(
			Object.entries(receptors as Record<string, [number, Receptor['kind']]>).map(([name, [tau, kind]]) => [name, { tau, kind }]),
		);
		const point = new PointNeuron(new LeakyIntegrateAndFire(), kinds, hold);
		const v: number[] = [];
		const fired = current.map((pA: number, t: number) => {
			const s = point.step(pA, Object.fromEntries(Object.keys(kinds).map((name) => [name, arrivals[name][t]])), dt);
			v.push(point.cell.v);
			return s;
		});
		expect(times(fired)).toEqual(spikes);
		const error = worst(v, expected);
		console.log(`PointNeuron, hold ${hold}: ${spikes.length} spikes matched, membrane within ${error.toExponential(1)} mV`);
		expect(spikes.length).toBeGreaterThan(10);
		expect(error).toBeLessThan(1e-11);
	});
}

test('LeakyIntegrateAndFire on held AMPA, GABA-A and NMDA conductances', () => {
	const { current } = biology.point;
	const { conductance, v: expected, spikes } = biology.held;
	const cell = new LeakyIntegrateAndFire();
	const names = Object.keys(conductance);
	const v: number[] = [];
	const fired = current.map((pA: number, t: number) => {
		const s = cell.step({ current: pA, currents: [], conductance: names.map((name) => [name, conductance[name][t]]), jump: 0 }, dt);
		v.push(cell.v);
		return s;
	});
	expect(times(fired)).toEqual(spikes);
	const error = worst(v, expected);
	console.log(`LeakyIntegrateAndFire on conductances: ${spikes.length} spikes matched, membrane within ${error.toExponential(1)} mV`);
	expect(spikes.length).toBeGreaterThan(10);
	expect(error).toBeLessThan(1e-11);
});

test('HodgkinHuxley through its onset of firing', () => {
	const { current, spikes, v: expected, m, h, n } = biology.hodgkin;
	const cell = new HodgkinHuxley();
	const trace: Record<string, number[]> = { v: [], m: [], h: [], n: [] };
	let peak = [0, 0];
	const fired = current.map((pA: number) => {
		const s = cell.step(pA, dt);
		for (const key of ['v', 'm', 'h', 'n'] as const) trace[key].push(cell[key]);
		peak = peak.map((g, k) => Math.max(g, cell.conductances[k]));
		return s;
	});
	expect(times(fired)).toEqual(spikes);
	const error = worst(trace.v, expected);
	const gates = Math.max(worst(trace.m, m), worst(trace.h, h), worst(trace.n, n));
	console.log(
		`HodgkinHuxley: ${spikes.length} spikes matched, membrane within ${error.toExponential(1)} mV, gates within ${gates.toExponential(1)}; peak sodium ${peak[0].toFixed(0)} nS, potassium ${peak[1].toFixed(0)} nS`,
	);
	expect(spikes.length).toBeGreaterThan(5);
	expect(error).toBeLessThan(1e-8);
	expect(gates).toBeLessThan(1e-10);
});

test('a background GABA-A conductance shrinks every input, the chapter figure at -60 mV', () => {
	const peak = (name: string, kind: Receptor['kind'], weight: number, background: number) => {
		const point = new PointNeuron(new LeakyIntegrateAndFire(20, 200, -60, Number.POSITIVE_INFINITY), { [name]: { tau: 5, kind } });
		const held: [string, number][] = [['gaba_a', background]];
		const current = 10 * (-60 + 60) + background * (-60 + 80);
		point.cell.v = -60;
		let most = 0;
		for (let t = 0; t < 2000; t++) {
			point.step(current, t === 0 ? { [name]: weight } : {}, 0.1, held);
			if (Math.abs(point.cell.v + 60) > Math.abs(most)) most = point.cell.v + 60;
		}
		return most;
	};
	const shown = [0, 40].map((g) => [peak('ex', 'current', 180, g), peak('ampa', 'conductance', 3, g), peak('gaba_a', 'conductance', 3, g)]);
	console.log(`peaks at -60 mV, background 0 nS: ${shown[0].map((x) => x.toFixed(2)).join(', ')} mV; 40 nS: ${shown[1].map((x) => x.toFixed(2)).join(', ')} mV`);
	for (let k = 0; k < 3; k++) expect(Math.abs(shown[1][k])).toBeLessThan(Math.abs(shown[0][k]) / 2);
});
