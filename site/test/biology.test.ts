// The browser's conductance neurons and Hodgkin-Huxley against sparx's float64 runs (site/lab/fixtures.py).
import { expect, test } from 'bun:test';
import { HodgkinHuxley, LeakyIntegrateAndFire, PointNeuron, type Receptor } from '../src/engines/biology';

const { biology } = await Bun.file(new URL('fixtures/sparx.json', import.meta.url)).json();
const { dt } = biology;

function worst(got: number[], expected: number[]): number {
	return Math.max(...got.map((x, t) => Math.abs(x - expected[t])));
}

const times = (fired: number[]) => fired.flatMap((s, t) => (s ? [t] : []));

test('PointNeuron with a current synapse and AMPA, GABA-A and NMDA conductances', () => {
	const { receptors, current, arrivals, v: expected, spikes } = biology.point;
	const kinds = Object.fromEntries(
		Object.entries(receptors as Record<string, [number, Receptor['kind']]>).map(([name, [tau, kind]]) => [name, { tau, kind }]),
	);
	const point = new PointNeuron(new LeakyIntegrateAndFire(), kinds);
	const v: number[] = [];
	const fired = current.map((pA: number, t: number) => {
		const s = point.step(pA, Object.fromEntries(Object.keys(kinds).map((name) => [name, arrivals[name][t]])), dt);
		v.push(point.cell.v);
		return s;
	});
	expect(times(fired)).toEqual(spikes);
	const error = worst(v, expected);
	console.log(`PointNeuron: ${spikes.length} spikes matched, membrane within ${error.toExponential(1)} mV`);
	expect(spikes.length).toBeGreaterThan(10);
	expect(error).toBeLessThan(1e-11);
});

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
