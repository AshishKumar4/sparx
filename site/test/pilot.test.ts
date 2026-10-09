// The browser's pilot against sparx: the same weights and the same flights, both in float64
// (site/lab/pilot.py record). Every spike must match, and the drone's path to rounding.
import { expect, test } from 'bun:test';
import { Pilot, type PilotModel } from '../src/engines/pilot';

const model: PilotModel = await Bun.file(new URL('../public/pilot/pilot.json', import.meta.url)).json();
const recorded: {
	flights: {
		box: number[];
		start: number[];
		targets: number[][];
		kicks: number[][];
		states: number[][];
		readout: number[][];
		spikes: number[][];
	}[];
} = await Bun.file(new URL('fixtures/pilot.json', import.meta.url)).json();

for (const [k, flight] of recorded.flights.entries()) {
	test(`flight ${k}: ${flight.targets.length} steps in a box ${flight.box[0] * 2} m wide`, () => {
		const pilot = new Pilot(model);
		pilot.state.set(flight.start);
		let state = 0;
		let readout = 0;
		let spikes = 0;
		for (let t = 0; t < flight.targets.length; t++) {
			for (let i = 0; i < 6; i++) pilot.state[i] += flight.kicks[t][i];
			pilot.step(flight.targets[t][0], flight.targets[t][1], flight.box[0], flight.box[1]);
			for (let i = 0; i < 6; i++) state = Math.max(state, Math.abs(pilot.state[i] - flight.states[t][i]));
			for (let i = 0; i < 2; i++) readout = Math.max(readout, Math.abs(pilot.readout[i] - flight.readout[t][i]));
			const fired = [...pilot.spikes.keys()].filter((i) => pilot.spikes[i]);
			expect(fired).toEqual(flight.spikes[t]);
			spikes += fired.length;
		}
		console.log(`flight ${k}: ${spikes} spikes all matched; largest difference ${state.toExponential(1)} in the drone's state, ${readout.toExponential(1)} in the readout`);
		expect(state).toBeLessThan(1e-9);
		expect(readout).toBeLessThan(1e-9);
	});
}
