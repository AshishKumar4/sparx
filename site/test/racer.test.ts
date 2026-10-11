// The browser's racer against sparx: the same weights, tracks and starts, both in float64
// (site/lab/racer.py record). Every event and every spike must match, and the car's path to rounding.
import { expect, test } from 'bun:test';
import { Racer, type RacerModel, Track } from '../src/engines/racer';
import { Distance, READOUT_LIMIT } from './parity';

const model: RacerModel = await Bun.file(new URL('../public/racer/racer.json', import.meta.url)).json();
const recorded: {
	laps: { points: number[][]; start: number[]; cars: number[][]; readout: number[][]; events: number[][]; spikes: number[][] }[];
} = await Bun.file(new URL('fixtures/racer.json', import.meta.url)).json();

for (const [k, lap] of recorded.laps.entries()) {
	test(`lap ${k}: ${lap.cars.length} steps`, () => {
		const racer = new Racer(model, new Track(lap.points, model.world));
		racer.reset(lap.start);
		const distance = new Distance();
		let events = 0;
		let spikes = 0;
		for (let t = 0; t < lap.cars.length; t++) {
			racer.step();
			const fired = [...racer.input.keys()].filter((i) => i < 2 * racer.pixels && racer.input[i]);
			expect(fired).toEqual(lap.events[t]);
			const spiked = [...racer.network.spikes.keys()].filter((i) => racer.network.spikes[i]);
			expect(spiked).toEqual(lap.spikes[t]);
			distance.add(racer.network.readout, lap.readout[t], racer.car, lap.cars[t]);
			events += fired.length;
			spikes += spiked.length;
		}
		// The car is held to its rounding bound by lab/car_exact.py, which the Site workflow runs after these
		// tests; here it is reported.
		console.log(`lap ${k}: ${events} events and ${spikes} spikes matched; readouts ${distance.readoutMaxRelative.toExponential(1)} ` +
			`max-relative, ${distance.readoutRmsRelative.toExponential(1)} RMS-relative; the car within ${distance.carUlps.toFixed(1)} units in the last place`);
		expect(distance.readoutMaxRelative).toBeLessThan(READOUT_LIMIT);
		// A step of the published network's 64 x 32 camera takes about 10 ms on a CI runner's core.
	}, 60_000);
}
