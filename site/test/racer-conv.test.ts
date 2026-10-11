// The browser's convolutional racer against sparx: a small spiking network on events at its random start,
// with its laps recorded in float64 (site/lab/racer.py fixture). Every
// event and every unit's output must match, the readouts to 1e-15 of their size, and the car's path to rounding.
import { expect, test } from 'bun:test';
import { Racer, type RacerModel, Track } from '../src/engines/racer';
import { Distance } from './parity';

interface Recorded {
	model: RacerModel;
	laps: { points: number[][]; start: number[]; cars: number[][]; readout: number[][]; events: number[][]; spikes: number[][] }[];
}

for (const name of ['lif-events']) {
	const recorded: Recorded = await Bun.file(new URL(`fixtures/racer-conv-${name}.json`, import.meta.url)).json();
	for (const [k, lap] of recorded.laps.entries()) {
		test(`${name}, lap ${k}: ${lap.cars.length} steps`, () => {
			const racer = new Racer(recorded.model, new Track(lap.points, recorded.model.world));
			racer.reset(lap.start);
			const distance = new Distance();
			let active = 0;
			for (let t = 0; t < lap.cars.length; t++) {
				racer.step();
				expect([...racer.events.keys()].filter((i) => racer.events[i])).toEqual(lap.events[t]);
				const fired = [...racer.network.spikes.keys()].filter((i) => racer.network.spikes[i]);
				expect(fired).toEqual(lap.spikes[t]);
				distance.add(racer.network.readout, lap.readout[t], racer.car, lap.cars[t]);
				active += fired.length;
			}
			// The car is held to its rounding bound by lab/car_exact.py, which the Site workflow runs after these
			// tests; here it is reported.
			console.log(`${name}, lap ${k}: ${active} unit outputs matched; readouts ${distance.readoutMaxRelative.toExponential(1)} ` +
				`max-relative; the car within ${distance.carUlps.toFixed(1)} units in the last place`);
			expect(distance.readoutMaxRelative).toBeLessThan(1e-15);
			expect(active).toBeGreaterThan(0);
		});
	}
}
