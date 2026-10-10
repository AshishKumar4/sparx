// The browser's convolutional racer against sparx: small networks at their random start, one spiking on
// events and one graded on frames, with their laps recorded in float64 (site/lab/racer.py fixture). Every
// event and every unit's output must match, and the car's path to rounding.
import { expect, test } from 'bun:test';
import { Racer, type RacerModel, Track } from '../src/engines/racer';

interface Recorded {
	model: RacerModel;
	laps: { points: number[][]; start: number[]; cars: number[][]; events: number[][]; spikes: number[][] }[];
}

for (const name of ['lif-events', 'relu-frames']) {
	const recorded: Recorded = await Bun.file(new URL(`fixtures/racer-conv-${name}.json`, import.meta.url)).json();
	for (const [k, lap] of recorded.laps.entries()) {
		test(`${name}, lap ${k}: ${lap.cars.length} steps`, () => {
			const racer = new Racer(recorded.model, new Track(lap.points, recorded.model.world));
			racer.reset(lap.start);
			let car = 0;
			let active = 0;
			for (let t = 0; t < lap.cars.length; t++) {
				racer.step();
				expect([...racer.events.keys()].filter((i) => racer.events[i])).toEqual(lap.events[t]);
				const fired = [...racer.network.spikes.keys()].filter((i) => racer.network.spikes[i]);
				expect(fired).toEqual(lap.spikes[t]);
				for (let i = 0; i < 4; i++) car = Math.max(car, Math.abs(racer.car[i] - lap.cars[t][i]));
				active += fired.length;
			}
			console.log(`${name}, lap ${k}: ${active} unit outputs matched; the car within ${car.toExponential(1)}`);
			expect(active).toBeGreaterThan(0);
			expect(car).toBeLessThan(1e-9);
		});
	}
}
