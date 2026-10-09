// The browser's racer against sparx: the same weights, tracks and starts, both in float64
// (site/lab/racer.py record). Every event and every spike must match, and the car's path to rounding.
import { expect, test } from 'bun:test';
import { Racer, type RacerModel, Track } from '../src/engines/racer';

const model: RacerModel = await Bun.file(new URL('../public/racer/racer.json', import.meta.url)).json();
const recorded: {
	laps: { points: number[][]; start: number[]; cars: number[][]; readout: number[][]; events: number[][]; spikes: number[][] }[];
} = await Bun.file(new URL('fixtures/racer.json', import.meta.url)).json();

for (const [k, lap] of recorded.laps.entries()) {
	test(`lap ${k}: ${lap.cars.length} steps`, () => {
		const racer = new Racer(model, new Track(lap.points, model.world));
		racer.reset(lap.start);
		let car = 0;
		let events = 0;
		let spikes = 0;
		for (let t = 0; t < lap.cars.length; t++) {
			racer.step();
			const fired = [...racer.input.keys()].filter((i) => i < 2 * racer.pixels && racer.input[i]);
			expect(fired).toEqual(lap.events[t]);
			const spiked = [...racer.network.spikes.keys()].filter((i) => racer.network.spikes[i]);
			expect(spiked).toEqual(lap.spikes[t]);
			for (let i = 0; i < 4; i++) car = Math.max(car, Math.abs(racer.car[i] - lap.cars[t][i]));
			events += fired.length;
			spikes += spiked.length;
		}
		console.log(`lap ${k}: ${events} events and ${spikes} spikes matched; the car within ${car.toExponential(1)}`);
		expect(car).toBeLessThan(1e-9);
	});
}
