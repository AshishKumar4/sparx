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
		// The readout membranes against sparx's: the largest difference and the RMS difference, each relative
		// to the readouts' own size over the drive.
		let worst = 0;
		let squares = 0;
		let scale = 0;
		let norm = 0;
		for (let t = 0; t < lap.cars.length; t++) {
			racer.step();
			const fired = [...racer.input.keys()].filter((i) => i < 2 * racer.pixels && racer.input[i]);
			expect(fired).toEqual(lap.events[t]);
			const spiked = [...racer.network.spikes.keys()].filter((i) => racer.network.spikes[i]);
			expect(spiked).toEqual(lap.spikes[t]);
			for (let i = 0; i < 4; i++) car = Math.max(car, Math.abs(racer.car[i] - lap.cars[t][i]));
			for (let i = 0; i < 2; i++) {
				const d = racer.network.readout[i] - lap.readout[t][i];
				worst = Math.max(worst, Math.abs(d));
				scale = Math.max(scale, Math.abs(lap.readout[t][i]));
				squares += d * d;
				norm += lap.readout[t][i] ** 2;
			}
			events += fired.length;
			spikes += spiked.length;
		}
		const maxRelative = worst / scale;
		const rmsRelative = Math.sqrt(squares / norm);
		console.log(`lap ${k}: ${events} events and ${spikes} spikes matched; the car within ${car.toExponential(1)}; ` +
			`readouts ${maxRelative.toExponential(1)} max-relative, ${rmsRelative.toExponential(1)} RMS-relative`);
		expect(car).toBeLessThan(5e-14);
		expect(maxRelative).toBeLessThan(1e-15);
		// A step of the published network's 64 x 32 camera takes about 10 ms on a CI runner's core.
	}, 60_000);
}
