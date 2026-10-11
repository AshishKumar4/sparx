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
			// The car's x, y, heading and speed: each one's largest error over the drive, in units in the last place
			// of the largest value it takes.
			const error = [0, 0, 0, 0];
			const size = [0, 0, 0, 0];
			let active = 0;
			for (let t = 0; t < lap.cars.length; t++) {
				racer.step();
				expect([...racer.events.keys()].filter((i) => racer.events[i])).toEqual(lap.events[t]);
				const fired = [...racer.network.spikes.keys()].filter((i) => racer.network.spikes[i]);
				expect(fired).toEqual(lap.spikes[t]);
				for (let i = 0; i < 4; i++) {
					error[i] = Math.max(error[i], Math.abs(racer.car[i] - lap.cars[t][i]));
					size[i] = Math.max(size[i], Math.abs(lap.cars[t][i]));
				}
				active += fired.length;
			}
			car = Math.max(...error.map((e, i) => e / ulp(size[i])));
			console.log(`${name}, lap ${k}: ${active} unit outputs matched; the car within ${car.toFixed(1)} units in the last place`);
			expect(active).toBeGreaterThan(0);
			// sparx's float64 run itself moves by tens of units in the last place between machines (XLA's CPU
			// transcendentals differ by instruction set): measured 0.3 to 12.2 on armada, 0.3 to 24.0 on CI's runner.
			expect(car).toBeLessThan(32);
		});
	}
}

/** The spacing of float64 numbers at `x`'s magnitude: one unit in the last place. */
function ulp(x: number): number {
	const a = Math.abs(x);
	return a < 2 ** -1022 ? 2 ** -1074 : 2 ** (Math.floor(Math.log2(a)) - 52);
}
