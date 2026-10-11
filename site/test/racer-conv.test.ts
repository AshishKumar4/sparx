// The browser's convolutional racer against sparx: small networks at their random start, one spiking on
// events and one graded on frames, with their laps recorded in float64 (site/lab/racer.py fixture). Every
// event and every unit's output must match, and the car's path to rounding.
import { expect, test } from 'bun:test';
import { Racer, type RacerModel, Track } from '../src/engines/racer';

interface Recorded {
	model: RacerModel;
	laps: { points: number[][]; start: number[]; cars: number[][]; events: number[][]; spikes: number[][] }[];
}

// Each fixture's bound on the car, in units in the last place. On the spiking fixture the browser measured
// 0.3 and 1.0 on armada and on CI's runner alike. On the graded one, sparx's own float64 run differs by up
// to 14.5 between those two machines (site/lab/probes/references.py), and the browser measured up to 9.5
// against armada's and 24.0 against the runner's.
const BOUND: Record<string, number> = { 'lif-events': 2, 'relu-frames': 32 };

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
			expect(car).toBeLessThan(BOUND[name]);
		});
	}
}

/** The spacing of float64 numbers at `x`'s magnitude: one unit in the last place. */
function ulp(x: number): number {
	const a = Math.abs(x);
	return a < 2 ** -1022 ? 2 ** -1074 : 2 ** (Math.floor(Math.log2(a)) - 52);
}
