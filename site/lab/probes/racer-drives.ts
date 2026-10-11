// The browser's own drives of the racer fixtures (test/fixtures/racer*.json): each step's readout membranes
// and car, written as JSON to the file named, for car_exact.py to hold against an exact car.
//
//     bun lab/probes/racer-drives.ts <out.json>
import { Racer, type RacerModel, Track } from '../../src/engines/racer';

const fixtures = new URL('../../test/fixtures/', import.meta.url);
const out: Record<string, { readout: number[][]; cars: number[][] }[]> = {};
for (const name of ['racer', 'racer-conv-lif-events', 'racer-conv-relu-frames']) {
	const recorded = await Bun.file(new URL(`${name}.json`, fixtures)).json();
	const model: RacerModel = recorded.model ?? (await Bun.file(new URL('../../public/racer/racer.json', import.meta.url)).json());
	out[name] = recorded.laps.map((lap: { points: number[][]; start: number[]; cars: number[][] }) => {
		const racer = new Racer(model, new Track(lap.points, model.world));
		racer.reset(lap.start);
		const readout: number[][] = [];
		const cars: number[][] = [];
		for (let t = 0; t < lap.cars.length; t++) {
			racer.step();
			readout.push(Array.from(racer.network.readout));
			cars.push(Array.from(racer.car));
		}
		return { readout, cars };
	});
}
await Bun.write(process.argv[2], JSON.stringify(out));
