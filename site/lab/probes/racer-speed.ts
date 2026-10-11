// How long the browser's racer takes per step on the published network: the camera, the network and the
// car, on a circular track, after a second of warm-up. Run with bun from site/.
import { Racer, type RacerModel, Track } from '../../src/engines/racer';

const model: RacerModel = await Bun.file(new URL('../../public/racer/racer.json', import.meta.url)).json();
const n = model.world.points;
const points = Array.from({ length: n }, (_, k) => [8 * Math.cos((2 * Math.PI * k) / n), 8 * Math.sin((2 * Math.PI * k) / n)]);
const racer = new Racer(model, new Track(points, model.world));
racer.reset([points[0][0], points[0][1], Math.PI / 2, 0]);
for (let t = 0; t < 50; t++) racer.step();
const steps = 1000;
let network = 0;
const start = performance.now();
for (let t = 0; t < steps; t++) {
	racer.step();
	const before = performance.now();
	racer.network.step(racer.input);
	network += performance.now() - before;
}
const total = performance.now() - start - network;
console.log(`${(total / steps).toFixed(2)} ms a step in all, of which about ${(network / steps).toFixed(2)} ms the network (timed once more apart); a step is ${model.world.dt * 1000} ms of driving`);
