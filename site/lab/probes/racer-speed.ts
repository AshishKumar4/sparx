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
// The whole step, timed on the racer alone, and the network's part on a second copy fed the same inputs.
const inputs: Float64Array[] = [];
const start = performance.now();
for (let t = 0; t < steps; t++) {
	racer.step();
	inputs.push(Float64Array.from(racer.input));
}
const total = (performance.now() - start) / steps;
const twin = new Racer(model, new Track(points, model.world));
const before = performance.now();
for (const x of inputs) twin.network.step(x);
const network = (performance.now() - before) / steps;
console.log(`${total.toFixed(2)} ms a step in all, ${network.toFixed(2)} ms of it the network's; a step is ${model.world.dt * 1000} ms of driving`);
