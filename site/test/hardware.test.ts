// The browser's event camera against the racer's pixels in sparx, float64 (site/lab/fixtures.py).
import { expect, test } from 'bun:test';
import { EventCamera } from '../src/engines/events';

const { events } = await Bun.file(new URL('fixtures/sparx.json', import.meta.url)).json();

test('event camera pixels on wandering log brightness', () => {
	const camera = new EventCamera(events.seen[0].length, events.threshold);
	camera.reset(events.seen[0]);
	let count = 0;
	for (let t = 1; t < events.seen.length; t++) {
		count += camera.step(events.seen[t]);
		const on = [...camera.on.keys()].filter((p) => camera.on[p]);
		const off = [...camera.off.keys()].filter((p) => camera.off[p]);
		expect(on).toEqual(events.on[t - 1]);
		expect(off).toEqual(events.off[t - 1]);
	}
	console.log(`event camera: ${count} events matched over ${events.seen.length - 1} steps of ${events.seen[0].length} pixels`);
	expect(count).toBeGreaterThan(500);
});
