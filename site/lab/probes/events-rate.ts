import { EventCamera } from '../../src/engines/events';
// The widget's scene and stepping (src/widgets/events.ts), without the DOM: a fan turning `speed` times a second,
// stepped each millisecond.
import { HEIGHT, WIDTH, scene } from '../../src/widgets/events';
const pixels = WIDTH * HEIGHT;
const seen = new Float64Array(pixels);
const rows: string[] = [];
for (const threshold of [0.15, 0.075, 0.3]) {
	for (const speed of [0, 0.5, 1, 2, 3]) {
		const camera = new EventCamera(pixels, threshold);
		camera.reset(scene(0, seen));
		let n = 0, t = 0;
		const touched = new Uint8Array(pixels);
		for (let k = 0; k < 1100; k++) {
			t += 0.001;
			const sent = camera.step(scene(2 * Math.PI * speed * t, seen));
			if (k >= 100) { n += sent; for (let p = 0; p < pixels; p++) if (camera.on[p] || camera.off[p]) touched[p] = 1; }
		}
		rows.push(`threshold ${threshold}, ${speed} turns/s: ${n} events/s, ${(100 * n / (pixels * 1000)).toFixed(2)}%, ${touched.reduce((a, b) => a + b, 0)} pixels ever fired`);
	}
}
console.log(rows.join('\n'));
