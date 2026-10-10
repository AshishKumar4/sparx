import { HodgkinHuxley } from '../../src/engines/biology';
// Slider-like: the current changes by 10 pA every 50 ms, up from 0 to 1500 and back down.
const run = (levels: number[]) => {
	const cell = new HodgkinHuxley();
	const out: string[] = [];
	let firing = false;
	for (const pA of levels) {
		let n = 0;
		for (let t = 0; t < 2000; t++) n += cell.step(pA, 0.1); // 200 ms per level
		const now = n >= 2;
		if (now !== firing) out.push(`${now ? 'starts' : 'stops'} at ${pA} pA (${n * 5} Hz)`);
		firing = now;
	}
	return out.join(', ');
};
const up = Array.from({ length: 151 }, (_, k) => 10 * k);
console.log('up:', run(up));
console.log('up then down:', run([...up, ...up.slice().reverse()]));
// From rest, a step straight to each current.
const steps: string[] = [];
for (const pA of [300, 500, 600, 620, 640, 650, 700, 800, 900, 950, 1000]) {
	const cell = new HodgkinHuxley();
	let first = 0, last = 0;
	for (let t = 0; t < 10000; t++) { const s = cell.step(pA, 0.1); if (t < 5000) first += s; else last += s; }
	steps.push(`${pA}: ${first} spikes in the first 500 ms, ${last} in the next`);
}
console.log(steps.join('\n'));
// Peak conductances during firing at 800 pA.
const cell = new HodgkinHuxley();
let peak = [0, 0], low = 0;
for (let t = 0; t < 10000; t++) { cell.step(800, 0.1); if (t > 2000) { peak = peak.map((g, k) => Math.max(g, cell.conductances[k])); low = Math.min(low, cell.v); } }
console.log(`800 pA: peak sodium ${peak[0].toFixed(0)} nS, potassium ${peak[1].toFixed(0)} nS, lowest v ${low.toFixed(1)} mV`);
