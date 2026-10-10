import { HodgkinHuxley } from '../../src/engines/biology';
for (const pA of [620, 625, 630, 635, 640]) {
	const cell = new HodgkinHuxley();
	let first = 0, last = 0;
	for (let t = 0; t < 10000; t++) { const s = cell.step(pA, 0.1); if (t < 5000) first += s; else last += s; }
	console.log(`from rest ${pA}: ${first} then ${last} spikes per 500 ms`);
}
// Firing at 800, lowered 10 pA every 200 ms: the rate at each level until it stops.
const cell = new HodgkinHuxley();
for (let t = 0; t < 5000; t++) cell.step(800, 0.1);
const rows: string[] = [];
for (let pA = 800; pA >= 600; pA -= 10) {
	let n = 0;
	for (let t = 0; t < 2000; t++) n += cell.step(pA, 0.1);
	if (pA <= 680) rows.push(`${pA}: ${n * 5} Hz`);
}
console.log('lowered:', rows.join(', '));
// Dropped straight from 800 to each level while firing.
for (const pA of [630, 625, 620]) {
	const c = new HodgkinHuxley();
	for (let t = 0; t < 5000; t++) c.step(800, 0.1);
	let n = 0;
	for (let t = 0; t < 10000; t++) n += t >= 5000 ? c.step(pA, 0.1) : (c.step(pA, 0.1), 0);
	console.log(`dropped from 800 to ${pA}: ${n * 2} Hz over the second 500 ms`);
}
