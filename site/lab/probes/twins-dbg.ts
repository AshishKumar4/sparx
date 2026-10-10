import { Twins } from '../../src/engines/twins';
for (const j of [1, 0.1]) {
	const twins = new Twins({ order: 250, g: 5, eta: 2, j });
	for (let t = 0; t < 1000; t++) twins.step();
	const i = twins.nudge();
	const va = twins.a.network.v[i], vb = twins.b.network.v[i];
	const d0 = twins.step();
	console.log(`j ${j}: nudged ${i}, v before ${va} ${vb}, d at nudge ${d0}, fired a ${twins.a.network.fired[i]} b ${twins.b.network.fired[i]}, v after ${twins.a.network.v[i]} ${twins.b.network.v[i]}, ref ${twins.a.network.refractory[i]} ${twins.b.network.refractory[i]}`);
	let maxv = 0;
	for (let t = 1; t < 450; t++) {
		const d = twins.step();
		let m = 0;
		for (let k = 0; k < twins.size; k++) m = Math.max(m, Math.abs(twins.a.network.v[k] - twins.b.network.v[k]));
		if (t < 4 || t % 100 === 0) console.log(`  t ${t} d ${d} max|dv| ${m.toExponential(2)}`);
		if (d && t > 420) break;
	}
}
