import { Twins } from '../../src/engines/twins';
for (const [g, eta] of [[5, 2], [3, 2], [6, 4], [4.5, 0.9]] as const) {
	const twins = new Twins({ order: 250, g, eta, j: 1 });
	let before = 0;
	for (let t = 0; t < 1000; t++) before += twins.step();
	twins.nudge();
	let first = -1, differ = 0, total = 0;
	const bins: string[] = [];
	let bd = 0;
	for (let t = 0; t < 3000; t++) {
		const d = twins.step();
		if (d && first < 0) first = t;
		bd += d;
		if (t >= 500 && t < 1000) {
			differ += d;
			for (let i = 0; i < twins.size; i++) total += twins.a.network.fired[i] + twins.b.network.fired[i];
		}
		if (t % 100 === 99) { bins.push(String(bd)); bd = 0; }
	}
	console.log(`g ${g} eta ${eta}: before ${before}, first ${first}, 50-100 ms ${differ}/${total} = ${(differ / total).toFixed(3)}; per 10 ms: ${bins.join(' ')}`);
}
