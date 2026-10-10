import { brunel } from '../../src/engines/brunel';
for (const [name, g, eta] of [['AI', 5, 2], ['SR', 3, 2], ['SIfast', 6, 4], ['SIslow', 4.5, 0.9]] as const) {
	for (const order of [250, 500]) {
		const a = brunel({ order, g, eta }, 7);
		const b = brunel({ order, g, eta }, 7);
		const n = a.network.size;
		const kickAt = 1000;
		const bins: string[] = [];
		let diff = 0;
		let total = 0;
		let first = -1;
		for (let t = 0; t < 5000; t++) {
			const ea = a.draw();
			const eb = b.draw();
			if (t === kickAt) {
				let i = 0;
				while (b.network.refractory[i] > 0) i++;
				eb[i] += 20;
			}
			a.network.step(ea);
			b.network.step(eb);
			for (let i = 0; i < n; i++) {
				if (a.network.fired[i] !== b.network.fired[i]) {
					diff++;
					if (first < 0) first = t;
				}
				total += a.network.fired[i];
			}
			if (t % 200 === 199) {
				bins.push(`${diff}/${total}`);
				diff = 0;
				total = 0;
			}
		}
		let vd = 0;
		for (let i = 0; i < n; i++) vd = Math.max(vd, Math.abs(a.network.v[i] - b.network.v[i]));
		console.log(name, order, 'first diff step', first, 'max v diff at end', vd.toFixed(3), 'per 20 ms (differing/total):', bins.join(' '));
	}
}
