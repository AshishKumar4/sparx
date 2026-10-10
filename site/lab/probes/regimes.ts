import { brunel } from '../../src/engines/brunel';
const regimes = [['SR', 3, 2], ['AI', 5, 2], ['SIfast', 6, 4], ['SIslow', 4.5, 0.9]] as const;
const configs: [number, number][] = [];
for (const [order, j] of configs) {
	const rows: string[] = [];
	for (const [name, g, eta] of regimes) {
		const { network, draw } = brunel({ order, g, eta, j }, 7);
		const ne = network.spec.excitatory;
		const last = new Float64Array(ne).fill(-1);
		const isis: number[][] = Array.from({ length: ne }, () => []);
		const counts: number[] = [];
		let spikes = 0;
		let windowCount = 0;
		const steps = 6000;
		const warm = 2000;
		const t0 = performance.now();
		for (let t = 0; t < steps; t++) {
			network.step(draw());
			if (t < warm) continue;
			for (let i = 0; i < ne; i++) {
				if (!network.fired[i]) continue;
				spikes++;
				windowCount++;
				if (last[i] >= 0) isis[i].push(t - last[i]);
				last[i] = t;
			}
			if ((t - warm) % 10 === 9) {
				counts.push(windowCount);
				windowCount = 0;
			}
		}
		const rate = spikes / ne / ((steps - warm) * 0.1e-3);
		const cvs = isis.filter((x) => x.length >= 3).map((x) => {
			const m = x.reduce((a, b) => a + b, 0) / x.length;
			const v = x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length;
			return Math.sqrt(v) / m;
		}).sort((a, b) => a - b);
		const cv = cvs.length ? cvs[Math.floor(cvs.length / 2)] : NaN;
		const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
		const fano = mean ? counts.reduce((a, b) => a + (b - mean) ** 2, 0) / counts.length / mean : 0;
		rows.push(`${name} ${rate.toFixed(1)}Hz CV ${cv.toFixed(2)} F ${fano.toFixed(1)} (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
	}
	console.log(`order ${order} (${5 * order} neurons, C_E ${0.4 * order}) J ${j}: ${rows.join(' | ')}`);
}
