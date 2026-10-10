import { STARTS, arrival, fly } from '../../src/widgets/pilot/recovery';
const model = await Bun.file(new URL('../../public/pilot/pilot.json', import.meta.url)).json();
for (const [name, start] of Object.entries(STARTS)) {
	const f = fly(model, start);
	const per = Array.from({ length: 6 }, (_, k) => f.spikes.slice(50 * k, 50 * k + 50).reduce((n, r) => n + r.reduce((a, b) => a + b, 0), 0));
	const total = per.reduce((a, b) => a + b, 0);
	const upright = f.states.findIndex((s) => Math.cos(s[4]) > 0.9);
	const minT = Math.min(...f.thrust.map((t) => Math.min(...t))), maxT = Math.max(...f.thrust.map((t) => Math.max(...t)));
	const end = f.states[300];
	console.log(`${name}: arrival ${arrival(f)}, upright (cos>0.9) from step ${upright}, spikes per 0.5 s ${per.join(' ')}, total ${total}; thrust ${minT.toFixed(2)}..${maxT.toFixed(2)} N; end at (${end[0].toFixed(2)}, ${end[1].toFixed(2)}) θ ${end[4].toFixed(2)}`);
}
