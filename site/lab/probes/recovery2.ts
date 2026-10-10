import { STARTS, fly } from '../../src/widgets/pilot/recovery';
const model = await Bun.file(new URL('../../public/pilot/pilot.json', import.meta.url)).json();
for (const [name, start] of Object.entries(STARTS)) {
	const f = fly(model, start);
	const share = (from: number, to: number) => {
		let n = 0;
		for (let t = from; t < to; t++) for (const T of f.thrust[t]) if (T < 0.5 || T > 11.5) n++;
		return (100 * n / (2 * (to - from))).toFixed(0);
	};
	console.log(`${name}: rotor-steps within 0.5 N of a limit: first 0.8 s ${share(0, 80)}%, first 1.5 s ${share(0, 150)}%, last 1.5 s ${share(150, 300)}%`);
}
