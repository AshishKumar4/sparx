// Steps a Brunel network off the main thread and posts what the raster shows: the fired neurons among
// those shown, and each population's spike count, per step.
import { brunel } from '../../engines/brunel';

export interface Setup {
	order: number;
	g: number;
	eta: number;
	seed: number;
	shown: number[];
	/** Simulated milliseconds per second of wall time. */
	speed: number;
}

export interface Batch {
	steps: number;
	/** For each step, how many shown neurons fired, then their positions in `shown`. */
	spikes: Int32Array;
	excitatory: Int32Array;
	inhibitory: Int32Array;
	time: number;
}

let run: ReturnType<typeof brunel> | null = null;
let setup: Setup | null = null;
let shownIndex = new Int32Array(0);
let timer = 0;

self.onmessage = (event: MessageEvent<Setup | 'pause' | 'resume'>) => {
	if (event.data === 'pause') {
		clearTimeout(timer);
		timer = 0;
		return;
	}
	if (event.data === 'resume') {
		if (!timer && run) tick(performance.now());
		return;
	}
	setup = event.data;
	run = brunel({ order: setup.order, g: setup.g, eta: setup.eta }, setup.seed);
	shownIndex = new Int32Array(run.network.size).fill(-1);
	for (const [k, i] of setup.shown.entries()) shownIndex[i] = k;
	clearTimeout(timer);
	tick(performance.now());
};

function tick(last: number) {
	if (!run || !setup) return;
	const now = performance.now();
	const dt = run.network.spec.dt;
	const steps = Math.max(1, Math.min(400, Math.round(((now - last) / 1000) * setup.speed / dt)));
	const spikes: number[] = [];
	const excitatory = new Int32Array(steps);
	const inhibitory = new Int32Array(steps);
	const { network, draw } = run;
	for (let s = 0; s < steps; s++) {
		network.step(draw());
		const mark = spikes.length;
		spikes.push(0);
		let e = 0;
		let i = 0;
		for (let n = 0; n < network.size; n++) {
			if (!network.fired[n]) continue;
			if (n < network.spec.excitatory) e++;
			else i++;
			if (shownIndex[n] >= 0) {
				spikes.push(shownIndex[n]);
				spikes[mark]++;
			}
		}
		excitatory[s] = e;
		inhibitory[s] = i;
	}
	const batch: Batch = { steps, spikes: Int32Array.from(spikes), excitatory, inhibitory, time: network.t * dt };
	(self as unknown as Worker).postMessage(batch, [batch.spikes.buffer, excitatory.buffer, inhibitory.buffer]);
	timer = setTimeout(() => tick(now), 16) as unknown as number;
}
