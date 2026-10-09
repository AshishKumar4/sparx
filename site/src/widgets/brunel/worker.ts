// Steps a Brunel network off the main thread and posts what the raster shows: the fired neurons among
// those shown and each population's spike count, per step, and the excitatory population's irregularity
// since the first 100 ms: the median CV of its neurons' intervals and the Fano factor of its count in 1 ms.
import { type BrunelRun, brunel } from '../../engines/brunel';

export interface Setup {
	order: number;
	g: number;
	eta: number;
	/** Each excitatory synapse's jump, mV. */
	j: number;
	seed: number;
	shown: number[];
	/** Simulated milliseconds per second of wall time. */
	speed: number;
	/** Simulated milliseconds sent in the first batch, so the figure starts full. */
	ahead: number;
}

export interface Batch {
	steps: number;
	/** For each step, how many shown neurons fired, then their positions in `shown`. */
	spikes: Int32Array;
	excitatory: Int32Array;
	inhibitory: Int32Array;
	time: number;
	cv: number;
	fano: number;
}

const SETTLE = 1000;
const WINDOW = 10;

let run: BrunelRun | null = null;
let setup: Setup | null = null;
let shownIndex = new Int32Array(0);
let timer = 0;
let last = new Float64Array(0);
let isiSum = new Float64Array(0);
let isiSquares = new Float64Array(0);
let isiCount = new Uint32Array(0);
let windowCount = 0;
let windows = 0;
let countSum = 0;
let countSquares = 0;

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
	run = brunel({ order: setup.order, g: setup.g, eta: setup.eta, j: setup.j }, setup.seed);
	shownIndex = new Int32Array(run.network.size).fill(-1);
	for (const [k, i] of setup.shown.entries()) shownIndex[i] = k;
	const ne = run.network.spec.excitatory;
	last = new Float64Array(ne).fill(-1);
	isiSum = new Float64Array(ne);
	isiSquares = new Float64Array(ne);
	isiCount = new Uint32Array(ne);
	windowCount = windows = countSum = countSquares = 0;
	clearTimeout(timer);
	tick(performance.now(), setup.ahead);
};

function statistics(): { cv: number; fano: number } {
	const cvs: number[] = [];
	for (let i = 0; i < isiCount.length; i++) {
		if (isiCount[i] < 3) continue;
		const mean = isiSum[i] / isiCount[i];
		cvs.push(Math.sqrt(Math.max(isiSquares[i] / isiCount[i] - mean * mean, 0)) / mean);
	}
	cvs.sort((a, b) => a - b);
	const mean = windows ? countSum / windows : 0;
	return {
		cv: cvs.length ? cvs[Math.floor(cvs.length / 2)] : Number.NaN,
		fano: mean ? (countSquares / windows - mean * mean) / mean : Number.NaN,
	};
}

function tick(previous: number, ahead = 0) {
	if (!run || !setup) return;
	const now = performance.now();
	const dt = run.network.spec.dt;
	const steps = ahead ? Math.round(ahead / dt) : Math.max(1, Math.min(400, Math.round((((now - previous) / 1000) * setup.speed) / dt)));
	const spikes: number[] = [];
	const excitatory = new Int32Array(steps);
	const inhibitory = new Int32Array(steps);
	const { network, draw } = run;
	const ne = network.spec.excitatory;
	for (let s = 0; s < steps; s++) {
		network.step(draw());
		const mark = spikes.length;
		spikes.push(0);
		let e = 0;
		let i = 0;
		const settled = network.t > SETTLE;
		for (let n = 0; n < network.size; n++) {
			if (!network.fired[n]) continue;
			if (n < ne) {
				e++;
				if (settled && last[n] >= 0) {
					const isi = network.t - last[n];
					isiSum[n] += isi;
					isiSquares[n] += isi * isi;
					isiCount[n]++;
				}
				last[n] = network.t;
			} else i++;
			if (shownIndex[n] >= 0) {
				spikes.push(shownIndex[n]);
				spikes[mark]++;
			}
		}
		excitatory[s] = e;
		inhibitory[s] = i;
		if (settled) {
			windowCount += e;
			if (network.t % WINDOW === 0) {
				windows++;
				countSum += windowCount;
				countSquares += windowCount * windowCount;
				windowCount = 0;
			}
		}
	}
	const batch: Batch = { steps, spikes: Int32Array.from(spikes), excitatory, inhibitory, time: network.t * dt, ...statistics() };
	(self as unknown as Worker).postMessage(batch, [batch.spikes.buffer, excitatory.buffer, inhibitory.buffer]);
	timer = setTimeout(() => tick(now), 16) as unknown as number;
}
