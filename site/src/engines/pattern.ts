// Masquelier, Guyonneau and Thorpe's (2008) experiment, small: afferents fire at random, and half of them
// now and then replay one frozen 50 ms pattern at the same rate, so no afferent's rate gives it away. One
// LeakyIntegrateAndFire neuron learns by sparx's PairSTDP and comes to fire only within the pattern.
import { generator } from './brunel';
import { LeakyIntegrateAndFire } from './neurons';
import { PairSTDP, type Rule } from './stdp';

export interface Setup {
	afferents: number;
	rate: number;
	noise: number;
	length: number;
	share: number;
	jump: number;
	threshold: number;
	rule: Rule;
	seed: number;
}

export const SETUP: Setup = {
	afferents: 400,
	rate: 30,
	noise: 10,
	length: 50,
	share: 0.25,
	jump: 1,
	threshold: 90,
	// Masquelier et al.'s time constants with weaker depression: at their alpha of 0.85 this smaller network
	// falls silent before it finds the pattern.
	rule: { tau_plus: 16.8, tau_minus: 33.7, lambda: 0.005, alpha: 0.55, mu_plus: 0, mu_minus: 0, w_max: 1 },
	seed: 3,
};

export class Pattern {
	readonly random: () => number;
	readonly frozen: Uint8Array[];
	readonly spikes: Uint8Array;
	readonly weights: Float64Array;
	readonly neuron: LeakyIntegrateAndFire;
	readonly stdp: PairSTDP;
	private readonly pre: Int32Array;
	private readonly post: Int32Array;
	private readonly fired = new Uint8Array(1);
	private arriving = 0;
	t = 0;
	/** The step within the current pattern, or -1 outside one. */
	phase = -1;
	learning = true;

	constructor(readonly setup: Setup = SETUP) {
		const { afferents, rate, length, seed } = setup;
		this.random = generator(seed);
		this.frozen = Array.from({ length }, () => Uint8Array.from({ length: afferents }, (_, i) => (i < afferents / 2 && this.random() < rate / 1000 ? 1 : 0)));
		this.spikes = new Uint8Array(afferents);
		this.weights = new Float64Array(afferents).fill(0.475);
		this.neuron = new LeakyIntegrateAndFire(10, 250, 0, setup.threshold, 0, 1);
		this.stdp = new PairSTDP(setup.rule, afferents, 1);
		this.pre = Int32Array.from({ length: afferents }, (_, i) => i);
		this.post = new Int32Array(afferents);
	}

	/** One step of 1 ms; returns 1 if the neuron fired. */
	step(): number {
		const { afferents, rate, noise, length, share, jump } = this.setup;
		if (this.t % length === 0) this.phase = this.random() < share ? 0 : -1;
		else if (this.phase >= 0) this.phase++;
		let input = 0;
		for (let i = 0; i < afferents; i++) {
			const replay = this.phase >= 0 && i < afferents / 2;
			const s = (replay ? this.frozen[this.phase][i] === 1 : this.random() < rate / 1000) || this.random() < noise / 1000 ? 1 : 0;
			this.spikes[i] = s;
			if (s) input += this.weights[i];
		}
		// Spikes reach the neuron 1 ms after they are sent, so each one that helps it fire comes before its spike.
		this.fired[0] = this.neuron.step(0, 1, jump * this.arriving);
		this.arriving = input;
		if (this.learning) this.stdp.step(this.weights, this.spikes, this.fired, this.pre, this.post, 1);
		this.t++;
		return this.fired[0];
	}
}
