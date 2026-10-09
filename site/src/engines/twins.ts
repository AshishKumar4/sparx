// Two copies of one Brunel network (src/engines/brunel.ts): the same connections and the same external
// input, step for step, until one neuron of the second copy is nudged over threshold once.
import { type Brunel, type BrunelRun, brunel } from './brunel';

export class Twins {
	readonly a: BrunelRun;
	readonly b: BrunelRun;
	private kick = -1;

	constructor(spec: Brunel, seed = 7) {
		this.a = brunel(spec, seed);
		this.b = brunel(spec, seed);
	}

	get size(): number {
		return this.a.network.size;
	}

	/** Pushes the first neuron of the second copy that is not refractory over threshold at the next step. */
	nudge(): number {
		let i = 0;
		while (this.b.network.refractory[i] > 0) i++;
		this.kick = i;
		return i;
	}

	/** One step of both copies; returns how many neurons fired in one copy and not the other. */
	step(): number {
		const ea = this.a.draw();
		const eb = this.b.draw();
		if (this.kick >= 0) {
			// 1 V crosses threshold from any membrane potential; the reset discards the excess.
			eb[this.kick] += 1000;
			this.kick = -1;
		}
		this.a.network.step(ea);
		this.b.network.step(eb);
		let differ = 0;
		for (let i = 0; i < this.size; i++) differ += this.a.network.fired[i] !== this.b.network.fired[i] ? 1 : 0;
		return differ;
	}
}
