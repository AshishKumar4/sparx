// Two copies of one Brunel network (src/engines/brunel.ts): the same connections and the same external
// input, step for step, until one neuron of the second copy is nudged over threshold once.
import { brunel, type Brunel } from './brunel';

export class Twins {
	readonly a: ReturnType<typeof brunel>;
	readonly b: ReturnType<typeof brunel>;
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
			eb[this.kick] += this.b.network.spec.v_th - this.b.network.spec.v_reset + 10;
			this.kick = -1;
		}
		this.a.network.step(ea);
		this.b.network.step(eb);
		let differ = 0;
		for (let i = 0; i < this.size; i++) differ += this.a.network.fired[i] !== this.b.network.fired[i] ? 1 : 0;
		return differ;
	}
}
