// The spiking stack the site's trained networks share (site/lab/stack.py): Dense and LIF layers, then a
// Dense layer into leaky integrators, stepped one step at a time in double precision.

export type Layer =
	| { kind: 'dense'; kernel: string; bias: string; inputs: number; outputs: number }
	| { kind: 'lif'; decay: number; threshold: number; reset: 'zero' | 'subtract' }
	| { kind: 'li'; decay: number };

type Step =
	| { kind: 'dense'; kernel: Float64Array; bias: Float64Array; inputs: number; outputs: number; out: Float64Array }
	| { kind: 'lif'; decay: number; threshold: number; zero: boolean; v: Float64Array; out: Float64Array; offset: number }
	| { kind: 'li'; decay: number; v: Float64Array };

function floats(base64: string): Float64Array {
	const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
	return Float64Array.from(new Float32Array(bytes.buffer));
}

export class Stack {
	readonly steps: Step[] = [];
	readonly inputs: number;
	/** Every spiking neuron's output this step, layer after layer. */
	readonly spikes: Uint8Array;
	/** Neurons silenced by the reader: they neither fire nor hold a charge. */
	readonly silenced: Uint8Array;
	readonly readout: Float64Array;
	readonly neurons: number;

	constructor(layers: Layer[]) {
		const first = layers[0];
		if (first?.kind !== 'dense') throw new Error('a stack starts with a dense layer');
		this.inputs = first.inputs;
		let width = first.inputs;
		let offset = 0;
		for (const layer of layers) {
			if (layer.kind === 'dense') {
				if (layer.inputs !== width) throw new Error(`a dense layer takes ${layer.inputs} inputs, not ${width}`);
				width = layer.outputs;
				this.steps.push({ ...layer, kernel: floats(layer.kernel), bias: floats(layer.bias), out: new Float64Array(width) });
			} else if (layer.kind === 'lif') {
				this.steps.push({ kind: 'lif', decay: layer.decay, threshold: layer.threshold, zero: layer.reset === 'zero',
					v: new Float64Array(width), out: new Float64Array(width), offset });
				offset += width;
			} else {
				this.steps.push({ kind: 'li', decay: layer.decay, v: new Float64Array(width) });
			}
		}
		const last = this.steps.at(-1);
		if (last?.kind !== 'li') throw new Error('a stack ends in a leaky-integrator readout');
		this.neurons = offset;
		this.spikes = new Uint8Array(offset);
		this.silenced = new Uint8Array(offset);
		this.readout = last.v;
	}

	/** The dense layers' kernels, `[inputs * outputs]` each, row-major by input. */
	kernels(): { kernel: Float64Array; inputs: number; outputs: number }[] {
		return this.steps.flatMap((s) => (s.kind === 'dense' ? [s] : []));
	}

	/** Every membrane at rest. */
	rest(): void {
		for (const step of this.steps) if (step.kind !== 'dense') step.v.fill(0);
		this.spikes.fill(0);
	}

	/** One step on `input`; returns the readout membranes. */
	step(input: ArrayLike<number>): Float64Array {
		let x: ArrayLike<number> = input;
		for (const step of this.steps) {
			if (step.kind === 'dense') {
				const { kernel, bias, inputs, outputs, out } = step;
				for (let j = 0; j < outputs; j++) {
					let sum = 0;
					for (let i = 0; i < inputs; i++) sum += x[i] * kernel[i * outputs + j];
					out[j] = sum + bias[j];
				}
				x = out;
			} else if (step.kind === 'lif') {
				const { v, out, decay, threshold, zero, offset } = step;
				for (let j = 0; j < v.length; j++) {
					let u = decay * v[j] + x[j];
					let s = u >= threshold ? 1 : 0;
					if (this.silenced[offset + j]) {
						u = 0;
						s = 0;
					} else if (s) u = zero ? 0 : u - threshold;
					v[j] = u;
					out[j] = s;
					this.spikes[offset + j] = s;
				}
				x = out;
			} else {
				const { v, decay } = step;
				for (let j = 0; j < v.length; j++) v[j] = decay * v[j] + x[j];
				x = v;
			}
		}
		return this.readout;
	}
}
