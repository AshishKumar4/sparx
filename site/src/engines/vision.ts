// The convolutional network of site/lab/stack.py's Vision: a camera's image through two strided
// convolutions, each into a population of units, a dense layer with the inputs past the image, and
// leaky-integrator readouts, stepped in double precision. Its units are LIF neurons that reset to zero,
// or graded units, a leaky membrane read through a ReLU.

export interface VisionModel {
	config: {
		/** Rows, columns and channels of the image at the front of each input. */
		shape: [number, number, number];
		features: number[];
		kernels: number[];
		hidden: number;
		outputs: number;
		neuron: string;
		bits?: number;
		decay: number;
		readout_decay: number;
	};
	/** Every parameter by path, such as `Conv_0/kernel` `[kernel, kernel, in, out]`, as base64 float32. */
	params: Record<string, { shape: number[]; data: string }>;
}

interface Conv {
	kernel: Float64Array;
	bias: Float64Array;
	size: number;
	/** The input's rows, columns and channels, the output's, and the padding before the first row and column. */
	rows: number;
	columns: number;
	channels: number;
	outRows: number;
	outColumns: number;
	features: number;
	top: number;
	left: number;
}

interface Dense {
	kernel: Float64Array;
	bias: Float64Array;
	inputs: number;
	outputs: number;
}

interface Units {
	v: Float64Array;
	out: Float64Array;
	offset: number;
}

function floats(base64: string): Float64Array {
	const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
	return Float64Array.from(new Float32Array(bytes.buffer));
}

/** XLA's `SAME` padding: the output is the input's size over the stride, rounded up, and the padding that
 * takes splits with the smaller half before. */
function before(size: number, kernel: number, stride: number): number {
	const out = Math.ceil(size / stride);
	return Math.floor(Math.max((out - 1) * stride + kernel - size, 0) / 2);
}

export class Vision {
	readonly inputs: number;
	readonly neurons: number;
	/** Every unit's output this step, 1 where it fired or is active, population after population. */
	readonly spikes: Uint8Array;
	/** Units silenced by the reader: they neither fire nor hold a charge. */
	readonly silenced: Uint8Array;
	readonly readout: Float64Array;
	private readonly convs: Conv[] = [];
	private readonly hidden: Dense;
	private readonly output: Dense;
	private readonly units: Units[] = [];
	private readonly image: Float64Array;
	private readonly joined: Float64Array;
	private readonly drive: Float64Array;
	private readonly pixels: number;
	private readonly decay: number;
	private readonly readoutDecay: number;
	private readonly graded: boolean;

	constructor(model: VisionModel) {
		const { config, params } = model;
		if (config.neuron !== 'lif' && config.neuron !== 'relu') throw new Error(`no ${config.neuron} units in the browser`);
		if ((config.bits ?? 1) !== 1) throw new Error('the browser sends one-bit spikes');
		const param = (path: string) => {
			const found = params[path];
			if (!found) throw new Error(`the model has no ${path}`);
			return floats(found.data);
		};
		this.graded = config.neuron === 'relu';
		this.decay = config.decay;
		this.readoutDecay = config.readout_decay;
		let [rows, columns, channels] = config.shape;
		this.pixels = rows * columns * channels;
		this.image = new Float64Array(this.pixels);
		let offset = 0;
		for (const [k, features] of config.features.entries()) {
			const size = config.kernels[k];
			const outRows = Math.ceil(rows / 2);
			const outColumns = Math.ceil(columns / 2);
			const conv = { kernel: param(`Conv_${k}/kernel`), bias: param(`Conv_${k}/bias`), size, rows, columns, channels,
				outRows, outColumns, features, top: before(rows, size, 2), left: before(columns, size, 2) };
			if (conv.kernel.length !== size * size * channels * features) throw new Error(`Conv_${k}'s kernel has the wrong size`);
			this.convs.push(conv);
			const width = outRows * outColumns * features;
			this.units.push({ v: new Float64Array(width), out: new Float64Array(width), offset });
			offset += width;
			[rows, columns, channels] = [outRows, outColumns, features];
		}
		const flat = rows * columns * channels;
		const hiddenKernel = param('Dense_0/kernel');
		const extra = hiddenKernel.length / config.hidden - flat;
		if (!Number.isInteger(extra) || extra < 0) throw new Error("Dense_0's kernel does not fit the image");
		this.inputs = this.pixels + extra;
		this.joined = new Float64Array(flat + extra);
		this.hidden = { kernel: hiddenKernel, bias: param('Dense_0/bias'), inputs: flat + extra, outputs: config.hidden };
		this.units.push({ v: new Float64Array(config.hidden), out: new Float64Array(config.hidden), offset });
		offset += config.hidden;
		this.output = { kernel: param('Dense_1/kernel'), bias: param('Dense_1/bias'), inputs: config.hidden, outputs: config.outputs };
		this.neurons = offset;
		this.spikes = new Uint8Array(offset);
		this.silenced = new Uint8Array(offset);
		this.readout = new Float64Array(config.outputs);
		this.drive = new Float64Array(config.outputs);
	}

	/** Every membrane at rest. */
	rest(): void {
		for (const units of this.units) units.v.fill(0);
		this.readout.fill(0);
		this.spikes.fill(0);
	}

	/** One step on `input`, the image channel by channel, then the rest; returns the readout membranes. */
	step(input: ArrayLike<number>): Float64Array {
		const [rows, columns, channels] = [this.convs[0].rows, this.convs[0].columns, this.convs[0].channels];
		// The input holds each channel's image in turn; the convolutions read rows, columns, then channels.
		for (let c = 0; c < channels; c++)
			for (let p = 0; p < rows * columns; p++) this.image[p * channels + c] = input[c * rows * columns + p];
		let x: Float64Array = this.image;
		for (const [k, conv] of this.convs.entries()) {
			const units = this.units[k];
			this.convolve(conv, x, units.out);
			this.fire(units, units.out);
			x = units.out;
		}
		this.joined.set(x);
		for (let i = this.pixels; i < this.inputs; i++) this.joined[x.length + i - this.pixels] = input[i];
		const hidden = this.units[this.convs.length];
		dense(this.hidden, this.joined, hidden.out);
		this.fire(hidden, hidden.out);
		dense(this.output, hidden.out, this.drive);
		for (let j = 0; j < this.readout.length; j++) this.readout[j] = this.readoutDecay * this.readout[j] + this.drive[j];
		return this.readout;
	}

	private convolve(conv: Conv, x: Float64Array, into: Float64Array): void {
		const { kernel, bias, size, rows, columns, channels, outRows, outColumns, features, top, left } = conv;
		for (let i = 0; i < outRows; i++)
			for (let j = 0; j < outColumns; j++)
				for (let f = 0; f < features; f++) {
					let sum = 0;
					for (let di = 0; di < size; di++) {
						const r = 2 * i + di - top;
						if (r < 0 || r >= rows) continue;
						for (let dj = 0; dj < size; dj++) {
							const c = 2 * j + dj - left;
							if (c < 0 || c >= columns) continue;
							for (let ch = 0; ch < channels; ch++)
								sum += x[(r * columns + c) * channels + ch] * kernel[((di * size + dj) * channels + ch) * features + f];
						}
					}
					into[(i * outColumns + j) * features + f] = sum + bias[f];
				}
	}

	/** Each unit's membrane takes in `drive` (which it overwrites with the units' outputs): a LIF neuron fires
	 * at 1 and resets to zero, a graded unit sends its membrane through a ReLU. */
	private fire(units: Units, drive: Float64Array): void {
		const { v, offset } = units;
		for (let j = 0; j < v.length; j++) {
			let u = this.decay * v[j] + drive[j];
			let s: number;
			if (this.silenced[offset + j]) {
				u = 0;
				s = 0;
			} else if (this.graded) s = Math.max(u, 0);
			else {
				s = u >= 1 ? 1 : 0;
				if (s) u = 0;
			}
			v[j] = u;
			drive[j] = s;
			this.spikes[offset + j] = s > 0 ? 1 : 0;
		}
	}
}

function dense(layer: Dense, x: ArrayLike<number>, into: Float64Array): void {
	const { kernel, bias, inputs, outputs } = layer;
	for (let j = 0; j < outputs; j++) {
		let sum = 0;
		for (let i = 0; i < inputs; i++) sum += x[i] * kernel[i * outputs + j];
		into[j] = sum + bias[j];
	}
}
