// The engines read only a layer's nonzero inputs (dense in stack.ts, convolve in vision.ts). Each must give
// exactly, bit for bit, what a sum over every input gives, on the published racer's weights: every
// output's terms arrive in the same order, and a zero adds nothing.
import { expect, test } from 'bun:test';
import type { RacerModel } from '../src/engines/racer';
import { dense, floats } from '../src/engines/stack';
import { type Conv, convolve, Vision } from '../src/engines/vision';

const model: RacerModel = await Bun.file(new URL('../public/racer/racer.json', import.meta.url)).json();

/** Inputs like a layer's: `share` of them nonzero, spikes of 1 or graded values, the rest +0 or -0. */
function inputs(n: number, share: number, graded: boolean, seed: number): Float64Array {
	let state = seed;
	const random = () => {
		state = (state * 1664525 + 1013904223) % 4294967296;
		return state / 4294967296;
	};
	return Float64Array.from({ length: n }, () => (random() < share ? (graded ? random() * 3 : 1) : random() < 0.5 ? 0 : -0));
}

function gatherConv(conv: Conv, x: Float64Array): Float64Array {
	const { kernel, bias, size, rows, columns, channels, outRows, outColumns, features, top, left } = conv;
	const out = new Float64Array(outRows * outColumns * features);
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
						for (let ch = 0; ch < channels; ch++) sum += x[(r * columns + c) * channels + ch] * kernel[((di * size + dj) * channels + ch) * features + f];
					}
				}
				out[(i * outColumns + j) * features + f] = sum + bias[f];
			}
	return out;
}

const vision = new Vision(model.vision as NonNullable<RacerModel['vision']>);
for (const [k, conv] of vision.convs.entries())
	for (const [share, graded] of [[0.09, false], [0.53, true], [1, true]] as const)
		test(`convolution ${k}, ${share * 100}% of inputs nonzero${graded ? ', graded' : ''}: bit for bit`, () => {
			const x = inputs(conv.rows * conv.columns * conv.channels, share, graded, 7 + k);
			const out = new Float64Array(conv.outRows * conv.outColumns * conv.features);
			convolve(conv, x, out);
			expect(Array.from(out)).toEqual(Array.from(gatherConv(conv, x)));
		});

test('the dense layer, 9% and all of its inputs nonzero: bit for bit', () => {
	const params = (model.vision as NonNullable<RacerModel['vision']>).params;
	const kernel = floats(params['Dense_0/kernel'].data);
	const bias = floats(params['Dense_0/bias'].data);
	const outputs = bias.length;
	const layer = { kernel, bias, inputs: kernel.length / outputs, outputs };
	for (const [share, graded] of [[0.09, false], [1, true]] as const) {
		const x = inputs(layer.inputs, share, graded, 3);
		const out = new Float64Array(outputs);
		dense(layer, x, out);
		const reference = new Float64Array(outputs);
		for (let j = 0; j < outputs; j++) {
			let sum = 0;
			for (let i = 0; i < layer.inputs; i++) sum += x[i] * kernel[i * outputs + j];
			reference[j] = sum + bias[j];
		}
		expect(Array.from(out)).toEqual(Array.from(reference));
	}
});

test('a weight that is not finite is refused when the model loads', () => {
	for (const bad of [Number.POSITIVE_INFINITY, Number.NaN]) {
		const bytes = new Uint8Array(new Float32Array([0.5, bad, 1]).buffer);
		expect(() => floats(btoa(String.fromCharCode(...bytes)))).toThrow('not finite');
	}
});
