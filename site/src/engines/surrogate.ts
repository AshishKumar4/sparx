// The derivatives of sparx.surrogate, at `x = v - threshold`. The forward pass is the step `x >= 0`.

export interface Surrogate {
	name: string;
	call: string;
	derivative: (x: number) => number;
}

export function surrogates(sharpness = 1): Surrogate[] {
	const alpha = 2 * sharpness;
	const sigmoidAlpha = 4 * sharpness;
	const slope = 25 * sharpness;
	const width = 1 / sharpness;
	const sigma = 0.5 / sharpness;
	return [
		{
			name: 'ATan',
			call: `ATan(alpha=${alpha.toFixed(1)})`,
			derivative: (x) => alpha / 2 / (1 + ((Math.PI / 2) * alpha * x) ** 2),
		},
		{
			name: 'Sigmoid',
			call: `Sigmoid(alpha=${sigmoidAlpha.toFixed(1)})`,
			derivative: (x) => {
				const s = 1 / (1 + Math.exp(-sigmoidAlpha * x));
				return sigmoidAlpha * s * (1 - s);
			},
		},
		{
			name: 'FastSigmoid',
			call: `FastSigmoid(slope=${slope.toFixed(0)})`,
			derivative: (x) => 1 / (slope * Math.abs(x) + 1) ** 2,
		},
		{
			name: 'Triangle',
			call: `Triangle(width=${width.toFixed(2)}, scale=1.0)`,
			derivative: (x) => Math.max(0, 1 - Math.abs(x) / width),
		},
		{
			name: 'Rectangle',
			call: `Rectangle(width=${width.toFixed(2)})`,
			derivative: (x) => (Math.abs(x) < width / 2 ? 1 / width : 0),
		},
		{
			name: 'Gaussian',
			call: `Gaussian(sigma=${sigma.toFixed(2)})`,
			derivative: (x) => Math.exp(-0.5 * (x / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)),
		},
	];
}
