// The front page's pilot: the network and the drone of site/lab/pilot.py, stepped in double precision.
// test/pilot.test.ts holds it to sparx's own run of the same flights.

export interface Drone {
	dt: number;
	mass: number;
	arm: number;
	inertia: number;
	gravity: number;
	max_thrust: number;
	drag: number;
	spin_drag: number;
	reach: number;
	bounce: number;
	hover_logit: number;
}

type Layer =
	| { kind: 'dense'; kernel: string; bias: string; inputs: number; outputs: number }
	| { kind: 'lif'; decay: number; threshold: number; reset: 'zero' | 'subtract' }
	| { kind: 'li'; decay: number };

export interface PilotModel {
	drone: Drone;
	layers: Layer[];
	evaluation?: Record<string, number | null>;
	meta?: Record<string, unknown>;
}

type Step =
	| { kind: 'dense'; kernel: Float64Array; bias: Float64Array; inputs: number; outputs: number; out: Float64Array }
	| { kind: 'lif'; decay: number; threshold: number; zero: boolean; v: Float64Array; out: Float64Array; offset: number }
	| { kind: 'li'; decay: number; v: Float64Array };

export const INPUTS = ['to target x', 'to target y', 'velocity x', 'velocity y', 'sin θ', 'cos θ', 'spin'];

function floats(base64: string): Float64Array {
	const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
	return Float64Array.from(new Float32Array(bytes.buffer));
}

const clip = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

export class Pilot {
	readonly drone: Drone;
	readonly steps: Step[] = [];
	/** x, y, vx, vy, θ, ω: metres, metres per second, radians, radians per second. */
	readonly state = new Float64Array(6);
	readonly obs = new Float64Array(7);
	/** Every spiking neuron's output this step, layer after layer. */
	readonly spikes: Uint8Array;
	/** Neurons silenced by the visitor: they neither fire nor hold a charge. */
	readonly silenced: Uint8Array;
	readonly readout: Float64Array;
	readonly thrust = new Float64Array(2);
	readonly neurons: number;

	constructor(model: PilotModel) {
		this.drone = model.drone;
		let width = 7;
		let offset = 0;
		for (const layer of model.layers) {
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
		this.neurons = offset;
		this.spikes = new Uint8Array(offset);
		this.silenced = new Uint8Array(offset);
		const last = this.steps.at(-1);
		if (last?.kind !== 'li' || last.v.length !== 2) throw new Error('the pilot ends in a readout of two membranes');
		this.readout = last.v;
	}

	/** Every membrane at rest. */
	rest(): void {
		for (const step of this.steps) if (step.kind !== 'dense') step.v.fill(0);
		this.spikes.fill(0);
	}

	observe(tx: number, ty: number): Float64Array {
		const [x, y, vx, vy, theta, omega] = this.state;
		const r = this.drone.reach;
		const o = this.obs;
		o[0] = clip(tx - x, -r, r) / r;
		o[1] = clip(ty - y, -r, r) / r;
		o[2] = vx / 4;
		o[3] = vy / 4;
		o[4] = Math.sin(theta);
		o[5] = Math.cos(theta);
		o[6] = omega / 8;
		return o;
	}

	/** One step of the network on `input`; returns the readout membranes. */
	think(input: Float64Array): Float64Array {
		let x = input;
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
		return x;
	}

	/** Thrust from the readout, then one semi-implicit Euler step of the drone in the box `[w, h]`. */
	fly(w: number, h: number): void {
		const d = this.drone;
		const s = this.state;
		for (let k = 0; k < 2; k++) this.thrust[k] = d.max_thrust / (1 + Math.exp(-(this.readout[k] + d.hover_logit)));
		const [left, right] = this.thrust;
		const total = left + right;
		let [x, y, vx, vy, theta, omega] = s;
		vx = vx + (d.dt * (-total * Math.sin(theta) - d.drag * vx)) / d.mass;
		vy = vy + d.dt * ((total * Math.cos(theta) - d.drag * vy) / d.mass - d.gravity);
		omega = omega + (d.dt * ((right - left) * d.arm - d.spin_drag * omega)) / d.inertia;
		x = x + d.dt * vx;
		y = y + d.dt * vy;
		theta = theta + d.dt * omega;
		if (Math.abs(x) > w) vx = -d.bounce * vx;
		if (Math.abs(y) > h) vy = -d.bounce * vy;
		s[0] = clip(x, -w, w);
		s[1] = clip(y, -h, h);
		s[2] = vx;
		s[3] = vy;
		s[4] = theta;
		s[5] = omega;
	}

	/** Observe, think and fly: one step of 10 ms, as `fly` in site/lab/pilot.py takes it. */
	step(tx: number, ty: number, w: number, h: number): void {
		this.think(this.observe(tx, ty));
		this.fly(w, h);
	}
}
