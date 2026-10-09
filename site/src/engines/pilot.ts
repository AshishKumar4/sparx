// The front page's pilot: the network and the drone of site/lab/pilot.py, stepped in double precision.
import { type Layer, Stack } from './stack';
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

export interface PilotModel {
	drone: Drone;
	layers: Layer[];
	evaluation?: Record<string, number | null>;
	meta?: Record<string, unknown>;
}

export const INPUTS = ['to target x', 'to target y', 'velocity x', 'velocity y', 'sin θ', 'cos θ', 'spin'];

const clip = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

export class Pilot {
	readonly drone: Drone;
	readonly network: Stack;
	/** x, y, vx, vy, θ, ω: metres, metres per second, radians, radians per second. */
	readonly state = new Float64Array(6);
	readonly obs = new Float64Array(7);
	readonly thrust = new Float64Array(2);

	constructor(model: PilotModel) {
		this.drone = model.drone;
		this.network = new Stack(model.layers);
		if (this.network.inputs !== 7 || this.network.readout.length !== 2) throw new Error('the pilot reads 7 numbers and drives 2 rotors');
	}

	get spikes(): Uint8Array {
		return this.network.spikes;
	}

	get silenced(): Uint8Array {
		return this.network.silenced;
	}

	get readout(): Float64Array {
		return this.network.readout;
	}

	get neurons(): number {
		return this.network.neurons;
	}

	/** Every membrane at rest. */
	rest(): void {
		this.network.rest();
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
		return this.network.step(input);
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
