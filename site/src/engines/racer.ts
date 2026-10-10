// The racer of site/lab/racer.py: a track, a car, its event camera and the spiking network that drives it,
// stepped in double precision with the same operations in the same order. test/racer.test.ts holds it to
// sparx's own float64 laps.
import { type Layer, Stack } from './stack';

export interface World {
	dt: number;
	wheelbase: number;
	max_steer: number;
	max_speed: number;
	speed_tau: number;
	half_width: number;
	edge: number;
	dash_period: number;
	dash_width: number;
	threshold: number;
	columns: number;
	rows: number;
	/** Points around a track. */
	points: number;
	behind: number;
	ahead: number;
}

export interface RacerModel {
	world: World;
	ground: { forward: number[]; side: number[] };
	layers: Layer[];
	meta?: Record<string, unknown>;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** A closed track through `points`, with each segment's vector, length and distance along the track. */
export class Track {
	readonly n: number;
	readonly x: Float64Array;
	readonly y: Float64Array;
	readonly dx: Float64Array;
	readonly dy: Float64Array;
	readonly length2: Float64Array;
	readonly length: Float64Array;
	readonly along: Float64Array;
	readonly total: number;
	readonly dashPeriod: number;

	constructor(points: ArrayLike<ArrayLike<number>>, world: World) {
		const n = points.length;
		this.n = n;
		this.x = Float64Array.from({ length: n }, (_, k) => points[k][0]);
		this.y = Float64Array.from({ length: n }, (_, k) => points[k][1]);
		this.dx = new Float64Array(n);
		this.dy = new Float64Array(n);
		this.length2 = new Float64Array(n);
		this.length = new Float64Array(n);
		this.along = new Float64Array(n);
		let run = 0;
		for (let k = 0; k < n; k++) {
			this.dx[k] = this.x[(k + 1) % n] - this.x[k];
			this.dy[k] = this.y[(k + 1) % n] - this.y[k];
			this.length2[k] = this.dx[k] ** 2 + this.dy[k] ** 2;
			this.length[k] = Math.sqrt(this.length2[k]);
			run += this.length[k];
			this.along[k] = run - this.length[k];
		}
		this.total = run;
		this.dashPeriod = run / Math.max(Math.round(run / world.dash_period), 1);
	}

	/** The index of the track point nearest `(x, y)`. */
	nearest(x: number, y: number): number {
		let best = 0;
		let d = Infinity;
		for (let k = 0; k < this.n; k++) {
			const e = (this.x[k] - x) ** 2 + (this.y[k] - y) ** 2;
			if (e < d) {
				d = e;
				best = k;
			}
		}
		return best;
	}

	/** Distance from `(x, y)` to the middle, distance along the track and heading there, over the segments
	 * from `behind` before point `k0` to `ahead` after it. */
	locate(world: World, x: number, y: number, k0: number, out: Float64Array): Float64Array {
		let best = Infinity;
		for (let w = -world.behind; w < world.ahead; w++) {
			const k = (((k0 + w) % this.n) + this.n) % this.n;
			const rx = x - this.x[k];
			const ry = y - this.y[k];
			const t = Math.min(Math.max((rx * this.dx[k] + ry * this.dy[k]) / this.length2[k], 0), 1);
			const qx = rx - t * this.dx[k];
			const qy = ry - t * this.dy[k];
			const d2 = qx * qx + qy * qy;
			if (d2 < best) {
				best = d2;
				out[1] = this.along[k] + t * this.length[k];
				out[2] = Math.atan2(this.dy[k], this.dx[k]);
			}
		}
		out[0] = Math.sqrt(best + 1e-6);
		return out;
	}
}

export class Racer {
	readonly world: World;
	readonly network: Stack;
	readonly forward: Float64Array;
	readonly side: Float64Array;
	readonly pixels: number;
	/** x, y, heading, speed: metres, radians, metres per second. */
	readonly car = new Float64Array(4);
	readonly level: Float64Array;
	readonly seen: Float64Array;
	/** This step's events, ON for each pixel then OFF for each pixel, then the speed: the network's input. */
	readonly input: Float64Array;
	/** Where the car is: distance from the middle, distance along the track, the track's heading. */
	readonly place = new Float64Array(3);
	steer = 0;
	target = 0;
	private readonly scratch = new Float64Array(3);

	constructor(
		model: RacerModel,
		public track: Track,
	) {
		this.world = model.world;
		this.network = new Stack(model.layers);
		this.forward = Float64Array.from(model.ground.forward);
		this.side = Float64Array.from(model.ground.side);
		this.pixels = this.forward.length;
		this.level = new Float64Array(this.pixels);
		this.seen = new Float64Array(this.pixels);
		this.input = new Float64Array(2 * this.pixels + 1);
		if (this.network.inputs !== this.input.length) throw new Error(`the network reads ${this.network.inputs} inputs, not ${this.input.length}`);
	}

	/** Puts the car at `car` with every pixel's level at what it sees there and every neuron at rest. */
	reset(car: ArrayLike<number>): void {
		this.car.set(car);
		this.look(this.level);
		this.network.rest();
	}

	/** The log brightness each pixel sees from where the car is. */
	look(into: Float64Array): Float64Array {
		const w = this.world;
		const [x, y, psi] = this.car;
		const c = Math.cos(psi);
		const s = Math.sin(psi);
		const k0 = this.track.nearest(x, y);
		for (let p = 0; p < this.pixels; p++) {
			const px = x + this.forward[p] * c - this.side[p] * s;
			const py = y + this.forward[p] * s + this.side[p] * c;
			const [distance, along] = this.track.locate(w, px, py, k0, this.scratch);
			const verge = sigmoid((distance - w.half_width) / w.edge);
			const dash = sigmoid((w.dash_width - distance) / 0.03) * sigmoid(6 * Math.sin((2 * Math.PI * along) / this.track.dashPeriod));
			into[p] = Math.log(0.12 + 0.7 * verge + 0.6 * dash * (1 - verge));
		}
		return into;
	}

	/** One step: look, fire events, think, drive. Returns how many events fired. */
	step(): number {
		const w = this.world;
		this.look(this.seen);
		let events = 0;
		for (let p = 0; p < this.pixels; p++) {
			const change = this.seen[p] - this.level[p];
			const on = change - w.threshold >= 0 ? 1 : 0;
			const off = -change - w.threshold >= 0 ? 1 : 0;
			this.level[p] += w.threshold * (on - off);
			this.input[p] = on;
			this.input[this.pixels + p] = off;
			events += on + off;
		}
		this.input[2 * this.pixels] = this.car[3] / w.max_speed;
		const u = this.network.step(this.input);
		this.steer = w.max_steer * Math.tanh(u[0]);
		this.target = w.max_speed * sigmoid(u[1]);
		let [x, y, psi, v] = this.car;
		v = v + (w.dt * (this.target - v)) / w.speed_tau;
		psi = psi + (w.dt * v * Math.tan(this.steer)) / w.wheelbase;
		x = x + w.dt * v * Math.cos(psi);
		y = y + w.dt * v * Math.sin(psi);
		this.car.set([x, y, psi, v]);
		this.track.locate(w, x, y, this.track.nearest(x, y), this.place);
		return events;
	}
}
