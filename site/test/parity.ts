// What the racer's parity tests measure against sparx's float64 runs, step by step: the readout membranes'
// largest and RMS difference relative to their own size, and the car's largest difference per component
// in units in the last place of its largest value, which lab/car_exact.py bounds.
/** How far the browser's readouts may be from sparx's, relative to their largest size: the parity tests
 * hold them to it, and lab/car_exact.py budgets the car's rounding with it. */
export const READOUT_LIMIT = 1e-15;

export class Distance {
	private worst = 0;
	private scale = 0;
	private squares = 0;
	private norm = 0;
	private readonly error = [0, 0, 0, 0];
	private readonly size = [0, 0, 0, 0];

	/** One step: the browser's readouts and car against the recorded ones. */
	add(readout: ArrayLike<number>, recordedReadout: number[], car: ArrayLike<number>, recordedCar: number[]): void {
		for (let i = 0; i < recordedReadout.length; i++) {
			const d = readout[i] - recordedReadout[i];
			this.worst = Math.max(this.worst, Math.abs(d));
			this.scale = Math.max(this.scale, Math.abs(recordedReadout[i]));
			this.squares += d * d;
			this.norm += recordedReadout[i] ** 2;
		}
		for (let i = 0; i < 4; i++) {
			this.error[i] = Math.max(this.error[i], Math.abs(car[i] - recordedCar[i]));
			this.size[i] = Math.max(this.size[i], Math.abs(recordedCar[i]));
		}
	}

	/** The readouts' largest difference relative to their largest size; 0 when both stay at 0. */
	get readoutMaxRelative(): number {
		return this.scale ? this.worst / this.scale : this.worst;
	}

	get readoutRmsRelative(): number {
		return this.norm ? Math.sqrt(this.squares / this.norm) : Math.sqrt(this.squares);
	}

	/** The car's largest difference over x, y, heading and speed, in units in the last place. */
	get carUlps(): number {
		return Math.max(...this.error.map((e, i) => e / ulp(this.size[i])));
	}
}

/** The spacing of float64 numbers at `x`'s magnitude: one unit in the last place. */
function ulp(x: number): number {
	const a = Math.abs(x);
	return a < 2 ** -1022 ? 2 ** -1074 : 2 ** (Math.floor(Math.log2(a)) - 52);
}
