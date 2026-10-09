// An event camera's pixels, as the racer's car carries them (site/lab/racer.py, `sense`): a pixel fires ON
// when the logarithm of the brightness it sees has risen `threshold` above the level of its last event, OFF
// when it has fallen as far, and each event moves its level by the threshold. A still scene sends nothing.

export class EventCamera {
	readonly level: Float64Array;
	readonly on: Uint8Array;
	readonly off: Uint8Array;

	constructor(
		readonly pixels: number,
		public threshold = 0.15,
	) {
		this.level = new Float64Array(pixels);
		this.on = new Uint8Array(pixels);
		this.off = new Uint8Array(pixels);
	}

	/** Sets every pixel's level to the log brightness it sees, as a camera switched on in front of a scene. */
	reset(seen: ArrayLike<number>): void {
		this.level.set(seen);
		this.on.fill(0);
		this.off.fill(0);
	}

	/** One step on the log brightness each pixel sees; returns how many events it sent. */
	step(seen: ArrayLike<number>): number {
		let count = 0;
		for (let p = 0; p < this.pixels; p++) {
			const change = seen[p] - this.level[p];
			const on = change - this.threshold >= 0 ? 1 : 0;
			const off = -change - this.threshold >= 0 ? 1 : 0;
			this.level[p] += this.threshold * (on - off);
			this.on[p] = on;
			this.off[p] = off;
			count += on + off;
		}
		return count;
	}
}
