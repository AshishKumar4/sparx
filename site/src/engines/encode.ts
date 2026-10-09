// sparx.encode's encoders for one record, as the browser's figures use them. Latency and delta are
// deterministic and held to sparx by test/learning.test.ts, so they compute in float32 as sparx's do and
// round half to even as jnp.round does; rate draws its own random numbers.

const f32 = Math.fround;

function roundHalfEven(x: number): number {
	const r = Math.round(x);
	return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** `LatencyEncoder`: each value in [0, 1] fires once, at step `round((1 - x) * (steps - 1))`, if it reaches `threshold`. */
export function latency(x: number[], steps: number, threshold = 0.01): number[][] {
	return x.map((v) => {
		const value = Math.min(Math.max(f32(v), 0), 1);
		return value >= f32(threshold) ? [roundHalfEven(f32(f32(1 - value) * (steps - 1)))] : [];
	});
}

/** `RateEncoder`: each value fires at each step with probability `x`. */
export function rate(x: number[], steps: number, random: () => number): number[][] {
	return x.map((v) => {
		const p = Math.min(Math.max(v, 0), 1);
		const times: number[] = [];
		for (let t = 0; t < steps; t++) if (random() < p) times.push(t);
		return times;
	});
}

/** `DeltaEncoder` with `off_spikes`: +1 where the signal rose by at least `threshold` since the step before, -1 where it fell as far. */
export function delta(signal: number[], threshold: number): number[] {
	const limit = f32(threshold);
	return signal.map((v, t) => {
		const change = f32(f32(v) - f32(t ? signal[t - 1] : 0));
		return change >= limit ? 1 : change <= -limit ? -1 : 0;
	});
}

/** Send-on-delta, as an event camera's pixel: an event each time the signal moves `threshold` from the level of the last
 * event, which then moves by `threshold`. Returns the events and the level a receiver tracks. */
export function sendOnDelta(signal: number[], threshold: number): { events: number[]; level: number[] } {
	let level = 0;
	const events: number[] = [];
	const levels: number[] = [];
	for (const v of signal) {
		let e = 0;
		while (v - level >= threshold) {
			level += threshold;
			e++;
		}
		while (level - v >= threshold) {
			level -= threshold;
			e--;
		}
		events.push(e);
		levels.push(level);
	}
	return { events, level: levels };
}
