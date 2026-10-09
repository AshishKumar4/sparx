// The wordmark's dot: one LIF neuron, v <- 0.95 v + x with x a noisy current, firing at 1 and resetting
// to 0, stepped every 25 ms. It flashes on each spike, a few times a second.
const dots = document.querySelectorAll<HTMLElement>('[data-neuron]');
if (dots.length && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
	let v = 0;
	setInterval(() => {
		if (document.hidden) return;
		v = 0.95 * v + 0.012 + 0.05 * (Math.random() - 0.5);
		if (v < 1) return;
		v = 0;
		for (const dot of dots) {
			dot.classList.add('fired');
			setTimeout(() => dot.classList.remove('fired'), 90);
		}
	}, 25);
}
