// The site's colors as canvas code reads them, refreshed when the theme changes.

export interface Palette {
	page: string;
	panel: string;
	line: string;
	lineStrong: string;
	ink: string;
	ink2: string;
	muted: string;
	faint: string;
	spike: string;
	membrane: string;
	bio: string;
	learn: string;
	dark: boolean;
}

export function palette(): Palette {
	const css = getComputedStyle(document.documentElement);
	const read = (name: string) => css.getPropertyValue(`--sx-${name}`).trim();
	return {
		page: read('page'),
		panel: read('panel'),
		line: read('line'),
		lineStrong: read('line-strong'),
		ink: read('ink'),
		ink2: read('ink-2'),
		muted: read('muted'),
		faint: read('faint'),
		spike: read('spike'),
		membrane: read('membrane'),
		bio: read('bio'),
		learn: read('learn'),
		dark: document.documentElement.dataset.theme !== 'light',
	};
}

/** Calls `apply` now and whenever the page's theme changes. */
export function onTheme(apply: (colors: Palette) => void): void {
	apply(palette());
	new MutationObserver(() => apply(palette())).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

/** `color` (#rrggbb) at `alpha`. */
export function alpha(color: string, a: number): string {
	const n = Number.parseInt(color.slice(1), 16);
	return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** A canvas sized to its box at the device's pixel ratio, capped at 2; returns the context in CSS pixels. */
export function fit(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; width: number; height: number } {
	const ratio = Math.min(devicePixelRatio || 1, 2);
	const { width, height } = canvas.getBoundingClientRect();
	canvas.width = Math.max(1, Math.round(width * ratio));
	canvas.height = Math.max(1, Math.round(height * ratio));
	const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
	ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
	return { ctx, width, height };
}

/** Runs `frame(seconds since the last frame)` while `element` is on screen and the tab is visible. */
export function animate(element: Element, frame: (dt: number) => void): { running: () => boolean } {
	let visible = false;
	let handle = 0;
	let last = 0;
	const tick = (now: number) => {
		frame(Math.min((now - last) / 1000, 0.1));
		last = now;
		handle = requestAnimationFrame(tick);
	};
	const update = () => {
		const run = visible && !document.hidden;
		if (run && !handle) {
			last = performance.now();
			handle = requestAnimationFrame(tick);
		} else if (!run && handle) {
			cancelAnimationFrame(handle);
			handle = 0;
		}
	};
	new IntersectionObserver(([entry]) => {
		visible = entry.isIntersecting;
		update();
	}).observe(element);
	document.addEventListener('visibilitychange', update);
	return { running: () => handle !== 0 };
}

/** Calls `start` once, when `element` comes within `margin` of the screen, so a figure costs nothing until a
 * reader scrolls toward it. */
export function whenNear(element: Element, start: () => unknown, margin = '300px'): void {
	new IntersectionObserver(
		([entry], observer) => {
			if (!entry.isIntersecting) return;
			observer.disconnect();
			Promise.resolve(start()).catch((error) => console.error(error));
		},
		{ rootMargin: margin },
	).observe(element);
}
