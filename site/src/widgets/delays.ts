// The delays page's figure: three inputs that each fire once, delayed through a DelayedDense onto one
// leaky integrator. The reader sets the delays, or lets gradient ascent on the readout's peak set them
// while the Gaussians narrow (src/engines/delays.ts).
import { peak, type Toy } from '../engines/delays';
import { alpha, animate, fit, onTheme, type Palette } from './theme';

export const TOY: Toy = { times: [5, 18, 30], weight: 0.6, maxDelay: 45, steps: 80, decay: Math.exp(-1 / 4) };
/** The step the readout is read at. */
export const READ_AT = 50;
const NAMES = ['A', 'B', 'C'];

export function delays(root: HTMLElement) {
	const canvas = root.querySelector('canvas') as HTMLCanvasElement;
	const status = root.querySelector<HTMLElement>('[data-status]');
	const learnButton = root.querySelector<HTMLButtonElement>('[data-act="learn"]') as HTMLButtonElement;
	const sliders = [0, 1, 2].map((i) => root.querySelector<HTMLInputElement>(`input[name="d${i}"]`) as HTMLInputElement);
	const sigmaSlider = root.querySelector<HTMLInputElement>('input[name="sigma"]') as HTMLInputElement;
	let d = sliders.map((s) => Number(s.value));
	let sigma = Number(sigmaSlider.value);
	let learning = false;
	let m = [0, 0, 0];
	let v2 = [0, 0, 0];
	let updates = 0;
	let colors: Palette;
	let view = fit(canvas);

	const show = () => {
		for (const [i, s] of sliders.entries()) {
			s.value = String(d[i]);
			const out = root.querySelector<HTMLOutputElement>(`output[data-for="d${i}"]`);
			if (out) out.textContent = `${d[i].toFixed(1)} steps`;
		}
		sigmaSlider.value = String(sigma);
		const out = root.querySelector<HTMLOutputElement>('output[data-for="sigma"]');
		if (out) out.textContent = sigma.toFixed(2);
	};

	function draw() {
		if (!colors) return;
		const { ctx, width, height } = view;
		ctx.clearRect(0, 0, width, height);
		const result = peak(TOY, d, sigma, READ_AT);
		const left = 28;
		const right = width - 12;
		const x = (t: number) => left + (t / (TOY.steps - 1)) * (right - left);
		const rowH = (height * 0.55) / 3;
		ctx.font = `500 11px ${getComputedStyle(canvas).getPropertyValue('--sx-mono')}`;
		ctx.textBaseline = 'middle';
		for (let i = 0; i < 3; i++) {
			const base = 14 + (i + 1) * rowH - 8;
			ctx.strokeStyle = alpha(colors.ink, 0.1);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(left, base + 0.5);
			ctx.lineTo(right, base + 0.5);
			ctx.stroke();
			ctx.fillStyle = colors.muted;
			ctx.fillText(NAMES[i], 6, base - 8);
			const s = TOY.times[i];
			ctx.strokeStyle = colors.spike;
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(x(s), base);
			ctx.lineTo(x(s), base - rowH * 0.7);
			ctx.stroke();
			const k = result.kernels[i];
			ctx.fillStyle = alpha(colors.learn, 0.75);
			const barW = Math.max(2, ((right - left) / TOY.steps) * 0.8);
			for (let lag = 0; lag < k.length; lag++) {
				if (k[lag] < 1e-3 || s + lag >= TOY.steps) continue;
				const h = Math.min(1, k[lag]) * rowH * 0.7;
				ctx.fillRect(x(s + lag) - barW / 2, base - h, barW, h);
			}
			const center = s + Math.min(Math.max(d[i], 0), TOY.maxDelay);
			ctx.setLineDash([2, 3]);
			ctx.strokeStyle = alpha(colors.learn, 0.6);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(x(s), base - rowH * 0.72);
			ctx.lineTo(x(center), base - rowH * 0.72);
			ctx.stroke();
			ctx.setLineDash([]);
		}
		const top = 14 + 3 * rowH + 10;
		const bottom = height - 18;
		const vmax = 2;
		const y = (v: number) => bottom - (Math.min(v, vmax) / vmax) * (bottom - top);
		ctx.fillStyle = colors.muted;
		ctx.fillText('readout', 6, top + 4);
		ctx.strokeStyle = colors.membrane;
		ctx.lineWidth = 1.75;
		ctx.beginPath();
		for (let t = 0; t < TOY.steps; t++) {
			if (t === 0) ctx.moveTo(x(t), y(result.v[t]));
			else ctx.lineTo(x(t), y(result.v[t]));
		}
		ctx.stroke();
		ctx.fillStyle = colors.membrane;
		ctx.beginPath();
		ctx.arc(x(result.at), y(result.v[result.at]), 3.5, 0, 2 * Math.PI);
		ctx.fill();
		ctx.fillStyle = colors.muted;
		ctx.textAlign = 'center';
		for (let t = 0; t < TOY.steps; t += 10) ctx.fillText(String(t), x(t), height - 7);
		ctx.textAlign = 'left';
		ctx.setLineDash([3, 3]);
		ctx.strokeStyle = alpha(colors.membrane, 0.6);
		ctx.beginPath();
		ctx.moveTo(x(READ_AT), 6);
		ctx.lineTo(x(READ_AT), height - 16);
		ctx.stroke();
		ctx.setLineDash([]);
		if (status) status.textContent = `readout at step ${READ_AT}: ${result.v[READ_AT].toFixed(3)} · sigma ${sigma.toFixed(2)}${sigma === 0 ? ', rounded as deployed' : ''}${updates ? ` · ${updates} updates` : ''}`;
		return result;
	}

	function update() {
		const result = peak(TOY, d, sigma, READ_AT);
		updates++;
		const lr = 0.6;
		d = d.map((x, i) => {
			const g = result.grad[i];
			m[i] = 0.9 * m[i] + 0.1 * g;
			v2[i] = 0.999 * v2[i] + 0.001 * g * g;
			const step = (lr * (m[i] / (1 - 0.9 ** updates))) / (Math.sqrt(v2[i] / (1 - 0.999 ** updates)) + 1e-8);
			return Math.min(Math.max(x - step, 0), TOY.maxDelay);
		});
		sigma = Math.max(0.5, sigma * (0.5 / 8) ** (1 / 260));
		if (updates > 260) {
			learning = false;
			sigma = 0;
			d = d.map((x) => Math.round(x));
			learnButton.textContent = 'Learn the delays';
		}
		show();
	}

	new ResizeObserver(() => {
		view = fit(canvas);
		draw();
	}).observe(canvas);
	onTheme((p) => {
		colors = p;
		draw();
	});
	for (const s of [...sliders, sigmaSlider]) {
		s.addEventListener('input', () => {
			d = sliders.map((x) => Number(x.value));
			sigma = Number(sigmaSlider.value);
			draw();
		});
	}
	learnButton.addEventListener('click', () => {
		learning = !learning;
		if (learning) {
			updates = 0;
			m = [0, 0, 0];
			v2 = [0, 0, 0];
			sigma = 8;
		}
		learnButton.textContent = learning ? 'Pause' : 'Learn the delays';
		show();
	});
	root.querySelector('[data-act="scramble"]')?.addEventListener('click', () => {
		d = [Math.random() * 45, Math.random() * 45, Math.random() * 45];
		sigma = 3;
		updates = 0;
		show();
		draw();
	});
	animate(canvas, () => {
		if (!learning) return;
		update();
		draw();
	});
}
