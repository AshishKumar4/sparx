// The sliders and choices of a live figure, read by name; each shows its value as it moves.

export function controls(root: HTMLElement, changed: (name: string) => void = () => {}) {
	const read = (name: string): number => {
		const input = root.querySelector<HTMLInputElement>(`input[name="${name}"]`);
		if (!input) throw new Error(`no control named ${name}`);
		return Number(input.value);
	};
	for (const input of root.querySelectorAll<HTMLInputElement>('input[type="range"]')) {
		const output = root.querySelector<HTMLOutputElement>(`output[data-for="${input.name}"]`);
		input.addEventListener('input', () => {
			if (output) {
				const unit = output.dataset.unit ? ` ${output.dataset.unit}` : '';
				output.textContent = `${Number(input.value).toFixed(Number(output.dataset.digits ?? 2))}${unit}`;
			}
			changed(input.name);
		});
	}
	for (const group of root.querySelectorAll<HTMLElement>('[role="radiogroup"]')) {
		for (const button of group.querySelectorAll<HTMLButtonElement>('button')) {
			button.addEventListener('click', () => {
				for (const other of group.querySelectorAll('button')) other.setAttribute('aria-checked', String(other === button));
				changed(group.dataset.name ?? '');
			});
		}
	}
	const choice = (name: string): string =>
		root.querySelector<HTMLButtonElement>(`[role="radiogroup"][data-name="${name}"] [aria-checked="true"]`)?.value ?? '';
	return { read, choice };
}
