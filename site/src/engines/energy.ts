// Energy per synapse per inference, from Horowitz's (ISSCC 2014) table of rough costs at 45 nm and 0.9 V,
// Figure 1.1.9. A conventional synapse reads its weight once and multiplies and adds; a spiking synapse
// reads its weight and adds, once for each spike that crosses it. Neuron updates are left out.

export const OPS = {
	fp32: { mac: 3.7 + 0.9, ac: 0.9, label: '32-bit float' },
	int8: { mac: 0.2 + 0.03, ac: 0.03, label: '8-bit integer' },
} as const;

export const MEMORY = {
	none: { pj: 0, label: 'Free (no memory cost)' },
	sram8k: { pj: 10, label: '8 KB cache' },
	sram1m: { pj: 100, label: '1 MB cache' },
	dram: { pj: 1300, label: 'DRAM' },
} as const;

export type Precision = keyof typeof OPS;
export type Memory = keyof typeof MEMORY;

/** pJ per synapse per inference for a conventional network and a spiking one at `spikes` spikes per synapse. */
export function energy(precision: Precision, memory: Memory, spikes: number) {
	const op = OPS[precision];
	const read = MEMORY[memory].pj;
	const ann = op.mac + read;
	const snn = spikes * (op.ac + read);
	return { ann, snn, breakEven: ann / (op.ac + read) };
}
