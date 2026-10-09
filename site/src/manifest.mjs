// The docs: where each page's Markdown lives in the repository and the URL it gets.
// scripts/sync-docs.mjs copies them into Starlight's collection and astro.config.mjs
// builds the sidebar from the same list.

export const repository = {
	url: 'https://github.com/AshishKumar4/sparx',
	branch: 'main',
};

export const groups = [
	{
		items: [
			{ source: 'README.md', slug: 'docs', label: 'Overview', title: 'Overview' },
			{ source: 'docs/guide.md', slug: 'docs/guide', label: 'Guide' },
		],
	},
	{
		label: 'Tutorials',
		items: [
			{ source: 'docs/tutorials/train-and-deploy.md', slug: 'docs/tutorials/train-and-deploy', label: 'Train, serve and export' },
			{ source: 'docs/tutorials/nest-and-brian2.md', slug: 'docs/tutorials/nest-and-brian2', label: 'From NEST and Brian2' },
			{ source: 'docs/tutorials/fit-a-circuit.md', slug: 'docs/tutorials/fit-a-circuit', label: 'Fit a circuit' },
		],
	},
	{
		label: 'Reference',
		items: [
			{ source: 'docs/status.md', slug: 'docs/status', label: 'Status' },
			{ source: 'docs/fidelity.md', slug: 'docs/fidelity', label: 'Fidelity ledger' },
			{ source: 'docs/units.md', slug: 'docs/units', label: 'Units' },
			{ source: 'docs/performance.md', slug: 'docs/performance', label: 'Performance' },
			{ source: 'docs/design.md', slug: 'docs/design', label: 'Design' },
		],
	},
	{ label: 'API', generated: 'api' },
];

export const pages = groups.flatMap((group) => group.items ?? []);
