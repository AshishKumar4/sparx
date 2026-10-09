import { existsSync, readFileSync } from 'node:fs';
import starlight from '@astrojs/starlight';
import { defineConfig, fontProviders } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import starlightLinksValidator from 'starlight-links-validator';
import { groups, repository } from './src/manifest.mjs';

function generated(name) {
	const file = new URL(`./src/generated/${name}.json`, import.meta.url);
	if (!existsSync(file)) throw new Error(`src/generated/${name}.json is missing: run \`pnpm content\` first`);
	return JSON.parse(readFileSync(file, 'utf8'));
}

const sidebar = groups.map((group) => {
	const items = (group.generated ? generated(group.generated) : group.items).map((item) => ({ label: item.label, slug: item.slug }));
	return group.label ? { label: group.label, collapsed: group.generated !== undefined, items } : items;
}).flat();

// Each family's Latin file only, which covers the site's text, so a page preloads exactly one
// file per family; Astro gives each a size-adjusted fallback so nothing moves when it arrives.
function latin(pkg, file) {
	const css = readFileSync(new URL(`./node_modules/${pkg}/${file}`, import.meta.url), 'utf8');
	const face = /\/\* [\w-]+-latin-[a-z]+-normal \*\/\s*@font-face\s*{([^}]*)}/.exec(css);
	if (!face) throw new Error(`${pkg}/${file} has no Latin face`);
	const property = (name) => new RegExp(`${name}:\\s*([^;]+);`).exec(face[1])[1].trim();
	return {
		src: [`${pkg}/files/${/url\(\.\/files\/([^)]+)\)/.exec(face[1])[1]}`],
		weight: property('font-weight'),
		style: 'normal',
		unicodeRange: property('unicode-range').split(','),
	};
}

const fonts = [
	['Inter Variable', '--font-sans', '@fontsource-variable/inter', 'opsz.css', ['ui-sans-serif', 'system-ui', 'sans-serif']],
	['JetBrains Mono Variable', '--font-mono', '@fontsource-variable/jetbrains-mono', 'wght.css', ['ui-monospace', 'monospace']],
].map(([name, cssVariable, pkg, file, fallbacks]) => ({
	provider: fontProviders.local(),
	name,
	cssVariable,
	fallbacks,
	options: { variants: [latin(pkg, file)] },
}));

export default defineConfig({
	site: 'https://sparxml.dev',
	trailingSlash: 'always',
	fonts,
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			title: 'sparx',
			description: 'Spiking neural networks in JAX: train them with gradients or local rules, and simulate circuits of biological neurons in millivolts and milliseconds.',
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: repository.url }],
			lastUpdated: true,
			pagination: true,
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			customCss: ['katex/dist/katex.min.css', './src/styles/theme.css'],
			expressiveCode: {
				themes: ['github-dark-default', 'github-light-default'],
				useStarlightUiThemeColors: true,
				styleOverrides: {
					borderRadius: '0.375rem',
					codeFontFamily: 'var(--sx-mono)',
					codeFontSize: '0.8125rem',
					codeLineHeight: '1.65',
					uiFontFamily: 'var(--sx-sans)',
				},
			},
			components: {
				Head: './src/components/starlight/Head.astro',
				Header: './src/components/starlight/Header.astro',
			},
			head: [
				{ tag: 'meta', attrs: { name: 'theme-color', content: '#0b0d10', media: '(prefers-color-scheme: dark)' } },
				{ tag: 'meta', attrs: { name: 'theme-color', content: '#fcfcfb', media: '(prefers-color-scheme: light)' } },
				{ tag: 'meta', attrs: { property: 'og:image', content: 'https://sparxml.dev/og.png' } },
				{ tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
			],
			sidebar,
			routeMiddleware: './src/route-data.mjs',
			plugins: [
				starlightLinksValidator({
					errorOnFallbackPages: true,
					errorOnInvalidHashes: true,
					errorOnLocalLinks: true,
					exclude: ['/learn/**', '/'],
				}),
			],
		}),
	],
});
