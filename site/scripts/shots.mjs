// Screenshots of the built site at 1440 and 390 px, dark and light, and a check of each page: no script
// errors, and nothing wider than a phone. Site CI runs it and uploads the screenshots:
//
//     pnpm exec astro preview --port 4329 &
//     node scripts/shots.mjs [out-dir] [path...]
//
// With no paths it shoots every page of dist/ outside the API reference. Each page is shot after its live
// figures have run for a moment. Chrome comes from CHROME, or /usr/bin/google-chrome. Exits 1 if any page
// threw or overflowed.
import { mkdir, readdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const [out = 'shots', ...paths] = process.argv.slice(2);
const base = process.env.BASE ?? 'http://localhost:4329';
const dist = new URL('../dist/', import.meta.url);
const pages = paths.length
	? paths
	: (await readdir(dist, { recursive: true }))
			.filter((file) => file.endsWith('index.html') && !file.includes('docs/api/') && !file.startsWith('repo/'))
			.map((file) => `/${file.slice(0, -'index.html'.length)}`)
			.sort();
const widths = [
	{ name: 'desktop', width: 1440, height: 900, mobile: false },
	{ name: 'mobile', width: 390, height: 844, mobile: true },
];

await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome', args: ['--enable-unsafe-swiftshader'] });
let failed = 0;
for (const size of widths) {
	for (const theme of ['dark', 'light']) {
		const context = await browser.newContext({
			viewport: { width: size.width, height: size.height },
			deviceScaleFactor: size.mobile ? 2 : 1,
			isMobile: size.mobile,
			hasTouch: size.mobile,
			colorScheme: theme,
		});
		await context.addInitScript((t) => localStorage.setItem('starlight-theme', t), theme);
		const page = await context.newPage();
		const errors = [];
		const notes = [];
		page.on('pageerror', (error) => errors.push(error.message));
		page.on('console', (message) => message.type() === 'error' && notes.push(message.text()));
		for (const path of pages) {
			await page.goto(base + path, { waitUntil: 'networkidle' });
			await page.waitForTimeout(Number(process.env.SETTLE ?? 2500));
			const name = `${path.replace(/^\/|\/$/g, '').replace(/\//g, '_') || 'home'}-${size.name}-${theme}`;
			await page.screenshot({ path: `${out}/${name}.png`, fullPage: process.env.FULL !== '0' });
			const overflow = await page.evaluate(() => {
				const width = document.documentElement.clientWidth;
				const wide = [...document.querySelectorAll('main *')].filter((el) => {
					const r = el.getBoundingClientRect();
					return r.width > 0 && r.right > width + 1 && !el.closest('.expressive-code, .katex-display, table, .sl-markdown-content pre');
				});
				return wide.slice(0, 5).map((el) => `${el.tagName.toLowerCase()}.${[...el.classList].join('.')}`);
			});
			const problems = [...errors, ...overflow.map((el) => `wider than the page: ${el}`)];
			failed += problems.length ? 1 : 0;
			console.log(`${name}${problems.length ? `  FAILED: ${problems.join(' | ')}` : ''}${notes.length ? `  console: ${notes.join(' | ')}` : ''}`);
			errors.length = 0;
			notes.length = 0;
		}
		await context.close();
	}
}
await browser.close();
console.log(`${pages.length} pages at ${widths.length * 2} sizes and themes; ${failed} failed`);
process.exit(failed ? 1 : 0);
