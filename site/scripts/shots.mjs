// Screenshots of the built site at 1440 and 390 px, dark and light, for review:
//
//     pnpm exec astro preview --port 4329 &
//     node scripts/shots.mjs [out-dir] [path...]
//
// Each page is shot after its live figures have run for a moment. Chrome comes from CHROME, or
// /usr/bin/google-chrome.
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const [out = '/tmp/sparx-shots', ...paths] = process.argv.slice(2);
const pages = paths.length ? paths : ['/', '/learn/', '/docs/', '/docs/guide/', '/docs/api/sparx.nn/'];
const base = process.env.BASE ?? 'http://localhost:4329';
const widths = [
	{ name: 'desktop', width: 1440, height: 900, mobile: false },
	{ name: 'mobile', width: 390, height: 844, mobile: true },
];

await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome', args: ['--enable-unsafe-swiftshader'] });
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
		page.on('pageerror', (error) => errors.push(error.message));
		page.on('console', (message) => message.type() === 'error' && errors.push(message.text()));
		for (const path of pages) {
			await page.goto(base + path, { waitUntil: 'networkidle' });
			await page.waitForTimeout(Number(process.env.SETTLE ?? 2500));
			const name = `${path.replace(/^\/|\/$/g, '').replace(/\//g, '_') || 'home'}-${size.name}-${theme}`;
			await page.screenshot({ path: `${out}/${name}.png`, fullPage: process.env.FULL !== '0' });
			console.log(`${out}/${name}.png${errors.length ? `  errors: ${errors.join(' | ')}` : ''}`);
			errors.length = 0;
		}
		await context.close();
	}
}
await browser.close();
