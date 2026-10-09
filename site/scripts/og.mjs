// public/og.png, the link preview: the front page's first screen at 1200 by 630, dark, after the pilot
// has flown for a few seconds. Run against a served build (scripts/shots.mjs says how).
import { chromium } from 'playwright-core';

const base = process.env.BASE ?? 'http://localhost:4329';
const browser = await chromium.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome' });
const context = await browser.newContext({ viewport: { width: 1200, height: 630 }, colorScheme: 'dark', deviceScaleFactor: 1 });
await context.addInitScript(() => localStorage.setItem('starlight-theme', 'dark'));
const page = await context.newPage();
await page.goto(`${base}/`, { waitUntil: 'networkidle' });
await page.addStyleTag({ content: 'header.header { border: 0 } .header .search, .header .sections, .header .right { visibility: hidden } .hero { height: calc(630px - var(--sl-nav-height)) !important; min-height: 0 !important } .panel, .actions { visibility: hidden }' });
await page.evaluate(() => dispatchEvent(new Event('resize')));
await page.waitForTimeout(4000);
await page.screenshot({ path: new URL('../public/og.png', import.meta.url).pathname });
await browser.close();
