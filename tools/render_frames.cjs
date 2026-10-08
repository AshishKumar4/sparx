// Rasterizes SVG files to PNG in headless Chromium, the browser Playwright installs.
//
//   NODE_PATH="$(npm root -g)" node tools/render_frames.cjs <out-dir> <scale> <file.svg>...
//
// Each SVG is drawn at its own width and height, times `scale`, on the page
// color of the GitHub theme its name ends in (-dark or -light; transparent
// otherwise), and saved as <out-dir>/<name>.png. tools/make_clips.py renders
// its frames through it, and the figures are checked by eye this way.
const { chromium } = require("playwright");
const { readFileSync } = require("node:fs");
const { basename, join } = require("node:path");

(async () => {
  const [outDir, scale, ...files] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: Number(scale) });
  for (const file of files) {
    const svg = readFileSync(file, "utf8");
    const [, width, height] = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
    await page.setViewportSize({ width: Math.ceil(Number(width)), height: Math.ceil(Number(height)) });
    const name = basename(file);
    const background = name.endsWith("-dark.svg") ? "#0d1117" : name.endsWith("-light.svg") ? "#ffffff" : "transparent";
    await page.setContent(`<html><body style="margin:0;background:${background}">${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready);
    const out = join(outDir, name.replace(/\.svg$/, ".png"));
    await page.locator("svg").screenshot({ path: out, omitBackground: background === "transparent" });
  }
  await browser.close();
})();
