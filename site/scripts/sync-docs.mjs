// Copy the docs pages of src/manifest.mjs into Starlight's collection. A page's title is its first
// heading; links between docs pages become site URLs, links to other repository files go to GitHub,
// and images are copied to public/repo. A <picture> with a dark and a light source becomes two
// images that follow the site's theme rather than the system's.
import { execFileSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import { pages, repository } from '../src/manifest.mjs';

const site = path.resolve(import.meta.dirname, '..');
const repo = path.resolve(site, '..');
const content = path.join(site, 'src/content/docs/docs');
const images = path.join(site, 'public/repo');
const slugs = new Map(pages.map((page) => [page.source, page.slug]));
const raw = `https://raw.githubusercontent.com/${new URL(repository.url).pathname.slice(1)}/${repository.branch}/`;
const parser = unified().use(remarkParse).use(remarkGfm);

async function target(url, source) {
	if (url.startsWith(raw)) url = `/${url.slice(raw.length)}`;
	if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('#') || url.startsWith('//')) return url;
	const [file, hash] = url.split('#', 2);
	const resolved = file.startsWith('/') ? file.slice(1) : path.posix.normalize(path.posix.join(path.posix.dirname(source), decodeURIComponent(file)));
	const anchor = hash ? `#${hash}` : '';
	if (slugs.has(resolved)) return `/${slugs.get(resolved)}/${anchor}`;
	const on = path.join(repo, resolved);
	if (resolved.startsWith('..') || !existsSync(on)) throw new Error(`${source}: broken link ${url}`);
	if (/\.(png|jpe?g|gif|svg|webp)$/i.test(resolved)) {
		await mkdir(path.dirname(path.join(images, resolved)), { recursive: true });
		await copyFile(on, path.join(images, resolved));
		return `/repo/${resolved}`;
	}
	if (resolved.startsWith('docs/') && resolved.endsWith('.md')) throw new Error(`${source}: ${url} is a docs page missing from src/manifest.mjs`);
	return `${repository.url}/${statSync(on).isDirectory() ? 'tree' : 'blob'}/${repository.branch}/${resolved}${anchor}`;
}

function pictures(markdown) {
	return markdown.replace(/<picture>\s*<source media="\(prefers-color-scheme: dark\)" srcset="([^"]+)">\s*<img([^>]*?)src="([^"]+)"([^>]*)>\s*<\/picture>/g,
		(_, dark, before, light, after) => {
			const attrs = `${before}${after}`.replace(/\s*width="[^"]*"/, '').trim();
			return `<img class="light:sl-hidden" src="${dark}" ${attrs} loading="lazy"><img class="dark:sl-hidden" src="${light}" ${attrs} loading="lazy">`;
		});
}

async function rewrite(markdown, source) {
	const tree = parser.parse(markdown);
	const edits = [];
	visit(tree, ['link', 'image', 'definition'], (node) => {
		const { start, end } = node.position;
		const text = markdown.slice(start.offset, end.offset);
		const marker = node.type === 'definition' ? text.indexOf(']:') : text.lastIndexOf('](');
		const at = marker < 0 ? -1 : text.indexOf(node.url, marker);
		if (at >= 0) edits.push({ from: start.offset + at, to: start.offset + at + node.url.length, url: node.url });
	});
	visit(tree, 'html', (node) => {
		for (const match of node.value.matchAll(/\b(src|href|srcset)="([^"]+)"/g)) {
			const from = node.position.start.offset + match.index + match[1].length + 2;
			edits.push({ from, to: from + match[2].length, url: match[2] });
		}
	});
	edits.sort((a, b) => b.from - a.from);
	let out = markdown;
	for (const edit of edits) out = out.slice(0, edit.from) + (await target(edit.url, source)) + out.slice(edit.to);
	return out;
}

function describe(body) {
	const paragraph = parser.parse(body).children.find((node) => node.type === 'paragraph');
	let text = '';
	if (paragraph) visit(paragraph, (node) => { if (node.type === 'text' || node.type === 'inlineCode') text += node.value; });
	text = text.replace(/\s+/g, ' ').trim();
	if (text.length <= 200) return text;
	const cut = text.slice(0, 200);
	const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('; '));
	return end > 60 ? cut.slice(0, end + 1) : `${cut.slice(0, cut.lastIndexOf(' '))}…`;
}

await rm(content, { recursive: true, force: true });
await rm(images, { recursive: true, force: true });
for (const page of pages) {
	let markdown = await readFile(path.join(repo, page.source), 'utf8');
	let title = page.title;
	if (title) {
		markdown = markdown.replace(/^\s*<picture>[\s\S]*?<\/picture>\s*/, '');
	} else {
		const heading = /^# (.+)\n/m.exec(markdown);
		if (!heading) throw new Error(`${page.source}: no first-level heading for the title`);
		title = heading[1].trim();
		markdown = markdown.slice(0, heading.index) + markdown.slice(heading.index + heading[0].length);
	}
	const body = pictures(await rewrite(markdown.trimStart(), page.source));
	const updated = execFileSync('git', ['log', '-1', '--format=%cI', '--', page.source], { cwd: repo, encoding: 'utf8' }).trim();
	const front = {
		title,
		description: describe(body),
		editUrl: `${repository.url}/edit/${repository.branch}/${page.source}`,
		...(updated && { lastUpdated: updated }),
	};
	const file = path.join(site, 'src/content/docs', `${page.slug}.md`);
	await mkdir(path.dirname(file), { recursive: true });
	await writeFile(file, `---\n${Object.entries(front).map(([k, v]) => `${k}: ${k === 'lastUpdated' ? v : JSON.stringify(v)}`).join('\n')}\n---\n\n${body.trimEnd()}\n`);
}
console.log(`sync-docs: ${pages.length} pages`);
