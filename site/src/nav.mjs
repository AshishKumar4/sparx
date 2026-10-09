// The header's sections. A page belongs to the section whose prefix is the longest match of its path.

export const sections = [
	{ key: 'learn', label: 'Learn', href: '/learn/' },
	{ key: 'docs', label: 'Docs', href: '/docs/' },
	{ key: 'api', label: 'API', href: '/docs/api/' },
];

export function sectionOf(pathname) {
	const path = pathname.endsWith('/') ? pathname : `${pathname}/`;
	const matches = sections.filter((section) => path.startsWith(section.href));
	return matches.sort((a, b) => b.href.length - a.href.length)[0]?.key;
}
