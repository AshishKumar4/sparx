// The site's files with X-Robots-Tag: noindex, so search engines leave the preview alone.
export default {
	async fetch(request, env) {
		const response = await env.ASSETS.fetch(request);
		const headers = new Headers(response.headers);
		headers.set('X-Robots-Tag', 'noindex');
		return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
	},
};
