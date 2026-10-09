export default {
	fetch(request) {
		const url = new URL(request.url);
		return Response.redirect(`https://sparxml.dev${url.pathname}${url.search}`, 301);
	},
};
