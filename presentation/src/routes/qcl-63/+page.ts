import type { SlideMetadata } from '$lib/types';

export const load = async ({ fetch }) => {
	const meta: SlideMetadata = await fetch('/plots/metadata_63um.json').then((r) => r.json());
	return { meta };
};
