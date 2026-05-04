// Slide registry for the presentation. Each slide knows how many sub-steps
// it has (popups / progressive reveals) before advancing to the next slide.
//
// Behavior of arrow keys (handled in +layout.svelte):
//   right: increment step within slide; if already at subSteps, navigate next
//   left:  decrement step within slide; if at step 0, navigate prev (last step)

export type Slide = {
	path: string;
	title: string;
	subSteps: number;
};

export const slides: Slide[] = [
	{ path: '/title', title: 'Title', subSteps: 0 },
	{ path: '/qcl-46', title: '46 µm QCL', subSteps: 1 },
	{ path: '/qcl-63', title: '63 µm QCL', subSteps: 1 },
	{ path: '/qcl-87', title: '87 µm QCL', subSteps: 1 },
	{ path: '/combined', title: 'PCR — All QCL Devices', subSteps: 0 }
];

export function findSlide(pathname: string): { slide: Slide; index: number } | null {
	const idx = slides.findIndex((s) => s.path === pathname);
	if (idx === -1) return null;
	return { slide: slides[idx], index: idx };
}
