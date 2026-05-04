<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { findSlide, slides } from '$lib/slides';

	let { children } = $props();

	function currentStep(): number {
		const v = page.url.searchParams.get('step');
		const n = v ? parseInt(v, 10) : 0;
		return Number.isFinite(n) && n >= 0 ? n : 0;
	}

	function navTo(path: string, step: number) {
		const url = step > 0 ? `${path}?step=${step}` : path;
		goto(url, { noScroll: true, keepFocus: true });
	}

	function advance() {
		const ctx = findSlide(page.url.pathname);
		if (!ctx) {
			navTo(slides[0].path, 0);
			return;
		}
		const step = currentStep();
		if (step < ctx.slide.subSteps) {
			navTo(ctx.slide.path, step + 1);
			return;
		}
		const next = slides[ctx.index + 1];
		if (next) navTo(next.path, 0);
	}

	function retreat() {
		const ctx = findSlide(page.url.pathname);
		if (!ctx) return;
		const step = currentStep();
		if (step > 0) {
			navTo(ctx.slide.path, step - 1);
			return;
		}
		const prev = slides[ctx.index - 1];
		if (prev) navTo(prev.path, prev.subSteps);
	}

	function onKeydown(e: KeyboardEvent) {
		// Don't hijack typing in inputs or when modifier keys are held.
		const target = e.target as HTMLElement | null;
		if (target && ['INPUT', 'TEXTAREA'].includes(target.tagName)) return;
		if (e.metaKey || e.ctrlKey || e.altKey) return;

		if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
			e.preventDefault();
			advance();
		} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
			e.preventDefault();
			retreat();
		}
	}
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>
<svelte:window on:keydown={onKeydown} />

<div class="slide-shell">
	{@render children()}
	<div class="slide-indicator">
		{#if findSlide(page.url.pathname)}
			{findSlide(page.url.pathname)!.index + 1} / {slides.length}
		{/if}
	</div>
</div>

<style>
	.slide-shell {
		position: relative;
		width: 100vw;
		height: 100vh;
		overflow: hidden;
		background: #0e1116;
		color: #f5f6f8;
	}

	.slide-indicator {
		position: fixed;
		bottom: 0.6rem;
		right: 1rem;
		font-size: 0.85rem;
		font-family: ui-monospace, Menlo, monospace;
		opacity: 0.55;
		letter-spacing: 0.05em;
		pointer-events: none;
	}
</style>
