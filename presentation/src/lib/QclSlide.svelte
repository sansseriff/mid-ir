<script lang="ts">
	import { page } from '$app/state';
	import Metadata from './Metadata.svelte';
	import type { SlideMetadata } from './types';

	type Props = {
		title: string;
		powerRamp: string;
		hist: string;
		pcr: string;
		metadata: SlideMetadata;
	};

	let { title, powerRamp, hist, pcr, metadata }: Props = $props();

	let step = $derived.by(() => {
		const v = page.url.searchParams.get('step');
		const n = v ? parseInt(v, 10) : 0;
		return Number.isFinite(n) && n >= 0 ? n : 0;
	});

	let pcrVisible = $derived(step >= 1);
</script>

<section class="qcl-slide">
	<h1>{title}</h1>
	<div class="grid">
		<figure>
			<img src={powerRamp} alt="{title} power ramp" />
			<Metadata groups={metadata.power_ramp.groups} />
		</figure>
		<figure>
			<img src={hist} alt="{title} histogram overlay" />
			<Metadata groups={metadata.hist.groups} />
		</figure>
	</div>

	{#if pcrVisible}
		<div class="overlay">
			<figure class="pcr-card">
				<img src={pcr} alt="{title} PCR" />
				<Metadata groups={metadata.pcr.groups} />
			</figure>
		</div>
	{/if}
</section>

<style>
	.qcl-slide {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding: 1.5rem 2rem 1.75rem;
		box-sizing: border-box;
		gap: 0.9rem;
	}

	h1 {
		margin: 0;
		font-size: clamp(1.4rem, 2.4vw, 2.2rem);
		font-weight: 600;
		letter-spacing: 0.01em;
	}

	.grid {
		flex: 1 1 auto;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.25rem;
		min-height: 0;
	}

	figure {
		margin: 0;
		display: flex;
		flex-direction: column;
		background: #ffffff;
		border-radius: 8px;
		padding: 0.6rem 0.75rem 0.5rem;
		min-height: 0;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
		overflow: hidden;
	}

	figure img {
		flex: 1 1 auto;
		min-height: 0;
		max-width: 100%;
		object-fit: contain;
	}

	.overlay {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(8, 10, 14, 0.78);
		backdrop-filter: blur(2px);
		animation: fade-in 180ms ease-out;
	}

	.pcr-card {
		width: min(80vw, 1200px);
		height: min(86vh, 820px);
		padding: 0.85rem 1rem 0.75rem;
		border-radius: 12px;
		box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6);
		animation: pop-in 220ms ease-out;
	}

	@keyframes fade-in {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	@keyframes pop-in {
		from {
			transform: scale(0.92);
			opacity: 0;
		}
		to {
			transform: scale(1);
			opacity: 1;
		}
	}
</style>
