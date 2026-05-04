<script lang="ts">
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { slides } from '$lib/slides';

	onMount(() => {
		goto(slides[0].path, { replaceState: true });
	});
</script>

<div class="splash">
	<p>Loading presentation…</p>
</div>

<style>
	.splash {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
		font-size: 1.1rem;
		opacity: 0.7;
	}
</style>
