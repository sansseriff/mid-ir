<script lang="ts">
	import QclSlide from '$lib/QclSlide.svelte';

	let { data } = $props();
</script>

<QclSlide
	title="63 µm QCL"
	powerRamp="/plots/power_ramp_63um.png"
	hist="/plots/hist_63um.png"
	pcr="/plots/pcr_63um.png"
	metadata={data.meta}
/>
