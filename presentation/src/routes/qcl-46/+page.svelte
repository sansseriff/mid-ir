<script lang="ts">
	import QclSlide from '$lib/QclSlide.svelte';

	let { data } = $props();
</script>

<QclSlide
	title="46 µm QCL"
	powerRamp="/plots/power_ramp_46um.png"
	hist="/plots/hist_46um.png"
	pcr="/plots/pcr_46um.png"
	metadata={data.meta}
/>
