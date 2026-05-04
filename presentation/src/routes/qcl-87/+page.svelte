<script lang="ts">
	import QclSlide from '$lib/QclSlide.svelte';

	let { data } = $props();
</script>

<QclSlide
	title="87 µm QCL"
	powerRamp="/plots/power_ramp_87um.png"
	hist="/plots/hist_87um.png"
	pcr="/plots/pcr_87um.png"
	metadata={data.meta}
/>
