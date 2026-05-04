<script lang="ts">
</script>

<section class="combined">
	<h1>PCR — All QCL Devices</h1>
	<figure>
		<img src="/plots/pcr_combined.png" alt="Combined PCR plot" />
	</figure>
</section>

<style>
	.combined {
		display: flex;
		flex-direction: column;
		height: 100%;
		padding: 2.5rem 3rem 3rem;
		box-sizing: border-box;
		gap: 1.25rem;
	}

	h1 {
		margin: 0;
		font-size: clamp(1.6rem, 2.8vw, 2.6rem);
		font-weight: 600;
	}

	figure {
		margin: 0;
		flex: 1 1 auto;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #ffffff;
		border-radius: 8px;
		padding: 0.75rem;
		min-height: 0;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
	}

	figure img {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
	}
</style>
