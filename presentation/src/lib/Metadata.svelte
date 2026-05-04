<script lang="ts">
	import type { MetaGroups } from './types';

	type Props = {
		groups: MetaGroups;
	};

	let { groups }: Props = $props();
</script>

<div class="meta">
	{#each Object.entries(groups) as [group, items] (group)}
		<div class="group">
			<span class="group-name">{group}</span>
			{#each items as item, i (item.label)}
				<span class="item">
					<span class="item-label">{item.label}</span>
					<span class="item-value">{item.value}</span>
				</span>
				{#if i < items.length - 1}<span class="sep">·</span>{/if}
			{/each}
		</div>
	{/each}
</div>

<style>
	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1rem;
		font-size: 0.72rem;
		color: #1f2530;
		font-family: ui-monospace, Menlo, monospace;
		padding: 0.25rem 0.4rem 0;
		line-height: 1.35;
	}

	.group {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.25rem 0.45rem;
	}

	.group-name {
		font-weight: 700;
		letter-spacing: 0.04em;
		color: #444b58;
		padding-right: 0.15rem;
	}

	.item {
		display: inline-flex;
		gap: 0.25rem;
	}

	.item-label {
		color: #6b7280;
	}

	.item-value {
		color: #111827;
	}

	.sep {
		color: #c7cad0;
	}
</style>
