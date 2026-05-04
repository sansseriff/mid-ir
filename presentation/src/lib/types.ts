export type MetaItem = { label: string; value: string };
export type MetaGroups = Record<string, MetaItem[]>;

export type SlideMetadata = {
	device_label: string;
	power_ramp: { filename: string; groups: MetaGroups };
	hist: { n_files: number; filename_pattern: string | null; groups: MetaGroups };
	pcr: { filename: string; groups: MetaGroups };
};
