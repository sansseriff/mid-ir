import argparse
import json
from pathlib import Path


# Coarse-graining factor for x/y. Use 2, 4, 8... to reduce noise.
# This is the default; you can override at runtime with --reduce.
DEFAULT_REDUCTION_FACTOR = 1


def load_correlation_json(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if "x_axis_ps" not in data or "histogram_counts" not in data:
		raise KeyError("Expected keys 'x_axis_ps' and 'histogram_counts' in JSON")

	x = data["x_axis_ps"]
	y = data["histogram_counts"]
	if not isinstance(x, list) or not isinstance(y, list):
		raise TypeError("'x_axis_ps' and 'histogram_counts' must be lists")
	if len(x) != len(y):
		raise ValueError(f"Length mismatch: len(x_axis_ps)={len(x)} vs len(histogram_counts)={len(y)}")

	return data


def coarse_grain_xy(x: list[float], y: list[float], factor: int) -> tuple[list[float], list[float]]:
	"""Coarse-bin consecutive points by `factor`.

	- x becomes the average x within each block
	- y becomes the sum within each block (appropriate for histogram counts)
	"""
	if factor <= 1:
		return x, y
	if factor < 1:
		raise ValueError("factor must be >= 1")

	n = min(len(x), len(y))
	n_trim = (n // factor) * factor
	if n_trim == 0:
		raise ValueError("Reduction factor is larger than the data length")

	x_out: list[float] = []
	y_out: list[float] = []
	for i in range(0, n_trim, factor):
		x_block = x[i : i + factor]
		y_block = y[i : i + factor]
		x_out.append(sum(x_block) / factor)
		y_out.append(sum(y_block))

	return x_out, y_out


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot correlation histogram from JSON.")
	parser.add_argument(
		"json_path",
		nargs="?",
		default="correlation_QCL_330mA.json",
		help="Path to the correlation JSON (default: correlation_QCL_330mA.json)",
	)
	parser.add_argument(
		"--out",
		default=None,
		help="Output image path (default: <json_stem>.png next to the JSON)",
	)
	parser.add_argument(
		"--reduce",
		type=int,
		default=DEFAULT_REDUCTION_FACTOR,
		help="Coarse-grain factor for binning (e.g. 2, 4, 8). 1 disables.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Show an interactive plot window (in addition to saving PNG).",
	)
	args = parser.parse_args()

	if args.reduce < 1:
		raise SystemExit("--reduce must be >= 1")

	json_path = Path(args.json_path).expanduser().resolve()
	data = load_correlation_json(json_path)

	x_ps = data["x_axis_ps"]
	y_counts = data["histogram_counts"]

	x_plot, y_plot = coarse_grain_xy(x_ps, y_counts, args.reduce)

	# Import matplotlib lazily so simply importing this module doesn't require it.
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
	ax.plot(x_plot, y_plot, linewidth=1.0)

	title_parts = [json_path.name]
	if "integration_time_value" in data:
		title_parts.append(f"integration={data['integration_time_value']}")
	if "binwidth_ps" in data:
		try:
			binwidth_ps = float(data["binwidth_ps"])
		except (TypeError, ValueError):
			binwidth_ps = None
		if binwidth_ps is not None:
			title_parts.append(f"binwidth_ps={binwidth_ps * args.reduce:g}")
		else:
			title_parts.append(f"binwidth_ps={data['binwidth_ps']}")
	if args.reduce != 1:
		title_parts.append(f"reduce={args.reduce}x")

	ax.set_title(" | ".join(title_parts))
	ax.set_xlabel("Delay (ps)")
	ax.set_ylabel("Counts (coarse-binned)")
	ax.grid(True, alpha=0.3)
	ax.set_ylim(0, 400)
	fig.tight_layout()

	out_path = Path(args.out).expanduser().resolve() if args.out else json_path.with_suffix(".png")
	fig.savefig(out_path)
	print(f"Wrote {out_path}")

	if args.show:
		plt.show()


if __name__ == "__main__":
	main()


