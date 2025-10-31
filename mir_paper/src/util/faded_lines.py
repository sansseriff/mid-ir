import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection


def draw_faded_hline(
    ax,
    y_val,
    x0,
    x1,
    *,
    n_segments=120,
    base_color="0.55",
    alpha_max=0.55,
    lw=0.6,
    fade_from="left",
    zorder=0,
):
    """Draw a horizontal line at y_val from x0 to x1 using n_segments with alpha gradient.
    fade_from: 'left' -> high alpha at left, fade to right; 'right' -> high alpha at right.
    """
    xs = np.linspace(x0, x1, n_segments + 1)
    segments = [((xs[i], y_val), (xs[i + 1], y_val)) for i in range(n_segments)]
    rgba = np.array([mcolors.to_rgba(base_color)] * n_segments)
    alphas = (
        np.linspace(alpha_max, 0.0, n_segments)
        if fade_from == "left"
        else np.linspace(0.0, alpha_max, n_segments)
    )
    rgba[:, 3] = 2.0 * alphas ** (1.5)
    lc = LineCollection(segments, colors=rgba, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)
