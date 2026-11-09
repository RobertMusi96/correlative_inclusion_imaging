"""Common matplotlib style for the figures of the project (fonts, sizes, axes, grid, legend)."""
import matplotlib as mpl
from cycler import cycler


def setup_matplotlib_parameter(color_cycle):
    """Sets the project's matplotlib style globally; color_cycle = colours used in order for
    the series of a plot."""
    mpl.rcParams.update({
        "axes.prop_cycle":  cycler(color=color_cycle),
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "sans-serif"],
        "font.size":        10,
        "axes.labelsize":   12,
        "axes.labelweight": "normal",
        "axes.titlesize":   10,
        "axes.titleweight": "normal",
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.edgecolor":   "#222222",
        "axes.linewidth":   0.8,
        "axes.facecolor":   "#ffffff",
        "figure.facecolor": "#ffffff",
        "grid.color":       "#dddddd",
        "grid.linewidth":   0.6,
        "axes.grid":        True,
        "legend.fontsize":  10,
        "legend.frameon":   False,
        "legend.labelcolor": "#444444",
    })
