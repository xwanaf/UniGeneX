"""Spatial color-mixture plot used for the GBM Visium manuscript panels."""

from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


def _value_to_color(cmap, minimum: float, maximum: float):
    if minimum > maximum:
        raise ValueError("maximum must be greater than or equal to minimum")
    if minimum == maximum:
        warnings.warn("A spatial program has no variation.", stacklevel=2)

        def constant(values):
            level = 0 if maximum == 0 else 0.5
            return cmap(np.full_like(values, level))

        return constant

    def scale(values):
        values = np.clip(values, minimum, maximum)
        return cmap((values - minimum) / (maximum - minimum))

    return scale


def _rgb_to_ryb(rgb: np.ndarray) -> np.ndarray:
    """Convert an array of RGB colors to the artist-oriented RYB space."""
    rgb = np.atleast_2d(np.asarray(rgb)).copy()
    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb -= white[:, None]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    nonzero = ~(ryb == 0).all(axis=1)
    if nonzero.any():
        scale = ryb[nonzero].max(axis=1) / rgb[nonzero].max(axis=1)
        ryb[nonzero] /= scale[:, None]
    return ryb + black[:, None]


def _ryb_to_rgb(ryb: np.ndarray) -> np.ndarray:
    """Convert an array of RYB colors back to RGB."""
    ryb = np.atleast_2d(np.asarray(ryb)).copy()
    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb -= black[:, None]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    nonzero = ~(ryb == 0).all(axis=1)
    if nonzero.any():
        scale = rgb[nonzero].max(axis=1) / ryb[nonzero].max(axis=1)
        rgb[nonzero] /= scale[:, None]
    return rgb + white[:, None]


def _alpha_colormap(hex_color: str, white_spacing: int = 20) -> ListedColormap:
    """Create the constant-color, increasing-alpha map used in the manuscript."""
    rgb = (np.asarray(mcolors.hex2color(hex_color)) * 255).astype(int) / 255
    spacing = int(white_spacing * 2.55)
    levels = 255 * 3
    alpha = np.concatenate(
        [np.zeros(spacing * 3), np.linspace(0, 1, levels - spacing * 3)]
    )
    colors = np.ones((levels, 4))
    colors[:, :3] = rgb
    colors[:, 3] = alpha
    return ListedColormap(colors)


def plot_spatial(
    adata,
    color: list[str],
    labels: list[str],
    custom_colors_hex: list[str],
    *,
    img_key: str = "hires",
    max_color_quantile: float = 0.98,
    circle_diameter: float = 4,
    crop_x: tuple[float, float] | None = None,
    crop_y: tuple[float, float] | None = None,
    axis_y_flipped: bool = False,
):
    """Blend several spatial programs into one tissue map.

    Each program is independently clipped at ``max_color_quantile``. Program
    colors are mixed in RYB space using squared relative abundances, matching
    the plotting procedure used for the manuscript.
    """
    if not (len(color) == len(labels) == len(custom_colors_hex)):
        raise ValueError("color, labels, and custom_colors_hex must have equal length")

    spatial = next(iter(adata.uns["spatial"].values()))
    image = spatial["images"][img_key]
    scale_factor = spatial["scalefactors"][f"tissue_{img_key}_scalef"]
    coords = np.asarray(adata.obsm["spatial"]) * scale_factor
    counts = adata.obs[color].to_numpy(copy=True)
    colormaps = [_alpha_colormap(value) for value in custom_colors_hex]

    with mpl.style.context("fast"):
        figure = plt.figure()
        grid = GridSpec(
            nrows=len(labels) + 2,
            ncols=2,
            width_ratios=[1, 0.15],
            height_ratios=[1, *([0.2] * len(labels)), 1],
            hspace=1.5,
            wspace=0.05,
        )
        axis = figure.add_subplot(grid[:, 0], aspect="equal", rasterized=True)
        colorbar_axes = [figure.add_subplot(grid[row, 1]) for row in range(1, len(labels) + 1)]

        axis.imshow(image, aspect="equal", alpha=1, origin="lower", cmap="Greys_r")
        if crop_x is not None:
            axis.set_xlim(*crop_x)
        if crop_y is not None:
            axis.set_ylim(*crop_y)
        if axis_y_flipped:
            axis.invert_yaxis()
        axis.invert_xaxis()
        axis.set_axis_off()

        rgba = np.zeros((*counts.shape, 4))
        weights = np.zeros_like(counts)
        for index, (label, cmap, colorbar_axis) in enumerate(
            zip(labels, colormaps, colorbar_axes)
        ):
            minimum = counts[:, index].min()
            maximum = np.quantile(counts[:, index], max_color_quantile)
            color_scale = _value_to_color(cmap, minimum, maximum)
            rgba[:, index] = color_scale(counts[:, index])
            weights[:, index] = np.clip(
                counts[:, index] / (maximum + 1e-10), 0, 1
            )

            colorbar = figure.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(vmin=minimum, vmax=maximum),
                    cmap=cmap,
                ),
                cax=colorbar_axis,
                orientation="horizontal",
                extend="both",
                ticks=[],
            )
            colorbar.outline.set_color("black")
            colorbar.outline.set_linewidth(2)
            colorbar.ax.grid(False)
            title_color = color_scale(maximum / 1.5)
            colorbar.ax.set_title(label, size=20, color=title_color, alpha=1)

        colors_ryb = np.stack(
            [_rgb_to_ryb(spot_colors[:, :3]) for spot_colors in rgba]
        )
        squared_weights = weights[:, :, None] ** 2
        denominator = squared_weights.sum(axis=1)
        denominator[denominator == 0] = 1
        mixed_ryb = (colors_ryb * squared_weights).sum(axis=1) / denominator

        mixed_colors = np.zeros((counts.shape[0], 4))
        mixed_colors[:, :3] = _ryb_to_rgb(mixed_ryb)
        mixed_colors[:, 3] = rgba[:, :, 3].max(axis=1)
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            c=mixed_colors,
            s=circle_diameter**2,
        )

    return figure
