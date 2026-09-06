"""Plotting helpers for query-to-atlas projections."""

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mapping import (
    _align_atlas_values,
    _prepare_connectivities,
    _project_connectivities,
)


def _aligned_vector(values, names: pd.Index, description: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        missing = names.difference(values.index)
        if len(missing):
            raise ValueError(f"{description} is missing {len(missing)} cells.")
        array = values.loc[names].to_numpy()
    else:
        array = np.asarray(values)
    if array.shape != (len(names),):
        raise ValueError(f"{description} must contain one value per cell.")
    return array


def plot_atlas_projection(
    mapping,
    atlas_coordinates,
    atlas_groups,
    query_colors,
    *,
    groups: Optional[Sequence] = None,
    query_mask=None,
    background_coordinates=None,
    connectivity_threshold: float = 0.5,
    display_quantile: float = 0.3,
    min_points: int = 1,
    atlas_color: str = "lightgray",
    atlas_size: float = 1.0,
    query_size: float = 0.05,
    query_alpha: float = 1.0,
    figsize: Tuple[float, float] = (13, 12),
    ax=None,
):
    """Plot query cells at connectivity-weighted atlas coordinates."""
    if not 0 <= display_quantile <= 1:
        raise ValueError("display_quantile must be between 0 and 1.")
    if min_points < 1:
        raise ValueError("min_points must be at least 1.")

    atlas_groups = _aligned_vector(atlas_groups, mapping.var_names, "atlas_groups")
    query_colors = _aligned_vector(query_colors, mapping.obs_names, "query_colors")
    if query_mask is None:
        query_mask = np.ones(mapping.n_obs, dtype=bool)
    else:
        query_mask = _aligned_vector(query_mask, mapping.obs_names, "query_mask").astype(bool)

    atlas_coordinates, _ = _align_atlas_values(
        atlas_coordinates, mapping.var_names
    )
    if background_coordinates is None:
        background_coordinates = atlas_coordinates
    else:
        background_coordinates = np.asarray(background_coordinates)

    if groups is None:
        groups = pd.Series(atlas_groups).value_counts().index
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.scatter(
        background_coordinates[:, 0],
        background_coordinates[:, 1],
        s=atlas_size,
        color=atlas_color,
    )

    connectivities = _prepare_connectivities(mapping, connectivity_threshold)
    for group in groups:
        coordinates, mass = _project_connectivities(
            connectivities,
            atlas_coordinates,
            atlas_groups == group,
            "subset",
        )
        positive = mass > 0
        if not np.any(positive):
            continue
        selected = (mass > np.quantile(mass[positive], display_quantile)) & query_mask
        if selected.sum() < min_points:
            continue
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=query_size,
            color=query_colors[selected],
            alpha=query_alpha,
        )
    return fig, ax
