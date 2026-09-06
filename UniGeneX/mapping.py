"""Atlas mapping utilities used by the UniGeneX reproducibility workflows."""

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass
class _AtlasNeighborIndex:
    """Nearest-neighbor model and arrays fitted on atlas PCA coordinates."""

    model: NearestNeighbors
    atlas_pcs: np.ndarray
    pcs: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray

    @property
    def n_neighbors(self) -> int:
        return int(self.neighbor_indices.shape[1])


def _compute_umap_connectivities(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_obs: int,
    n_neighbors: int,
):
    """Build fuzzy connectivities without relying on Scanpy's private API."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        from umap.umap_ import fuzzy_simplicial_set

    placeholder = sparse.coo_matrix((n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        placeholder,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    if isinstance(connectivities, tuple):
        connectivities = connectivities[0]
    return connectivities.tocsr()


def _build_atlas_index(
    atlas_adata: ad.AnnData,
    n_neighbors: int,
) -> _AtlasNeighborIndex:
    if "X_pca" not in atlas_adata.obsm or "PCs" not in atlas_adata.varm:
        raise ValueError("The atlas must contain obsm['X_pca'] and varm['PCs'].")
    if not 2 <= n_neighbors < atlas_adata.n_obs:
        raise ValueError("n_neighbors must be at least 2 and smaller than the atlas.")

    atlas_pcs = np.asarray(atlas_adata.obsm["X_pca"])
    pcs = np.asarray(atlas_adata.varm["PCs"])
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(atlas_pcs)
    distances, indices = model.kneighbors(return_distance=True)
    return _AtlasNeighborIndex(model, atlas_pcs, pcs, indices, distances)


def _validate_atlas_index(
    index: _AtlasNeighborIndex,
    atlas_adata: ad.AnnData,
    n_neighbors: int,
) -> None:
    if index.atlas_pcs.shape[0] != atlas_adata.n_obs:
        raise ValueError("The neighbor index and selected atlas have different cell counts.")
    if index.pcs.shape[0] != atlas_adata.n_vars:
        raise ValueError("The neighbor index and selected atlas have different gene counts.")
    if index.neighbor_indices.shape != index.neighbor_distances.shape:
        raise ValueError("Atlas neighbor index and distance arrays have different shapes.")
    if index.neighbor_indices.shape != (atlas_adata.n_obs, n_neighbors):
        raise ValueError("The cached atlas neighbors do not match n_neighbors.")
    if index.model.n_neighbors != n_neighbors:
        raise ValueError("The fitted neighbor model does not match n_neighbors.")
    if index.atlas_pcs.shape[1] != index.pcs.shape[1]:
        raise ValueError("Atlas PCA coordinates and loadings have different dimensions.")


def _project_query_pca(query_adata: ad.AnnData, atlas_adata: ad.AnnData, pcs: np.ndarray):
    atlas_mean = np.asarray(atlas_adata.X.mean(axis=0)).ravel()
    if sparse.issparse(query_adata.X):
        projected = query_adata.X @ pcs - atlas_mean @ pcs
    else:
        projected = (np.asarray(query_adata.X) - atlas_mean) @ pcs
    return np.asarray(projected)


def _transfer_majority_label(
    atlas_labels: pd.Series,
    neighbor_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_labels = atlas_labels.to_numpy()[neighbor_indices.reshape(-1)]
    votes = pd.DataFrame({
        "query_position": np.repeat(np.arange(neighbor_indices.shape[0]), neighbor_indices.shape[1]),
        "atlas_label": neighbor_labels,
    }).dropna(subset=["atlas_label"])

    n_query = neighbor_indices.shape[0]
    labels = np.full(n_query, np.nan, dtype=object)
    probabilities = np.zeros(n_query, dtype=np.float64)
    labeled_counts = np.zeros(n_query, dtype=np.int64)
    if votes.empty:
        return labels, probabilities, labeled_counts

    vote_counts = (
        votes.groupby(["query_position", "atlas_label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(np.arange(n_query), fill_value=0)
    )
    labeled_counts = vote_counts.sum(axis=1).to_numpy(dtype=np.int64)
    has_label = labeled_counts > 0
    labels[has_label] = vote_counts.loc[has_label].idxmax(axis=1).to_numpy()
    probabilities[has_label] = (
        vote_counts.loc[has_label].max(axis=1).to_numpy() / labeled_counts[has_label]
    )
    return labels, probabilities, labeled_counts


class AtlasMapper:
    """Fit an atlas neighbor index and map one or more query datasets."""

    def __init__(
        self,
        atlas_adata: ad.AnnData,
        label_column: str,
        *,
        n_neighbors: int = 30,
        keep_unlabeled_atlas: bool = False,
    ):
        if label_column not in atlas_adata.obs:
            raise KeyError(f"Atlas obs is missing label column: {label_column}")

        self.label_column = label_column
        self.n_neighbors = n_neighbors
        self.keep_unlabeled_atlas = keep_unlabeled_atlas
        self.n_atlas_input = atlas_adata.n_obs

        has_label = atlas_adata.obs[label_column].notna().to_numpy()
        self.n_unlabeled_atlas_input = int((~has_label).sum())
        if keep_unlabeled_atlas:
            self.atlas = atlas_adata
        else:
            self.atlas = atlas_adata[has_label].copy()

        if not 2 <= n_neighbors < self.atlas.n_obs:
            raise ValueError(
                "n_neighbors must be at least 2 and smaller than the selected atlas."
            )
        if not self.atlas.var_names.is_unique:
            raise ValueError("Atlas gene names are not unique.")
        self._index: Optional[_AtlasNeighborIndex] = None

    @property
    def is_fitted(self) -> bool:
        """Whether an atlas neighbor index is ready for mapping."""
        return self._index is not None

    def fit(self, *, recompute_pca: bool = False):
        """Fit the atlas neighbor index and return this mapper."""
        if recompute_pca:
            import scanpy as sc

            sc.tl.pca(self.atlas, n_comps=30, use_highly_variable=False)
        self._index = _build_atlas_index(self.atlas, self.n_neighbors)
        return self

    def load_index(self, path):
        """Load a previously fitted atlas neighbor index and return this mapper."""
        path = Path(path)
        with (path / "knnpickle_file").open("rb") as handle:
            model = pickle.load(handle)
        index = _AtlasNeighborIndex(
            model=model,
            atlas_pcs=np.load(path / "atlas_pcs.npy"),
            pcs=np.load(path / "pcs.npy"),
            neighbor_indices=np.load(path / "neigh_ind_atlas.npy"),
            neighbor_distances=np.load(path / "neigh_dist_atlas.npy"),
        )
        _validate_atlas_index(index, self.atlas, self.n_neighbors)
        self._index = index
        return self

    def save_index(self, path) -> None:
        """Save the fitted atlas neighbor index."""
        index = self._require_index()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "atlas_pcs.npy", index.atlas_pcs)
        np.save(path / "pcs.npy", index.pcs)
        np.save(path / "neigh_ind_atlas.npy", index.neighbor_indices)
        np.save(path / "neigh_dist_atlas.npy", index.neighbor_distances)
        with (path / "knnpickle_file").open("wb") as handle:
            pickle.dump(index.model, handle)

    def _require_index(self) -> _AtlasNeighborIndex:
        if self._index is None:
            raise RuntimeError("Call fit() or load_index() before map().")
        return self._index

    def _align_query_genes(self, query_adata: ad.AnnData) -> ad.AnnData:
        if not query_adata.var_names.is_unique:
            raise ValueError("Query gene names are not unique.")
        missing = self.atlas.var_names.difference(query_adata.var_names)
        if len(missing):
            raise ValueError(f"The query is missing {len(missing)} atlas genes.")
        if query_adata.var_names.equals(self.atlas.var_names):
            return query_adata
        return query_adata[:, self.atlas.var_names].copy()

    def map(self, query_adata: ad.AnnData) -> ad.AnnData:
        """Map query cells and return a compact query-to-atlas result."""
        index = self._require_index()
        query_adata = self._align_query_genes(query_adata)
        query_pcs = _project_query_pca(query_adata, self.atlas, index.pcs)
        query_distances, query_indices = index.model.kneighbors(
            query_pcs, return_distance=True
        )

        all_indices = np.vstack((index.neighbor_indices, query_indices))
        all_distances = np.vstack((index.neighbor_distances, query_distances))
        connectivities = _compute_umap_connectivities(
            all_indices,
            all_distances,
            n_obs=self.atlas.n_obs + query_adata.n_obs,
            n_neighbors=self.n_neighbors,
        )
        query_connectivities = connectivities[
            self.atlas.n_obs :, : self.atlas.n_obs
        ].copy()

        labels, probabilities, labeled_counts = _transfer_majority_label(
            self.atlas.obs[self.label_column], query_indices
        )
        prediction_column = f"predict_mapped_{self.label_column}"
        probability_column = f"{prediction_column}_prob"
        fraction_column = f"{prediction_column}_labeled_neighbor_fraction"

        query_obs = query_adata.obs.copy()
        query_obs[prediction_column] = labels
        query_obs[probability_column] = probabilities
        query_obs[fraction_column] = labeled_counts / self.n_neighbors

        mapping = ad.AnnData(
            X=query_connectivities,
            obs=query_obs,
            var=self.atlas.obs[[self.label_column]].copy(),
        )
        mapping.obsm["atlas_neighbor_indices"] = query_indices
        mapping.obsm["atlas_neighbor_distances"] = query_distances
        mapping.uns["matrix_semantics"] = (
            "X contains query-to-atlas UMAP fuzzy connectivities; rows are "
            "query observations and columns are atlas observations."
        )
        mapping.uns["mapping_parameters"] = {
            "n_neighbors": int(self.n_neighbors),
            "metric": "cosine",
            "atlas_label_column": self.label_column,
            "keep_unlabeled_atlas": bool(self.keep_unlabeled_atlas),
            "n_atlas_input": int(self.n_atlas_input),
            "n_atlas_used": int(self.atlas.n_obs),
            "n_unlabeled_atlas_used": int(
                self.atlas.obs[self.label_column].isna().sum()
            ),
            "n_unlabeled_atlas_input": self.n_unlabeled_atlas_input,
            "label_probability_denominator": (
                "atlas neighbors with non-missing labels"
            ),
        }
        return mapping


def load_mapping_result(
    path,
    *,
    expected_neighbors: Optional[int] = None,
    require_unique_query: bool = False,
    backed: Optional[Literal["r", "r+"]] = None,
) -> ad.AnnData:
    """Load and validate a compact query-to-atlas mapping result."""
    mapping = ad.read_h5ad(Path(path), backed=backed)
    try:
        validate_mapping_result(
            mapping,
            expected_neighbors=expected_neighbors,
            require_unique_query=require_unique_query,
        )
    except Exception:
        if mapping.isbacked:
            mapping.file.close()
        raise
    return mapping


def validate_mapping_result(
    mapping: ad.AnnData,
    *,
    expected_neighbors: Optional[int] = None,
    require_unique_query: bool = False,
) -> None:
    """Validate the row, column, neighbor, and parameter contract of a mapping."""
    if require_unique_query and not mapping.obs_names.is_unique:
        raise ValueError("Query observation names in the mapping are not unique.")
    if not mapping.var_names.is_unique:
        raise ValueError("Atlas observation names in the mapping are not unique.")
    if mapping.X.shape != (mapping.n_obs, mapping.n_vars):
        raise ValueError("The mapping matrix dimensions do not match obs and var.")

    required = {"atlas_neighbor_indices", "atlas_neighbor_distances"}
    missing = required.difference(mapping.obsm)
    if missing:
        raise KeyError(f"The mapping is missing obsm arrays: {sorted(missing)}")
    indices = np.asarray(mapping.obsm["atlas_neighbor_indices"])
    distances = np.asarray(mapping.obsm["atlas_neighbor_distances"])
    if indices.shape != distances.shape or indices.shape[0] != mapping.n_obs:
        raise ValueError("Neighbor arrays do not align with the mapping rows.")
    if np.any((indices < 0) | (indices >= mapping.n_vars)):
        raise ValueError("Neighbor indices fall outside the atlas-column range.")

    parameters = dict(mapping.uns.get("mapping_parameters", {}))
    recorded_neighbors = parameters.get("n_neighbors")
    if recorded_neighbors is not None and int(recorded_neighbors) != indices.shape[1]:
        raise ValueError("The recorded n_neighbors does not match the neighbor arrays.")
    if expected_neighbors is not None and indices.shape[1] != expected_neighbors:
        raise ValueError(
            f"The mapping uses {indices.shape[1]} neighbors; expected {expected_neighbors}."
        )


def align_atlas_to_mapping(atlas_adata: ad.AnnData, mapping: ad.AnnData) -> ad.AnnData:
    """Return an atlas view ordered exactly like the columns of a mapping result."""
    if not atlas_adata.obs_names.is_unique:
        raise ValueError("Atlas observation names are not unique.")
    missing = mapping.var_names.difference(atlas_adata.obs_names)
    if len(missing):
        raise ValueError(f"The atlas is missing {len(missing)} mapping columns.")
    aligned = atlas_adata[mapping.var_names]
    if not aligned.obs_names.equals(mapping.var_names):
        raise RuntimeError("Failed to align atlas observations to mapping columns.")
    return aligned


def _align_atlas_values(values, atlas_names: pd.Index) -> Tuple[np.ndarray, bool]:
    one_dimensional = isinstance(values, pd.Series) or np.asarray(values).ndim == 1
    if isinstance(values, (pd.Series, pd.DataFrame)):
        if not values.index.is_unique:
            raise ValueError("Atlas-value identifiers are not unique.")
        missing = atlas_names.difference(values.index)
        if len(missing):
            raise ValueError(f"Atlas values are missing {len(missing)} mapping columns.")
        array = values.loc[atlas_names].to_numpy()
    else:
        array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != len(atlas_names):
        raise ValueError("Atlas values must have one row per mapping column.")
    return array, one_dimensional


def _align_projection_values(values, atlas_names, mask, infer_mask):
    one_dimensional = isinstance(values, pd.Series) or np.asarray(values).ndim == 1
    if isinstance(values, (pd.Series, pd.DataFrame)):
        if not values.index.is_unique:
            raise ValueError("Atlas-value identifiers are not unique.")
        unknown = values.index.difference(atlas_names)
        if len(unknown):
            raise ValueError(f"Atlas values contain {len(unknown)} unknown cells.")
        if infer_mask:
            mask = atlas_names.isin(values.index)
        selected_names = atlas_names[mask]
        missing = selected_names.difference(values.index)
        if len(missing):
            raise ValueError(f"Atlas values are missing {len(missing)} selected cells.")
        selected_values = values.loc[selected_names].to_numpy()
    else:
        array = np.asarray(values)
        if infer_mask:
            mask = np.ones(len(atlas_names), dtype=bool)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError("Atlas values must be one- or two-dimensional.")
        if array.shape[0] == len(atlas_names):
            selected_values = array[mask]
        elif array.shape[0] == int(mask.sum()):
            selected_values = array
        else:
            raise ValueError("Atlas values do not match the selected mapping columns.")

    if selected_values.ndim == 1:
        selected_values = selected_values[:, None]
    if not np.any(mask):
        raise ValueError("No atlas cells were selected for projection.")
    full_values = np.zeros(
        (len(atlas_names), selected_values.shape[1]),
        dtype=np.result_type(selected_values.dtype, np.float64),
    )
    full_values[mask] = selected_values
    return full_values, np.asarray(mask, dtype=bool), one_dimensional


def _prepare_connectivities(mapping: ad.AnnData, threshold: Optional[float]):
    validate_mapping_result(mapping)
    matrix = mapping.X.to_memory() if hasattr(mapping.X, "to_memory") else mapping.X
    # Thresholding mutates sparse data; unthresholded projections can safely
    # reuse an in-memory CSR matrix without duplicating a large mapping result.
    connectivities = sparse.csr_matrix(matrix, copy=threshold is not None)
    if threshold is not None:
        connectivities.data[connectivities.data < threshold] = 0
        connectivities.eliminate_zeros()
    return connectivities


def _project_connectivities(
    connectivities,
    values,
    mask,
    normalize_by,
    *,
    normalize_weights=False,
):
    subset = connectivities[:, mask]
    subset_mass = np.asarray(subset.sum(axis=1)).ravel()
    if normalize_by == "all":
        denominator = np.asarray(connectivities.sum(axis=1)).ravel()
    else:
        denominator = subset_mass

    if normalize_weights:
        inverse = np.zeros_like(denominator, dtype=np.float64)
        nonzero = denominator > 0
        inverse[nonzero] = 1.0 / denominator[nonzero]
        if normalize_by == "all":
            normalized = connectivities.multiply(inverse[:, None]).tocsr()
            projected = normalized[:, mask].dot(values[mask])
        else:
            normalized = subset.multiply(inverse[:, None]).tocsr()
            projected = normalized.dot(values[mask])
        return np.asarray(projected), subset_mass

    numerator = np.asarray(subset.dot(values[mask]))
    projected = np.zeros((connectivities.shape[0], values.shape[1]), dtype=np.float64)
    nonzero = denominator > 0
    projected[nonzero] = numerator[nonzero] / denominator[nonzero, None]
    return projected, subset_mass


def project_query_to_atlas(
    mapping: ad.AnnData,
    atlas_values,
    *,
    atlas_mask=None,
    connectivity_threshold: Optional[float] = None,
    normalize_by: Literal["all", "subset"] = "subset",
) -> Tuple[np.ndarray, np.ndarray]:
    """Project atlas values to every query cell using mapping connectivities.

    Returns the projected values and each query cell's connectivity mass to the
    selected atlas subset. ``normalize_by='all'`` preserves how much weight lies
    outside the subset; ``'subset'`` computes a within-subset weighted average.
    """
    if normalize_by not in {"all", "subset"}:
        raise ValueError("normalize_by must be 'all' or 'subset'.")
    if connectivity_threshold is not None and connectivity_threshold < 0:
        raise ValueError("connectivity_threshold cannot be negative.")

    connectivities = _prepare_connectivities(mapping, connectivity_threshold)

    infer_mask = atlas_mask is None
    if infer_mask:
        mask = np.ones(mapping.n_vars, dtype=bool)
    elif isinstance(atlas_mask, pd.Series):
        missing = mapping.var_names.difference(atlas_mask.index)
        if len(missing):
            raise ValueError(f"Atlas mask is missing {len(missing)} mapping columns.")
        mask = atlas_mask.loc[mapping.var_names].to_numpy(dtype=bool)
    else:
        mask = np.asarray(atlas_mask, dtype=bool)
    if mask.shape != (mapping.n_vars,):
        raise ValueError("atlas_mask must contain one value per mapping column.")

    values, mask, one_dimensional = _align_projection_values(
        atlas_values, mapping.var_names, mask, infer_mask
    )
    projected, subset_mass = _project_connectivities(
        connectivities,
        values,
        mask,
        normalize_by,
        normalize_weights=True,
    )
    if one_dimensional:
        projected = projected[:, 0]
    return projected, subset_mass
