#!/usr/bin/env python3
"""Fit Cell2location references and deconvolve spatial transcriptomics data."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy import sparse

from cell2location.models import Cell2location, RegressionModel


def _require_cuda(task: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"A CUDA GPU is required to {task}. "
            "Use precomputed results on CPU-only systems."
        )


def fit_cell_state_signatures(
    reference: ad.AnnData,
    labels_key: str,
    output_path: Path | None = None,
    *,
    max_epochs: int = 250,
    batch_size: int = 2500,
    learning_rate: float = 0.002,
    posterior_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Learn a gene-by-cell-state signature table from a prepared reference."""
    _require_cuda("fit cell-state signatures")
    if labels_key not in reference.obs:
        raise KeyError(f"Reference labels not found in .obs: {labels_key}")

    scvi.settings.seed = seed
    RegressionModel.setup_anndata(reference, labels_key=labels_key)
    model = RegressionModel(reference)
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_size=1,
        lr=learning_rate,
        accelerator="gpu",
    )
    reference = model.export_posterior(
        reference,
        sample_kwargs={
            "num_samples": posterior_samples,
            "batch_size": batch_size,
            "accelerator": "gpu",
        },
    )

    state_names = list(reference.uns["mod"]["factor_names"])
    columns = [f"means_per_cluster_mu_fg_{state}" for state in state_names]
    if "means_per_cluster_mu_fg" in reference.varm:
        signatures = reference.varm["means_per_cluster_mu_fg"][columns].copy()
    else:
        signatures = reference.var[columns].copy()
    signatures.columns = state_names

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        signatures.to_csv(output_path)
        print(f"Saved cell-state signatures: {output_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return signatures


def prepare_spatial(
    spatial_path: Path,
    signatures: pd.DataFrame,
    minimum_counts: int = 1000,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Match genes and retain adequately sequenced, in-tissue locations."""
    spatial = sc.read_h5ad(spatial_path)
    spatial.var_names_make_unique()

    shared_genes = np.intersect1d(spatial.var_names, signatures.index)
    if shared_genes.size == 0:
        raise ValueError("The spatial data and signature table share no genes.")

    spatial = spatial[:, shared_genes].copy()
    signatures = signatures.loc[shared_genes].copy()

    if "in_tissue" in spatial.obs:
        spatial = spatial[spatial.obs["in_tissue"] == 1].copy()

    spatial.X = sparse.csr_matrix(spatial.X.astype(int))
    total_counts = np.asarray(spatial.X.sum(axis=1)).ravel()
    spatial.obs["total_counts"] = total_counts
    spatial = spatial[total_counts >= minimum_counts].copy()
    if spatial.n_obs == 0:
        raise ValueError(
            f"No spatial locations contain at least {minimum_counts:,} matched-gene counts."
        )

    print(
        f"Spatial input: {spatial.n_obs:,} locations x "
        f"{spatial.n_vars:,} shared genes",
        flush=True,
    )
    return spatial, signatures


def deconvolve_spatial(
    spatial_path: Path,
    signatures: pd.DataFrame,
    output_path: Path | None = None,
    *,
    minimum_counts: int = 1000,
    cells_per_location: float = 20,
    detection_alpha: float = 200,
    use_bg: bool = False,
    max_epochs: int = 30000,
    posterior_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit Cell2location and return q05 cell-state abundance per location."""
    _require_cuda("deconvolve spatial data")
    scvi.settings.seed = seed
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    spatial, signatures = prepare_spatial(
        spatial_path,
        signatures,
        minimum_counts,
    )
    Cell2location.setup_anndata(spatial)
    model = Cell2location(
        spatial,
        cell_state_df=signatures,
        N_cells_per_location=cells_per_location,
        nuclei_segmentation=False,
        n_s_cells_per_location=np.ones(spatial.n_obs),
        detection_alpha=detection_alpha,
        use_bg=use_bg,
    )
    model.train(
        max_epochs=max_epochs,
        batch_size=None,
        train_size=1,
        accelerator="gpu",
    )
    spatial = model.export_posterior(
        spatial,
        sample_kwargs={
            "num_samples": posterior_samples,
            "batch_size": spatial.n_obs,
            "accelerator": "gpu",
        },
    )

    abundance = spatial.obsm["q05_cell_abundance_w_sf"].copy()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        abundance.to_csv(output_path)
        print(f"Saved spatial decomposition: {output_path}", flush=True)

    del model, spatial
    gc.collect()
    torch.cuda.empty_cache()
    return abundance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional prepared count-based single-cell reference .h5ad.",
    )
    parser.add_argument(
        "--labels-key",
        help="Reference .obs column containing cell-state labels.",
    )
    parser.add_argument(
        "--signatures",
        required=True,
        type=Path,
        help="Gene-by-cell-state signature CSV to create or load.",
    )
    parser.add_argument(
        "--spatial",
        nargs="+",
        type=Path,
        help="One or more prepared spatial expression .h5ad files.",
    )
    parser.add_argument(
        "--output",
        nargs="+",
        type=Path,
        help="One output CSV for each spatial input.",
    )
    parser.add_argument(
        "--use-bg",
        action="store_true",
        help="Enable a gene-specific background profile.",
    )

    args = parser.parse_args()
    if args.reference is not None and args.labels_key is None:
        parser.error("--labels-key is required with --reference.")
    if (args.spatial is None) != (args.output is None):
        parser.error("--spatial and --output must be provided together.")
    if args.spatial is not None and len(args.spatial) != len(args.output):
        parser.error("--spatial and --output must contain the same number of paths.")
    if args.reference is None and args.spatial is None:
        parser.error("Provide --reference, --spatial, or both.")
    return args


def main() -> None:
    args = parse_args()
    signature_path = args.signatures.expanduser().resolve()
    signatures = None

    if args.reference is not None:
        reference_path = args.reference.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference data not found: {reference_path}")

        reference = ad.read_h5ad(reference_path)
        print(
            f"Reference input: {reference.n_obs:,} cells x "
            f"{reference.n_vars:,} genes",
            flush=True,
        )
        signatures = fit_cell_state_signatures(
            reference,
            labels_key=args.labels_key,
            output_path=signature_path,
        )
        del reference
        gc.collect()

    if args.spatial is not None:
        if signatures is None:
            if not signature_path.is_file():
                raise FileNotFoundError(
                    f"Signature table not found: {signature_path}"
                )
            signatures = pd.read_csv(signature_path, index_col=0)
            print(f"Signature input: {signature_path}", flush=True)

        for spatial_path, output_path in zip(args.spatial, args.output):
            deconvolve_spatial(
                spatial_path=spatial_path.expanduser().resolve(),
                signatures=signatures,
                output_path=output_path.expanduser().resolve(),
                use_bg=args.use_bg,
            )


if __name__ == "__main__":
    main()
