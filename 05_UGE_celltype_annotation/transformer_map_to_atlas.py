#!/usr/bin/env python3
"""Map a query UGE dataset to a reference atlas."""

import argparse
import logging
import sys
from pathlib import Path

import scanpy as sc


VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from UniGeneX.mapping import AtlasMapper


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Map query cells to a reference atlas using cosine nearest "
            "neighbors in the atlas PCA space."
        )
    )
    parser.add_argument("--atlas-path", "--atlas_path", required=True)
    parser.add_argument("--query-path", "--adata_inte_path", required=True)
    parser.add_argument("--atlas-index-path", "--fitted_NN_path", required=True)
    parser.add_argument("--output-path", "--save_nn_path", required=True)
    parser.add_argument(
        "--atlas-label-column",
        "--atlas_assign_label_col",
        required=True,
        help="Atlas obs column transferred to query cells.",
    )
    parser.add_argument(
        "--n-neighbors",
        "--n_neighbors",
        type=int,
        default=30,
        help="Number of cosine nearest neighbors (default: 30).",
    )
    parser.add_argument(
        "--recompute-pca",
        "--recompute_pca",
        action="store_true",
        help="Recompute atlas PCA before fitting the neighbor index.",
    )
    parser.add_argument(
        "--use-existing-index",
        "--use_existing_nn",
        action="store_true",
        help="Load the fitted atlas PCA and neighbor index from atlas-index-path.",
    )
    parser.add_argument(
        "--keep-unlabeled-atlas",
        "--keep_unlabeled_atlas",
        action="store_true",
        help="Keep atlas cells with missing values in atlas-label-column.",
    )
    args = parser.parse_args()
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2.")
    if args.recompute_pca and args.use_existing_index:
        parser.error("--recompute-pca and --use-existing-index cannot be combined.")
    return args


def configure_logger(output_path: Path):
    logger = logging.getLogger("unigenex.atlas_mapping")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_path / "transformer_map_to_atlas_log.log"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    index_path = Path(args.atlas_index_path)
    output_path.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_path)

    logger.info("Loading atlas: %s", args.atlas_path)
    atlas = sc.read_h5ad(args.atlas_path)
    logger.info("Atlas shape: %s", atlas.shape)
    logger.info("Loading query: %s", args.query_path)
    query = sc.read_h5ad(args.query_path)
    logger.info("Query shape: %s", query.shape)

    mapper = AtlasMapper(
        atlas,
        args.atlas_label_column,
        n_neighbors=args.n_neighbors,
        keep_unlabeled_atlas=args.keep_unlabeled_atlas,
    )
    if args.use_existing_index:
        logger.info("Loading existing atlas neighbor index: %s", index_path)
        mapper.load_index(index_path)
    else:
        logger.info("Fitting atlas neighbor index")
        mapper.fit(recompute_pca=args.recompute_pca)
        logger.info("Saving atlas neighbor index: %s", index_path)
        mapper.save_index(index_path)

    logger.info("Mapping query cells with %d neighbors", args.n_neighbors)
    mapping = mapper.map(query)

    missing_predictions = int(
        mapping.obs[f"predict_mapped_{args.atlas_label_column}"].isna().sum()
    )
    if missing_predictions:
        logger.warning("%d query cells have no labeled atlas neighbor", missing_predictions)

    result_path = output_path / "query_to_atlas_mapping.h5ad"
    logger.info("Saving compact mapping result: %s", result_path)
    mapping.write_h5ad(result_path)
    logger.info("Mapping completed")


if __name__ == "__main__":
    main()
