import unittest
from tempfile import TemporaryDirectory

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")

from UniGeneX.mapping import (
    AtlasMapper,
    align_atlas_to_mapping,
    load_mapping_result,
    project_query_to_atlas,
    validate_mapping_result,
)
from UniGeneX.plotting import plot_atlas_projection


def make_mapping():
    mapping = ad.AnnData(
        X=sparse.csr_matrix([[1.0, 1.0, 2.0], [0.0, 3.0, 1.0]]),
        obs=pd.DataFrame(index=["q1", "q2"]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    mapping.obsm["atlas_neighbor_indices"] = np.array([[0, 1], [1, 2]])
    mapping.obsm["atlas_neighbor_distances"] = np.array([[0.1, 0.2], [0.2, 0.3]])
    mapping.uns["mapping_parameters"] = {"n_neighbors": 2}
    return mapping


class AtlasMappingTests(unittest.TestCase):
    def test_validate_and_align(self):
        mapping = make_mapping()
        validate_mapping_result(
            mapping, expected_neighbors=2, require_unique_query=True
        )
        atlas = ad.AnnData(
            X=np.zeros((4, 1)),
            obs=pd.DataFrame(index=["c", "a", "unused", "b"]),
        )
        aligned = align_atlas_to_mapping(atlas, mapping)
        self.assertListEqual(aligned.obs_names.tolist(), ["a", "b", "c"])

    def test_projection_normalization(self):
        mapping = make_mapping()
        coordinates = pd.DataFrame(
            [[0.0, 4.0], [0.0, 0.0], [2.0, 0.0]],
            index=["c", "a", "b"],
        )
        atlas_mask = pd.Series([True, True, False], index=["a", "b", "c"])

        projected_subset, mass = project_query_to_atlas(
            mapping, coordinates, atlas_mask=atlas_mask, normalize_by="subset"
        )
        projected_all, _ = project_query_to_atlas(
            mapping, coordinates, atlas_mask=atlas_mask, normalize_by="all"
        )

        np.testing.assert_allclose(mass, [2.0, 3.0])
        np.testing.assert_allclose(projected_subset, [[1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_allclose(projected_all, [[0.5, 0.0], [1.5, 0.0]])

    def test_projection_infers_named_atlas_subset(self):
        mapping = make_mapping()
        selected_values = pd.Series([10.0, 20.0], index=["a", "c"])

        projected, mass = project_query_to_atlas(
            mapping, selected_values, normalize_by="subset"
        )

        np.testing.assert_allclose(mass, [3.0, 1.0])
        np.testing.assert_allclose(projected, [50.0 / 3.0, 20.0])

    def test_save_and_load_mapping_result(self):
        mapping = make_mapping()
        with TemporaryDirectory() as directory:
            path = f"{directory}/mapping.h5ad"
            mapping.write_h5ad(path)
            loaded = load_mapping_result(path, expected_neighbors=2)
        self.assertListEqual(loaded.obs_names.tolist(), ["q1", "q2"])
        np.testing.assert_allclose(loaded.X.toarray(), mapping.X.toarray())

    def test_small_mapping_result(self):
        atlas = ad.AnnData(
            X=np.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.1, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.1, 0.0, 1.0],
            ]),
            obs=pd.DataFrame(
                {"cell_type": ["A", "A", "B", "B", "C", "C"]},
                index=[f"a{i}" for i in range(6)],
            ),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        atlas.obsm["X_pca"] = atlas.X[:, :2].copy()
        atlas.varm["PCs"] = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        query = ad.AnnData(
            X=np.array([[0.05, 0.0, 0.0], [1.05, 1.0, 0.0]]),
            obs=pd.DataFrame(index=["q1", "q2"]),
            var=atlas.var.copy(),
        )

        mapper = AtlasMapper(atlas, "cell_type", n_neighbors=2).fit()
        mapping = mapper.map(query)

        self.assertEqual(mapping.shape, (2, 6))
        self.assertTrue(mapper.is_fitted)
        validate_mapping_result(mapping, expected_neighbors=2)
        self.assertIn("predict_mapped_cell_type", mapping.obs)

        with TemporaryDirectory() as directory:
            mapper.save_index(directory)
            restored = AtlasMapper(atlas, "cell_type", n_neighbors=2)
            restored.load_index(directory)
            restored_mapping = restored.map(query)
        np.testing.assert_array_equal(
            mapping.obsm["atlas_neighbor_indices"],
            restored_mapping.obsm["atlas_neighbor_indices"],
        )
        np.testing.assert_allclose(mapping.X.toarray(), restored_mapping.X.toarray())

    def test_projection_plot(self):
        mapping = make_mapping()
        coordinates = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
        figure, axis = plot_atlas_projection(
            mapping,
            coordinates,
            np.array(["first", "first", "second"]),
            np.array(["red", "blue"]),
            groups=["first", "second"],
            connectivity_threshold=0.0,
            display_quantile=0.0,
        )
        self.assertIsNotNone(axis)
        self.assertGreaterEqual(len(axis.collections), 2)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
