from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cafa_graph_dataset as graphs


def make_training_row(
    entry_id: str,
    taxonomy_id: str = "9606",
    sequence: str = "ACDE",
    go_terms_bpo: list[str] | None = None,
    go_terms_cco: list[str] | None = None,
    go_terms_mfo: list[str] | None = None,
) -> dict:
    return {
        "entry_id": entry_id,
        "taxonomy_id": taxonomy_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "go_terms_bpo": go_terms_bpo or [],
        "go_terms_cco": go_terms_cco or [],
        "go_terms_mfo": go_terms_mfo or [],
        "af_status": "ok",
    }


class CafaGraphDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def test_build_protein_graph_record_deduplicates_overlapping_residues_and_edges(self) -> None:
        training_row = make_training_row("P12345", sequence="ACDE", go_terms_bpo=["GO:1"])
        fragment_rows = [
            {"entry_id": "P12345", "model_entity_id": "AF-P12345-F1"},
            {"entry_id": "P12345", "model_entity_id": "AF-P12345-F2"},
        ]
        residue_rows = [
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F1",
                "cafa_residue_index": 1,
                "residue_index": 1,
                "residue_number": 1,
                "residue_name1": "A",
                "plddt": 70.0,
                "is_plddt_very_low": False,
                "is_plddt_low": False,
                "is_plddt_confident": True,
                "is_plddt_very_high": False,
                "pae_row_mean": 2.0,
                "pae_row_min": 0.0,
                "pae_row_p90": 5.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "sequence_position_fraction": 0.25,
            },
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F1",
                "cafa_residue_index": 2,
                "residue_index": 2,
                "residue_number": 2,
                "residue_name1": "C",
                "plddt": 60.0,
                "is_plddt_very_low": False,
                "is_plddt_low": True,
                "is_plddt_confident": False,
                "is_plddt_very_high": False,
                "pae_row_mean": 9.0,
                "pae_row_min": 1.0,
                "pae_row_p90": 12.0,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "sequence_position_fraction": 0.5,
            },
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F2",
                "cafa_residue_index": 2,
                "residue_index": 1,
                "residue_number": 1,
                "residue_name1": "C",
                "plddt": 95.0,
                "is_plddt_very_low": False,
                "is_plddt_low": False,
                "is_plddt_confident": False,
                "is_plddt_very_high": True,
                "pae_row_mean": 3.0,
                "pae_row_min": 0.0,
                "pae_row_p90": 4.0,
                "x": 1.5,
                "y": 0.0,
                "z": 0.0,
                "sequence_position_fraction": 0.5,
            },
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F2",
                "cafa_residue_index": 3,
                "residue_index": 2,
                "residue_number": 2,
                "residue_name1": "D",
                "plddt": 88.0,
                "is_plddt_very_low": False,
                "is_plddt_low": False,
                "is_plddt_confident": True,
                "is_plddt_very_high": False,
                "pae_row_mean": 5.0,
                "pae_row_min": 0.0,
                "pae_row_p90": 6.0,
                "x": 2.5,
                "y": 0.0,
                "z": 0.0,
                "sequence_position_fraction": 0.75,
            },
        ]
        edge_rows = [
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F1",
                "source_residue_index": 1,
                "target_residue_index": 2,
                "source_cafa_residue_index": 1,
                "target_cafa_residue_index": 2,
                "distance_ca": 5.0,
                "seq_separation": 1,
                "pae_mean_pair": 9.0,
                "is_sequential_neighbor": True,
                "is_short_range_sequence": True,
                "is_strict_contact": True,
            },
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F2",
                "source_residue_index": 1,
                "target_residue_index": 2,
                "source_cafa_residue_index": 2,
                "target_cafa_residue_index": 3,
                "distance_ca": 4.0,
                "seq_separation": 1,
                "pae_mean_pair": 3.0,
                "is_sequential_neighbor": True,
                "is_short_range_sequence": True,
                "is_strict_contact": True,
            },
        ]

        graph_record = graphs.build_protein_graph_record(
            training_row=training_row,
            fragment_rows=fragment_rows,
            residue_rows=residue_rows,
            edge_rows=edge_rows,
        )

        self.assertEqual(graph_record["cafa_residue_index"], [1, 2, 3])
        self.assertEqual(graph_record["selected_model_entity_id_by_residue"][1], "AF-P12345-F2")
        self.assertEqual(len(graph_record["x"]), 3)
        self.assertEqual(len(graph_record["edge_attr"]), 2)
        self.assertEqual(graph_record["edge_index"], [[1, 2], [2, 1]])
        self.assertEqual(len(graph_record["graph_feat"]), graphs.GRAPH_FEAT_DIM)
        self.assertEqual(graph_record["labels"]["BPO"], ["GO:1"])

    def test_build_vocab_is_stable_and_frequency_filtered(self) -> None:
        counts = {"GO:0002": 3, "GO:0001": 1, "GO:0003": 2}

        vocab = graphs.build_vocab(counts, min_term_frequency=2)

        self.assertEqual(vocab, ["GO:0002", "GO:0003"])

    def test_make_base_feature_has_expected_width(self) -> None:
        feature = graphs.make_base_feature(
            residue_row={
                "residue_name1": "A",
                "plddt": 80.0,
                "is_plddt_very_low": False,
                "is_plddt_low": False,
                "is_plddt_confident": True,
                "is_plddt_very_high": False,
                "pae_row_mean": 5.0,
                "pae_row_min": 1.0,
                "pae_row_p90": 6.0,
                "cafa_residue_index": 2,
                "sequence_position_fraction": 0.5,
            },
            sequence_length=4,
            contact_degree=3,
            strict_contact_degree=2,
        )

        self.assertEqual(len(feature), graphs.NODE_FEATURE_DIM)
        self.assertEqual(sum(feature[:21]), 1.0)
        self.assertEqual(feature[21], 80.0)
        self.assertEqual(feature[24], 1.0)
        self.assertEqual(feature[29], 3.0)
        self.assertEqual(feature[30], 2.0)
        self.assertEqual(feature[31], 0.5)


@unittest.skipIf(graphs.torch is None, "graph runtime is only available in the Python 3.11 env")
class CafaGraphRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def _write_graph_cache(self) -> Path:
        root = self.tmpdir / "graph_cache"
        metadata_dir = root / "metadata"
        graphs_dir = root / "graphs"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        payload = graphs.tensorize_graph_record(
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "fragment_ids": ["AF-P12345-F1"],
                "cafa_residue_index": [1, 2],
                "residue_index": [1, 2],
                "fragment_id": [0, 0],
                "x": [[0.0] * graphs.NODE_FEATURE_DIM, [0.0] * graphs.NODE_FEATURE_DIM],
                "pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "edge_index": [[0, 1], [1, 0]],
                "edge_attr": [[4.0, 1.0, 2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 2.0, 1.0, 1.0, 1.0]],
                "graph_feat": [0.0] * graphs.GRAPH_FEAT_DIM,
                "node_modality_mask": [[0, 0, 0], [0, 0, 0]],
                "labels": {"BPO": ["GO:0001"], "CCO": [], "MFO": []},
            }
        )
        graphs.torch.save(payload, graphs_dir / "P12345.pt")

        with (metadata_dir / "entries.json").open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "entry_id": "P12345",
                        "taxonomy_id": "9606",
                        "graph_path": str((graphs_dir / "P12345.pt").resolve()),
                        "fragment_count": 1,
                        "residue_count": 2,
                        "labels": {"BPO": ["GO:0001"], "CCO": [], "MFO": []},
                    }
                ],
                handle,
                indent=2,
            )
        with (metadata_dir / "term_counts.json").open("w", encoding="utf-8") as handle:
            json.dump({"BPO": {"GO:0001": 1}, "CCO": {}, "MFO": {}}, handle, indent=2)
        with (metadata_dir / "schema.json").open("w", encoding="utf-8") as handle:
            json.dump({}, handle)
        return root

    @unittest.skipIf(graphs.PygData is None, "torch_geometric not installed")
    def test_pyg_dataset_loads_cached_graph(self) -> None:
        root = self._write_graph_cache()

        dataset = graphs.CafaPyGDataset(root=root, aspect="BPO")
        sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample.x.shape[1], graphs.NODE_FEATURE_DIM)
        self.assertEqual(sample.edge_attr.shape[1], graphs.EDGE_ATTR_DIM)
        self.assertEqual(sample.graph_feat.shape[0], graphs.GRAPH_FEAT_DIM)
        self.assertEqual(int(sample.node_modality_mask.sum().item()), 0)

    @unittest.skipIf(graphs.dgl is None, "dgl not installed")
    def test_dgl_dataset_loads_cached_graph(self) -> None:
        root = self._write_graph_cache()

        dataset = graphs.CafaDGLDataset(root=root, aspect="BPO")
        graph = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(graph.num_nodes(), 2)
        self.assertEqual(graph.ndata["x"].shape[1], graphs.NODE_FEATURE_DIM)
        self.assertEqual(graph.edata["edge_attr"].shape[1], graphs.EDGE_ATTR_DIM)
        self.assertEqual(graph.graph_feat.shape[0], graphs.GRAPH_FEAT_DIM)
        self.assertEqual(int(graph.ndata["node_modality_mask"].sum().item()), 0)

    @unittest.skipIf(graphs.PygData is None, "torch_geometric not installed")
    def test_pyg_dataset_overlays_esm2_cache(self) -> None:
        root = self._write_graph_cache()
        cache_dir = root / "modality_cache" / "esm2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        graphs.torch.save(
            {
                "entry_id": "P12345",
                "cafa_residue_index": graphs.torch.tensor([1, 2], dtype=graphs.torch.long),
                "residue_embedding": graphs.torch.stack(
                    [
                        graphs.torch.full((graphs.ESM2_DIM,), 1.5, dtype=graphs.torch.float32),
                        graphs.torch.full((graphs.ESM2_DIM,), 2.5, dtype=graphs.torch.float32),
                    ]
                ),
                "protein_embedding": graphs.torch.zeros(graphs.ESM2_DIM, dtype=graphs.torch.float32),
            },
            cache_dir / "P12345.pt",
        )

        dataset = graphs.CafaPyGDataset(root=root, aspect="BPO", use_esm2=True)
        sample = dataset[0]

        self.assertTrue(bool(sample.node_modality_mask[:, 2].all().item()))
        self.assertEqual(float(sample.x[0, graphs.ESM2_SLICE.start].item()), 1.5)
        self.assertEqual(float(sample.x[1, graphs.ESM2_SLICE.start].item()), 2.5)

    @unittest.skipIf(graphs.PygData is None, "torch_geometric not installed")
    def test_pyg_dataset_overlays_structure_cache(self) -> None:
        root = self._write_graph_cache()
        cache_dir = root / "modality_cache" / "structure"
        cache_dir.mkdir(parents=True, exist_ok=True)
        features = graphs.torch.tensor(
            [
                [1.0] * graphs.DSSP_SASA_DIM,
                [2.0] * graphs.DSSP_SASA_DIM,
            ],
            dtype=graphs.torch.float32,
        )
        graphs.torch.save(
            {
                "entry_id": "P12345",
                "model_entity_id": "AF-P12345-F1",
                "residue_index": graphs.torch.tensor([1, 2], dtype=graphs.torch.long),
                "features": features,
                "dssp_mask": graphs.torch.tensor([True, False], dtype=graphs.torch.bool),
                "sasa_mask": graphs.torch.tensor([False, True], dtype=graphs.torch.bool),
            },
            cache_dir / "AF-P12345-F1.pt",
        )

        dataset = graphs.CafaPyGDataset(root=root, aspect="BPO", use_dssp=True, use_sasa=True)
        sample = dataset[0]

        self.assertEqual(float(sample.x[0, graphs.DSSP_SASA_SLICE.start].item()), 1.0)
        self.assertEqual(float(sample.x[1, graphs.DSSP_SASA_SLICE.start].item()), 2.0)
        self.assertEqual(sample.node_modality_mask[:, 0].tolist(), [True, False])
        self.assertEqual(sample.node_modality_mask[:, 1].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
