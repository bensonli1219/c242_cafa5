from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs
import export_graph_prediction_bundles as graph_exporter
import train_minimal_graph_model as training


@unittest.skipIf(graphs.torch is None, "training runtime is only available in the Python 3.11 env")
class MinimalGraphTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def _write_graph_cache(self, entry_count: int = 6) -> tuple[Path, Path]:
        root = self.tmpdir / "graph_cache"
        metadata_dir = root / "metadata"
        graphs_dir = root / "graphs"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for index in range(entry_count):
            entry_id = f"P{index + 1:05d}"
            payload = graphs.tensorize_graph_record(
                {
                    "entry_id": entry_id,
                    "taxonomy_id": "9606",
                    "fragment_ids": [f"AF-{entry_id}-F1"],
                    "cafa_residue_index": [1, 2],
                    "residue_index": [1, 2],
                    "fragment_id": [0, 0],
                    "x": [[0.0] * graphs.NODE_FEATURE_DIM, [0.0] * graphs.NODE_FEATURE_DIM],
                    "pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "edge_index": [[0, 1], [1, 0]],
                    "edge_attr": [
                        [4.0, 1.0, 2.0, 1.0, 1.0, 1.0],
                        [4.0, 1.0, 2.0, 1.0, 1.0, 1.0],
                    ],
                    "graph_feat": [0.0] * graphs.GRAPH_FEAT_DIM,
                    "node_modality_mask": [[0, 0, 0], [0, 0, 0]],
                    "labels": {
                        "BPO": ["GO:0001"],
                        "CCO": [],
                        "MFO": ["GO:2000"] if index % 2 == 0 else [],
                    },
                }
            )
            graph_path = graphs_dir / f"{entry_id}.pt"
            graphs.torch.save(payload, graph_path)
            entries.append(
                {
                    "entry_id": entry_id,
                    "taxonomy_id": "9606",
                    "graph_path": str(graph_path.resolve()),
                    "fragment_count": 1,
                    "residue_count": 2,
                    "labels": {
                        "BPO": ["GO:0001"],
                        "CCO": [],
                        "MFO": ["GO:2000"] if index % 2 == 0 else [],
                    },
                }
            )

        (metadata_dir / "entries.json").write_text(
            __import__("json").dumps(entries, indent=2),
            encoding="utf-8",
        )
        (metadata_dir / "term_counts.json").write_text(
            __import__("json").dumps(
                {
                    "BPO": {"GO:0001": entry_count},
                    "CCO": {},
                    "MFO": {"GO:2000": sum(1 for index in range(entry_count) if index % 2 == 0)},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (metadata_dir / "schema.json").write_text("{}", encoding="utf-8")

        split_root = self.tmpdir / "splits"
        dataloaders.export_split_manifests(root=root, output_dir=split_root, aspects=["BPO"])
        return root, split_root

    @unittest.skipIf(training.GCNConv is None, "torch_geometric not installed")
    def test_minimal_pyg_classifier_forward_shape(self) -> None:
        root, split_root = self._write_graph_cache()
        datasets = dataloaders.build_split_datasets("pyg", root=root, aspect="BPO", split_dir=split_root)
        loader = dataloaders.build_pyg_loader(datasets["train"], batch_size=2, shuffle=False)
        batch = next(iter(loader))
        model = training.MinimalPygClassifier(
            input_dim=graphs.NODE_FEATURE_DIM,
            graph_feat_dim=graphs.GRAPH_FEAT_DIM,
            hidden_dim=16,
            output_dim=1,
            dropout=0.1,
        )

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 1))

    @unittest.skipIf(training.GraphConv is None, "dgl not installed")
    def test_minimal_dgl_classifier_forward_shape(self) -> None:
        root, split_root = self._write_graph_cache()
        datasets = dataloaders.build_split_datasets("dgl", root=root, aspect="BPO", split_dir=split_root)
        loader = dataloaders.build_dgl_loader(datasets["train"], batch_size=2, shuffle=False)
        batch = next(iter(loader))
        model = training.MinimalDglClassifier(
            input_dim=graphs.NODE_FEATURE_DIM,
            graph_feat_dim=graphs.GRAPH_FEAT_DIM,
            hidden_dim=16,
            output_dim=1,
            dropout=0.1,
        )

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 1))

    def test_multilabel_metrics_include_macro_and_fmax(self) -> None:
        logits = graphs.torch.tensor(
            [
                [10.0, -10.0],
                [10.0, 10.0],
                [-10.0, 10.0],
                [-10.0, -10.0],
            ]
        )
        labels = graphs.torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        metrics = training.multilabel_metrics_from_logits(
            logits,
            labels,
            threshold=0.5,
            fmax_threshold_step=0.5,
        )

        self.assertAlmostEqual(metrics["micro_precision"], 0.75)
        self.assertAlmostEqual(metrics["micro_recall"], 1.0)
        self.assertAlmostEqual(metrics["micro_f1"], 6 / 7)
        self.assertAlmostEqual(metrics["macro_f1"], (1.0 + (2 / 3)) / 2)
        self.assertAlmostEqual(metrics["macro_f1_positive_labels"], (1.0 + (2 / 3)) / 2)
        self.assertAlmostEqual(metrics["fmax"], 6 / 7)
        self.assertEqual(metrics["label_count"], 2.0)
        self.assertEqual(metrics["label_count_with_positive"], 2.0)

    def test_metric_selection_helpers(self) -> None:
        record = {
            "val": {
                "loss": 0.2,
                "micro_f1": 0.5,
                "macro_f1": 0.1,
                "fmax": 0.4,
            }
        }

        self.assertEqual(training.metric_optimization_mode("val_loss"), "min")
        self.assertEqual(training.metric_optimization_mode("val_fmax"), "max")
        self.assertAlmostEqual(training.metric_value_from_record(record, "val_fmax"), 0.4)
        self.assertTrue(training.metric_is_improved(0.42, 0.4, "max", min_delta=0.01))
        self.assertFalse(training.metric_is_improved(0.405, 0.4, "max", min_delta=0.01))
        self.assertTrue(training.metric_is_improved(0.18, 0.2, "min", min_delta=0.01))

    def test_build_pos_weight_tensor_supports_power_and_cap(self) -> None:
        class StubDataset:
            aspect = "MFO"
            vocab = ["GO:1000", "GO:2000"]
            term_to_index = {"GO:1000": 0, "GO:2000": 1}
            entries = [
                {"labels": {"MFO": ["GO:1000"]}},
                {"labels": {"MFO": ["GO:1000", "GO:2000"]}},
                {"labels": {"MFO": []}},
                {"labels": {"MFO": []}},
            ]

        pos_weight = training.build_pos_weight_tensor(StubDataset(), power=1.0, max_pos_weight=None)
        self.assertAlmostEqual(float(pos_weight[0].item()), 1.0)
        self.assertAlmostEqual(float(pos_weight[1].item()), 3.0)

        capped = training.build_pos_weight_tensor(StubDataset(), power=0.5, max_pos_weight=1.5)
        self.assertAlmostEqual(float(capped[0].item()), 1.0)
        self.assertAlmostEqual(float(capped[1].item()), 1.5)

    def test_load_go_parent_pairs_filters_to_vocab(self) -> None:
        obo_path = self.tmpdir / "go-basic.obo"
        obo_path.write_text(
            "\n".join(
                [
                    "format-version: 1.2",
                    "",
                    "[Term]",
                    "id: GO:0001",
                    "name: root",
                    "",
                    "[Term]",
                    "id: GO:0002",
                    "name: child",
                    "is_a: GO:0001 ! root",
                    "is_a: GO:9999 ! outside_vocab",
                    "",
                    "[Term]",
                    "id: GO:0003",
                    "name: ignored",
                    "is_a: GO:0001 ! root",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        pairs = training.load_go_parent_pairs(obo_path, ["GO:0001", "GO:0002"])

        self.assertEqual(pairs, [(1, 0)])

    def test_compute_label_ontology_regularization_matches_expected_penalty(self) -> None:
        embedding = graphs.torch.nn.Embedding(2, 2)
        with graphs.torch.no_grad():
            embedding.weight.copy_(graphs.torch.tensor([[0.0, 0.0], [1.0, 2.0]]))

        class StubModel:
            def __init__(self, label_embeddings):
                self.label_embeddings = label_embeddings

            def parameters(self):
                yield self.label_embeddings.weight

        model = StubModel(embedding)
        config = {
            "enabled": True,
            "weight": 0.5,
            "regularizer": graphs.torch.tensor([[1, 0]], dtype=graphs.torch.long),
        }

        regularization = training.compute_label_ontology_regularization(model, config)

        self.assertAlmostEqual(float(regularization.item()), 1.25)

    @unittest.skipIf(training.GCNConv is None, "torch_geometric not installed")
    def test_export_graph_prediction_bundles_writes_sequence_compatible_bundle(self) -> None:
        root, split_root = self._write_graph_cache()
        checkpoint_path = self.tmpdir / "best.pt"
        model = training.build_model(
            framework="pyg",
            output_dim=1,
            hidden_dim=8,
            dropout=0.1,
            model_head="flat_linear",
        )
        graphs.torch.save(
            {
                "framework": "pyg",
                "aspect": "BPO",
                "model_state": model.state_dict(),
                "args": {
                    "root": str(root),
                    "split_dir": str(split_root),
                    "aspect": "BPO",
                    "framework": "pyg",
                    "batch_size": 2,
                    "hidden_dim": 8,
                    "dropout": 0.1,
                    "model_head": "flat_linear",
                    "min_term_frequency": 1,
                    "seed": 2026,
                },
            },
            checkpoint_path,
        )
        output_dir = self.tmpdir / "graph_bundles"

        result = graph_exporter.main(
            [
                "--checkpoint-path",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--export-splits",
                "val",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(result, 0)
        val_bundle = output_dir / "val"
        self.assertTrue((val_bundle / "scores.npy").exists())
        self.assertTrue((val_bundle / "logits.npy").exists())
        self.assertTrue((val_bundle / "entry_ids.txt").exists())
        self.assertTrue((val_bundle / "terms.txt").exists())
        scores = __import__("numpy").load(val_bundle / "scores.npy")
        self.assertEqual(scores.shape[1], 1)
        meta = __import__("json").loads((val_bundle / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["aspect"], "BPO")
        self.assertEqual(meta["framework"], "pyg")


if __name__ == "__main__":
    unittest.main()
