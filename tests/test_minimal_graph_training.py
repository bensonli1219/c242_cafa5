from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs
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


if __name__ == "__main__":
    unittest.main()
