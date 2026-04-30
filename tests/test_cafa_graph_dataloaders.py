from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs


class CafaGraphDataloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def _write_graph_cache(self, entry_count: int = 5, materialize_graphs: bool = False) -> Path:
        root = self.tmpdir / "graph_cache"
        metadata_dir = root / "metadata"
        graphs_dir = root / "graphs"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for index in range(entry_count):
            entry_id = f"P{index + 1:05d}"
            graph_path = graphs_dir / f"{entry_id}.pt"
            if materialize_graphs:
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
                            "CCO": ["GO:1000"] if index % 2 == 0 else [],
                            "MFO": ["GO:2000"] if index % 3 == 0 else [],
                        },
                    }
                )
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
                        "CCO": ["GO:1000"] if index % 2 == 0 else [],
                        "MFO": ["GO:2000"] if index % 3 == 0 else [],
                    },
                }
            )

        with (metadata_dir / "entries.json").open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)
        with (metadata_dir / "term_counts.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "BPO": {"GO:0001": entry_count},
                    "CCO": {"GO:1000": sum(1 for index in range(entry_count) if index % 2 == 0)},
                    "MFO": {"GO:2000": sum(1 for index in range(entry_count) if index % 3 == 0)},
                },
                handle,
                indent=2,
            )
        with (metadata_dir / "schema.json").open("w", encoding="utf-8") as handle:
            json.dump({}, handle)
        return root

    def test_allocate_split_counts_preserves_total(self) -> None:
        counts = dataloaders.allocate_split_counts(47, (0.8, 0.1, 0.1))

        self.assertEqual(sum(counts), 47)
        self.assertEqual(counts, (37, 5, 5))

    def test_export_split_manifest_writes_train_val_test_files(self) -> None:
        root = self._write_graph_cache(entry_count=5, materialize_graphs=False)
        split_root = self.tmpdir / "splits"

        summary = dataloaders.export_split_manifests(root=root, output_dir=split_root, aspects=["BPO"])

        self.assertEqual(summary["aspects"]["BPO"]["entry_count"], 5)
        self.assertEqual(summary["aspects"]["BPO"]["counts"], {"train": 3, "val": 1, "test": 1})
        self.assertTrue((split_root / "bpo" / "train.txt").exists())
        self.assertTrue((split_root / "bpo" / "val.txt").exists())
        self.assertTrue((split_root / "bpo" / "test.txt").exists())


@unittest.skipIf(graphs.torch is None, "graph runtime is only available in the Python 3.11 env")
class CafaGraphDataloaderRuntimeTests(CafaGraphDataloaderTests):
    def _prepare_split_root(self) -> tuple[Path, Path]:
        root = self._write_graph_cache(entry_count=5, materialize_graphs=True)
        split_root = self.tmpdir / "splits"
        dataloaders.export_split_manifests(root=root, output_dir=split_root, aspects=["BPO"])
        return root, split_root

    @unittest.skipIf(dataloaders.PygDataLoader is None, "torch_geometric not installed")
    def test_build_pyg_dataloaders_returns_batched_graphs(self) -> None:
        root, split_root = self._prepare_split_root()

        loaders = dataloaders.build_pyg_dataloaders(root=root, aspect="BPO", split_dir=split_root, batch_size=2)
        batch = next(iter(loaders["train"]))

        self.assertEqual(batch.x.shape[1], graphs.NODE_FEATURE_DIM)
        self.assertEqual(batch.edge_attr.shape[1], graphs.EDGE_ATTR_DIM)
        self.assertEqual(batch.y.shape[1], 1)
        self.assertGreaterEqual(batch.num_graphs, 1)

    @unittest.skipIf(dataloaders.DGLGraphDataLoader is None, "dgl not installed")
    def test_build_dgl_dataloaders_returns_batched_graphs(self) -> None:
        root, split_root = self._prepare_split_root()

        loaders = dataloaders.build_dgl_dataloaders(root=root, aspect="BPO", split_dir=split_root, batch_size=2)
        batch = next(iter(loaders["train"]))

        self.assertEqual(batch.ndata["x"].shape[1], graphs.NODE_FEATURE_DIM)
        self.assertEqual(batch.edata["edge_attr"].shape[1], graphs.EDGE_ATTR_DIM)
        self.assertEqual(batch.y.shape[1], 1)
        self.assertGreaterEqual(len(batch.batch_num_nodes()), 1)


if __name__ == "__main__":
    unittest.main()
