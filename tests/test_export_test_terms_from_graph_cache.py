from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs
from scripts import export_test_terms_from_graph_cache as exporter


@unittest.skipIf(graphs.torch is None, "graph runtime is only available in the Python 3.11 env")
class ExportTestTermsFromGraphCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def _write_graph_cache(self) -> tuple[Path, Path]:
        root = self.tmpdir / "graph_cache"
        metadata_dir = root / "metadata"
        graphs_dir = root / "graphs"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        labels_by_entry = {
            "P00001": {"CCO": ["GO:0001"], "MFO": ["GO:1000", "GO:1001"], "BPO": []},
            "P00002": {"CCO": ["GO:0002"], "MFO": ["GO:1000"], "BPO": []},
            "P00003": {"CCO": [], "MFO": [], "BPO": []},
        }
        entries = []
        for entry_id, labels in labels_by_entry.items():
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
                    "edge_attr": [[0.0] * graphs.EDGE_ATTR_DIM, [0.0] * graphs.EDGE_ATTR_DIM],
                    "graph_feat": [0.0] * graphs.GRAPH_FEAT_DIM,
                    "node_modality_mask": [[0, 0, 0], [0, 0, 0]],
                    "labels": labels,
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
                    "labels": labels,
                }
            )

        (metadata_dir / "entries.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        (metadata_dir / "term_counts.json").write_text(
            json.dumps(
                {"BPO": {}, "CCO": {"GO:0001": 1, "GO:0002": 1}, "MFO": {"GO:1000": 2, "GO:1001": 1}},
                indent=2,
            ),
            encoding="utf-8",
        )
        (metadata_dir / "schema.json").write_text("{}", encoding="utf-8")

        split_root = self.tmpdir / "splits"
        dataloaders.export_split_manifests(root=root, output_dir=split_root, aspects=["CCO", "MFO"], seed=2026)
        (split_root / "cco" / "test.txt").write_text("P00001\nP00003\n", encoding="utf-8")
        (split_root / "mfo" / "test.txt").write_text("P00001\nP00002\n", encoding="utf-8")
        return root, split_root

    def test_exporter_writes_cafa_style_rows(self) -> None:
        root, split_root = self._write_graph_cache()
        output_path = self.tmpdir / "test_terms.tsv"

        result = exporter.main(
            [
                "--root",
                str(root),
                "--split-dir",
                str(split_root),
                "--aspects",
                "CCO",
                "MFO",
                "--min-term-frequency",
                "1",
                "--output",
                str(output_path),
            ]
        )

        self.assertEqual(result, 0)
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))

        self.assertEqual(rows[0], ["EntryID", "term", "aspect"])
        self.assertEqual(
            rows[1:],
            [
                ["P00001", "GO:0001", "CCO"],
                ["P00001", "GO:1000", "MFO"],
                ["P00001", "GO:1001", "MFO"],
                ["P00002", "GO:1000", "MFO"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
