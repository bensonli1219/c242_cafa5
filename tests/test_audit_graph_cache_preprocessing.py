from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import audit_graph_cache_preprocessing as audit


class AuditGraphCachePreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def _write_cache(self) -> Path:
        root = self.tmpdir / "graph_cache"
        metadata = root / "metadata"
        graphs_dir = root / "graphs"
        metadata.mkdir(parents=True)
        graphs_dir.mkdir(parents=True)
        entries = []
        for index in range(10):
            entry_id = f"P{index:05d}"
            graph_path = graphs_dir / f"{entry_id}.pt"
            graph_path.write_bytes(b"placeholder")
            entries.append(
                {
                    "entry_id": entry_id,
                    "graph_path": str(graph_path),
                    "fragment_count": 1,
                    "residue_count": 100 + (index * 100),
                    "labels": {
                        "BPO": ["GO:1"] if index < 8 else ["GO:2"],
                        "CCO": ["GO:3"] if index % 2 == 0 else [],
                        "MFO": ["GO:4"] if index < 6 else [],
                    },
                }
            )
        (metadata / "entries.json").write_text(json.dumps(entries), encoding="utf-8")
        (metadata / "term_counts.json").write_text(
            json.dumps(
                {
                    "BPO": {"GO:1": 8, "GO:2": 2},
                    "CCO": {"GO:3": 5},
                    "MFO": {"GO:4": 6},
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_main_writes_audit_tables_and_experiment_splits(self) -> None:
        root = self._write_cache()
        output_dir = self.tmpdir / "audit"

        status = audit.main(
            [
                "--root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--aspects",
                "BPO",
                "MFO",
                "--min-term-frequencies",
                "2",
                "6",
                "--graph-residue-caps",
                "500",
                "1000",
                "--write-experiment-splits",
                "--experiment-min-term-frequency",
                "6",
                "--experiment-max-residues",
                "500",
                "--experiment-sample-per-aspect",
                "4",
            ]
        )

        self.assertEqual(status, 0)
        self.assertTrue((output_dir / "overview.csv").exists())
        self.assertTrue((output_dir / "min_frequency_scenarios.csv").exists())
        self.assertTrue((output_dir / "graph_cap_scenarios.csv").exists())
        split_summary = json.loads(
            (output_dir / "experiment_splits" / "sample_mtf20_cap1200" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(split_summary["aspects"]["BPO"]["entry_count"], 4)
        self.assertLessEqual(split_summary["aspects"]["MFO"]["entry_count"], 4)


if __name__ == "__main__":
    unittest.main()
