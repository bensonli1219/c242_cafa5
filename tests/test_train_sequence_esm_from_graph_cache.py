from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import train_sequence_esm_from_graph_cache as sequence_training


@unittest.skipIf(sequence_training.torch is None, "torch is required for sequence-side training tests")
class TrainSequenceEsmFromGraphCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)
        self.run_root = self.tmpdir / "run_root"
        self.feature_dir = self.run_root / "sequence_artifacts" / "protein_esm2_t30_150m_640_from_graph_cache"
        self.split_dir = self.run_root / "sequence_artifacts" / "matched_structure_splits"
        self.graph_root = self.run_root / "graph_cache"

        self.feature_dir.mkdir(parents=True, exist_ok=True)
        (self.split_dir / "mfo").mkdir(parents=True, exist_ok=True)
        (self.graph_root / "metadata").mkdir(parents=True, exist_ok=True)

        np.save(self.feature_dir / "X.npy", np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ))
        (self.feature_dir / "entry_ids.txt").write_text("P1\nP2\nP3\nP4\n", encoding="utf-8")
        (self.feature_dir / "meta.json").write_text(json.dumps({"entry_count": 4}, indent=2), encoding="utf-8")

        (self.split_dir / "mfo" / "train.txt").write_text("P1\nP2\n", encoding="utf-8")
        (self.split_dir / "mfo" / "val.txt").write_text("P3\n", encoding="utf-8")
        (self.split_dir / "mfo" / "test.txt").write_text("P4\n", encoding="utf-8")
        (self.split_dir / "mfo" / "vocab.json").write_text(
            json.dumps({"GO:0001": 2, "GO:0002": 1}, indent=2),
            encoding="utf-8",
        )

        entries = [
            {"entry_id": "P1", "labels": {"MFO": ["GO:0001"]}},
            {"entry_id": "P2", "labels": {"MFO": ["GO:0002"]}},
            {"entry_id": "P3", "labels": {"MFO": ["GO:0001"]}},
            {"entry_id": "P4", "labels": {"MFO": ["GO:0002"]}},
        ]
        (self.graph_root / "metadata" / "entries.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def test_main_trains_and_writes_prediction_bundles(self) -> None:
        checkpoint_dir = self.run_root / "sequence_runs" / "test_mfo"
        result = sequence_training.main(
            [
                "--run-root",
                str(self.run_root),
                "--aspect",
                "MFO",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )

        self.assertEqual(result, 0)
        summary = json.loads((checkpoint_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["aspect"], "MFO")
        self.assertEqual(summary["model_type"], "linear")
        val_bundle = checkpoint_dir / "prediction_bundles" / "val"
        self.assertTrue((val_bundle / "scores.npy").exists())
        self.assertTrue((val_bundle / "logits.npy").exists())
        self.assertEqual((val_bundle / "entry_ids.txt").read_text(encoding="utf-8").splitlines(), ["P3"])
        self.assertEqual((val_bundle / "terms.txt").read_text(encoding="utf-8").splitlines(), ["GO:0001", "GO:0002"])
        scores = np.load(val_bundle / "scores.npy")
        self.assertEqual(scores.shape, (1, 2))

    def test_load_terms_supports_graph_vocab_dict(self) -> None:
        vocab_path = self.tmpdir / "vocab.json"
        vocab_path.write_text(json.dumps({"GO:1": 3, "GO:2": 1}, indent=2), encoding="utf-8")
        self.assertEqual(sequence_training.load_terms(vocab_path), ["GO:1", "GO:2"])


if __name__ == "__main__":
    unittest.main()
