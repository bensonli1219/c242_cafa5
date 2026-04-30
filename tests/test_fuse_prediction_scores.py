from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fuse_prediction_scores as fusion


class FusePredictionScoresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def _write_bundle(self, root: Path, scores: np.ndarray, entry_ids: list[str], terms: list[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "scores.npy", scores.astype(np.float32))
        (root / "entry_ids.txt").write_text("\n".join(entry_ids) + "\n", encoding="utf-8")
        (root / "terms.txt").write_text("\n".join(terms) + "\n", encoding="utf-8")
        (root / "meta.json").write_text(json.dumps({"source": root.name}, indent=2), encoding="utf-8")

    def test_main_fuses_probability_scores(self) -> None:
        graph_root = self.tmpdir / "graph"
        sequence_root = self.tmpdir / "sequence"
        output_root = self.tmpdir / "fused"
        entry_ids = ["P1", "P2"]
        terms = ["GO:1", "GO:2"]
        self._write_bundle(
            graph_root,
            np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
            entry_ids,
            terms,
        )
        self._write_bundle(
            sequence_root,
            np.array([[0.6, 0.4], [0.1, 0.9]], dtype=np.float32),
            entry_ids,
            terms,
        )

        result = fusion.main(
            [
                "--graph-bundle",
                str(graph_root),
                "--sequence-bundle",
                str(sequence_root),
                "--output-dir",
                str(output_root),
                "--graph-weight",
                "0.25",
                "--sequence-weight",
                "0.75",
                "--score-space",
                "probabilities",
            ]
        )

        self.assertEqual(result, 0)
        fused_scores = np.load(output_root / "scores.npy")
        expected = np.array([[0.5, 0.5], [0.175, 0.825]], dtype=np.float32)
        self.assertTrue(np.allclose(fused_scores, expected))
        meta = json.loads((output_root / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["entry_count"], 2)
        self.assertEqual(meta["term_count"], 2)

    def test_main_fuses_logit_scores_and_writes_logits(self) -> None:
        graph_root = self.tmpdir / "graph_logits"
        sequence_root = self.tmpdir / "sequence_logits"
        output_root = self.tmpdir / "fused_logits"
        entry_ids = ["P1"]
        terms = ["GO:1", "GO:2"]
        self._write_bundle(
            graph_root,
            np.array([[0.0, 2.0]], dtype=np.float32),
            entry_ids,
            terms,
        )
        self._write_bundle(
            sequence_root,
            np.array([[2.0, 0.0]], dtype=np.float32),
            entry_ids,
            terms,
        )

        result = fusion.main(
            [
                "--graph-bundle",
                str(graph_root),
                "--sequence-bundle",
                str(sequence_root),
                "--output-dir",
                str(output_root),
                "--graph-weight",
                "0.5",
                "--sequence-weight",
                "0.5",
                "--score-space",
                "logits",
            ]
        )

        self.assertEqual(result, 0)
        fused_logits = np.load(output_root / "logits.npy")
        self.assertTrue(np.allclose(fused_logits, np.array([[1.0, 1.0]], dtype=np.float32)))
        fused_scores = np.load(output_root / "scores.npy")
        self.assertTrue(np.allclose(fused_scores, np.array([[0.7310586, 0.7310586]], dtype=np.float32), atol=1e-6))

    def test_validate_compatible_bundles_rejects_mismatched_terms(self) -> None:
        graph_bundle = {
            "entry_ids": ["P1"],
            "terms": ["GO:1"],
        }
        sequence_bundle = {
            "entry_ids": ["P1"],
            "terms": ["GO:2"],
        }

        with self.assertRaises(ValueError):
            fusion.validate_compatible_bundles(graph_bundle, sequence_bundle)

    def test_align_bundle_to_reference_reorders_matching_entry_set(self) -> None:
        reference = {
            "entry_ids": ["P1", "P2"],
            "terms": ["GO:1"],
            "scores": np.array([[0.1], [0.2]], dtype=np.float32),
            "meta": {},
        }
        bundle = {
            "entry_ids": ["P2", "P1"],
            "terms": ["GO:1"],
            "scores": np.array([[0.8], [0.7]], dtype=np.float32),
            "meta": {},
        }

        aligned = fusion.align_bundle_to_reference(reference, bundle)

        self.assertEqual(aligned["entry_ids"], ["P1", "P2"])
        self.assertTrue(np.allclose(aligned["scores"], np.array([[0.7], [0.8]], dtype=np.float32)))
        self.assertTrue(aligned["meta"]["reordered_to_match_reference"])

    def test_main_can_evaluate_fused_scores_against_graph_metadata(self) -> None:
        graph_bundle = self.tmpdir / "graph_eval_bundle"
        sequence_bundle = self.tmpdir / "sequence_eval_bundle"
        output_root = self.tmpdir / "fused_eval"
        graph_cache = self.tmpdir / "graph_cache"
        (graph_cache / "metadata").mkdir(parents=True)
        entry_ids = ["P1", "P2"]
        terms = ["GO:1", "GO:2"]
        self._write_bundle(
            graph_bundle,
            np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
            entry_ids,
            terms,
        )
        self._write_bundle(
            sequence_bundle,
            np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
            entry_ids,
            terms,
        )
        (graph_cache / "metadata" / "entries.json").write_text(
            json.dumps(
                [
                    {"entry_id": "P1", "labels": {"MFO": ["GO:1"]}},
                    {"entry_id": "P2", "labels": {"MFO": ["GO:2"]}},
                ]
            ),
            encoding="utf-8",
        )

        result = fusion.main(
            [
                "--graph-bundle",
                str(graph_bundle),
                "--sequence-bundle",
                str(sequence_bundle),
                "--output-dir",
                str(output_root),
                "--evaluate-with-graph-root",
                str(graph_cache),
                "--aspect",
                "MFO",
            ]
        )

        self.assertEqual(result, 0)
        meta = json.loads((output_root / "meta.json").read_text(encoding="utf-8"))
        self.assertIn("evaluation", meta)
        self.assertAlmostEqual(meta["evaluation"]["fmax"], 1.0)
        self.assertEqual(meta["evaluation"]["aspect"], "MFO")


if __name__ == "__main__":
    unittest.main()
