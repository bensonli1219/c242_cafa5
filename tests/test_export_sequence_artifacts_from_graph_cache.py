from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import export_sequence_artifacts_from_graph_cache as exporter


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


class ExportSequenceArtifactsFromGraphCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)
        self.run_root = self.tmpdir / "outputs"

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def test_main_mirrors_graph_splits_without_protein_export(self) -> None:
        graph_splits = self.run_root / "graph_cache" / "splits"
        graph_vocabs = self.run_root / "graph_cache" / "metadata" / "vocabs"
        graph_splits.mkdir(parents=True, exist_ok=True)
        (graph_splits / "summary.json").write_text(
            json.dumps({"aspects": {"BPO": {}, "CCO": {}, "MFO": {}}}, indent=2),
            encoding="utf-8",
        )
        for aspect in ("BPO", "CCO", "MFO"):
            src_dir = graph_splits / aspect.lower()
            src_dir.mkdir(parents=True, exist_ok=True)
            for name, content in {
                "train.txt": f"{aspect}_train\n",
                "val.txt": f"{aspect}_val\n",
                "test.txt": f"{aspect}_test\n",
                "summary.json": json.dumps({"aspect": aspect}, indent=2),
            }.items():
                (src_dir / name).write_text(content, encoding="utf-8")
            graph_vocabs.mkdir(parents=True, exist_ok=True)
            (graph_vocabs / f"{aspect}.json").write_text(
                json.dumps({"GO:1": 3}, indent=2),
                encoding="utf-8",
            )

        result = exporter.main(
            [
                "--run-root",
                str(self.run_root),
                "--min-term-frequency",
                "20",
                "--skip-protein-esm",
            ]
        )

        self.assertEqual(result, 0)
        matched_root = self.run_root / "sequence_artifacts" / "matched_structure_splits"
        self.assertTrue((matched_root / "bpo" / "train.txt").exists())
        self.assertEqual(
            (matched_root / "cco" / "vocab.json").read_text(encoding="utf-8"),
            (graph_vocabs / "CCO.json").read_text(encoding="utf-8"),
        )
        summary = json.loads((matched_root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["min_term_frequency"], 20)
        self.assertEqual(sorted(summary["aspects"]), ["BPO", "CCO", "MFO"])

    @unittest.skipIf(exporter.torch is None, "torch is required for protein ESM export tests")
    def test_main_exports_protein_esm_matrix_from_graph_cache(self) -> None:
        write_parquet(
            self.run_root / "manifests" / "training_index.parquet",
            [
                {"entry_id": "P1", "af_status": "ok"},
                {"entry_id": "P2", "af_status": "ok"},
                {"entry_id": "P3", "af_status": "not_found"},
            ],
        )
        esm_cache_dir = self.run_root / "graph_cache" / "modality_cache" / "esm2"
        esm_cache_dir.mkdir(parents=True, exist_ok=True)

        exporter.torch.save(
            {"entry_id": "P1", "protein_embedding": exporter.torch.tensor([1.0, 2.0, 3.0])},
            esm_cache_dir / "P1.pt",
        )
        exporter.torch.save(
            {"entry_id": "P2", "protein_embedding": exporter.torch.tensor([4.0, 5.0, 6.0])},
            esm_cache_dir / "P2.pt",
        )

        result = exporter.main(
            [
                "--run-root",
                str(self.run_root),
                "--skip-matched-splits",
                "--progress-every",
                "1",
                "--workers",
                "2",
                "--progress-mode",
                "log",
            ]
        )

        self.assertEqual(result, 0)
        protein_root = (
            self.run_root
            / "sequence_artifacts"
            / "protein_esm2_t30_150m_640_from_graph_cache"
        )
        matrix = np.load(protein_root / "X.npy")
        self.assertEqual(matrix.shape, (2, 3))
        self.assertTrue(np.allclose(matrix[0], np.array([1.0, 2.0, 3.0], dtype=np.float32)))
        self.assertTrue(np.allclose(matrix[1], np.array([4.0, 5.0, 6.0], dtype=np.float32)))
        entry_ids = (protein_root / "entry_ids.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(entry_ids, ["P1", "P2"])
        meta = json.loads((protein_root / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["entry_count"], 2)
        self.assertEqual(meta["embedding_dim"], 3)


if __name__ == "__main__":
    unittest.main()
