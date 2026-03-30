from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import alphafold_feature_extractor as extractor


def write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class AlphaFoldFeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def test_parse_pdb_ca_residues_reads_coordinates_and_plddt(self) -> None:
        pdb_path = self.tmpdir / "demo.pdb"
        pdb_path.write_text(
            "\n".join(
                [
                    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 91.50           N  ",
                    "ATOM      2  CA  ALA A   1       1.000   2.000   3.000  1.00 91.50           C  ",
                    "ATOM      3  CA  GLY A   2       4.000   5.000   6.000  1.00 42.00           C  ",
                ]
            ),
            encoding="utf-8",
        )

        residues = extractor.parse_pdb_ca_residues(pdb_path)

        self.assertEqual(len(residues), 2)
        self.assertEqual(residues[0].residue_name1, "A")
        self.assertEqual(residues[0].plddt, 91.50)
        self.assertEqual(residues[1].residue_name1, "G")
        self.assertEqual(residues[1].residue_number, 2)

    def test_run_extraction_writes_fragment_residue_and_edge_features(self) -> None:
        raw_dir = self.tmpdir / "raw" / "P12345"
        raw_dir.mkdir(parents=True, exist_ok=True)

        pdb_path = raw_dir / "AF-P12345-F1-model_v1.pdb"
        pdb_path.write_text(
            "\n".join(
                [
                    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 95.00           C  ",
                    "ATOM      2  CA  CYS A   2       0.000   0.000   6.000  1.00 72.00           C  ",
                    "ATOM      3  CA  ASP A   3       0.000   0.000  14.000  1.00 35.00           C  ",
                ]
            ),
            encoding="utf-8",
        )

        pae_path = raw_dir / "AF-P12345-F1-predicted_aligned_error_v1.json"
        pae_path.write_text(
            json.dumps(
                [
                    {
                        "predicted_aligned_error": [
                            [0.0, 2.0, 15.0],
                            [2.0, 0.0, 3.0],
                            [15.0, 3.0, 0.0],
                        ]
                    }
                ]
            ),
            encoding="utf-8",
        )

        training_rows = [
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "sequence": "ACD",
                "af_status": "ok",
                "af_model_entity_ids": ["AF-P12345-F1"],
            }
        ]
        fragment_rows = [
            {
                "entry_id": "P12345",
                "taxonomy_id": "9606",
                "model_entity_id": "AF-P12345-F1",
                "sequence_start": 1,
                "sequence_end": 3,
                "pdb_path": str(pdb_path),
                "pae_path": str(pae_path),
            }
        ]

        training_index_path = self.tmpdir / "manifests" / "training_index.parquet"
        fragment_manifest_path = self.tmpdir / "manifests" / "alphafold_fragments.parquet"
        write_parquet(training_index_path, training_rows)
        write_parquet(fragment_manifest_path, fragment_rows)

        args = extractor.parse_args(
            [
                "--training-index",
                str(training_index_path),
                "--fragment-manifest",
                str(fragment_manifest_path),
                "--output-dir",
                str(self.tmpdir / "features"),
                "--contact-threshold",
                "10.0",
                "--strict-contact-threshold",
                "8.0",
            ]
        )

        summary = extractor.run_extraction(args)

        self.assertEqual(summary["entries"], 1)
        self.assertEqual(summary["fragments"], 1)

        residue_rows = pq.read_table(self.tmpdir / "features" / "residue_features.parquet").to_pylist()
        edge_rows = pq.read_table(self.tmpdir / "features" / "contact_graph_edges.parquet").to_pylist()
        fragment_feature_rows = pq.read_table(
            self.tmpdir / "features" / "fragment_features.parquet"
        ).to_pylist()

        self.assertEqual(len(residue_rows), 3)
        self.assertEqual(len(edge_rows), 2)
        self.assertEqual(len(fragment_feature_rows), 1)
        self.assertEqual(residue_rows[0]["cafa_residue_index"], 1)
        self.assertEqual(residue_rows[2]["plddt_bin"], "very_low")
        self.assertEqual(edge_rows[0]["is_sequential_neighbor"], True)
        self.assertAlmostEqual(edge_rows[0]["distance_ca"], 6.0)
        self.assertAlmostEqual(fragment_feature_rows[0]["mean_plddt"], (95.0 + 72.0 + 35.0) / 3.0)
        self.assertEqual(fragment_feature_rows[0]["contact_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
