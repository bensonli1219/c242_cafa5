from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

import cafa5_alphafold_pipeline as pipeline


def make_args(tmpdir: Path, limit: int | None = None, resume: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        train_taxonomy=tmpdir / "train_taxonomy.tsv",
        train_sequences=tmpdir / "train_sequences.fasta",
        train_terms=tmpdir / "train_terms.tsv",
        output_dir=tmpdir / "output",
        taxonomy_ids=None,
        limit=limit,
        workers=2,
        request_delay=0.0,
        overwrite=False,
        resume=resume,
        log_level="ERROR",
    )


class FakeClient:
    def __init__(self, metadata_map, artifacts):
        self.metadata_map = metadata_map
        self.artifacts = artifacts
        self.fetch_calls = 0
        self.download_calls = 0

    def fetch_exact_metadata(self, entry_id: str) -> pipeline.FetchResult:
        self.fetch_calls += 1
        value = self.metadata_map[entry_id]
        if isinstance(value, pipeline.FetchResult):
            return value
        return pipeline.FetchResult(status="ok", records=value)

    def download_to_path(self, url: str, dest_path: Path) -> None:
        self.download_calls += 1
        payload = self.artifacts[url]
        pipeline.ensure_parent(dest_path)
        if isinstance(payload, bytes):
            dest_path.write_bytes(payload)
        else:
            dest_path.write_text(payload, encoding="utf-8")


class Cafa5AlphaFoldPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir_context = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_context.name)
        (self.tmpdir / "tests").mkdir(exist_ok=True)

    def tearDown(self) -> None:
        self.tmpdir_context.cleanup()

    def write_cafa_inputs(self) -> None:
        (self.tmpdir / "train_taxonomy.tsv").write_text(
            "EntryID\ttaxonomyID\n"
            "P12345\t9606\n"
            "P99999\t9606\n",
            encoding="utf-8",
        )
        (self.tmpdir / "train_sequences.fasta").write_text(
            ">P12345 some protein\n"
            "MSTNPKPQRK\n"
            ">P99999 other protein\n"
            "AAAAA\n",
            encoding="utf-8",
        )
        (self.tmpdir / "train_terms.tsv").write_text(
            "EntryID\tterm\taspect\n"
            "P12345\tGO:0001\tBPO\n"
            "P12345\tGO:0002\tMFO\n"
            "P99999\tGO:0003\tCCO\n",
            encoding="utf-8",
        )

    def test_build_protein_records_parses_and_joins_inputs(self) -> None:
        self.write_cafa_inputs()
        records = pipeline.build_protein_records(
            self.tmpdir / "train_taxonomy.tsv",
            self.tmpdir / "train_sequences.fasta",
            self.tmpdir / "train_terms.tsv",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].entry_id, "P12345")
        self.assertEqual(records[0].taxonomy_id, "9606")
        self.assertEqual(records[0].sequence, "MSTNPKPQRK")
        self.assertEqual(records[0].go_terms_bpo, ["GO:0001"])
        self.assertEqual(records[0].go_terms_mfo, ["GO:0002"])
        self.assertEqual(records[1].go_terms_cco, ["GO:0003"])

    def test_filter_exact_metadata_records_excludes_isoforms(self) -> None:
        payload = [
            {"uniprotAccession": "Q8IXT2", "modelEntityId": "AF-Q8IXT2-F1"},
            {"uniprotAccession": "Q8IXT2-2", "modelEntityId": "AF-Q8IXT2-2-F1"},
        ]

        exact = pipeline.filter_exact_metadata_records("Q8IXT2", payload)

        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["uniprotAccession"], "Q8IXT2")

    def test_pipeline_writes_fragment_and_training_manifests_for_multifragment_entry(self) -> None:
        self.write_cafa_inputs()
        pae_one = json.dumps([{"predicted_aligned_error": [[1.0, 2.0], [3.0, 4.0]]}])
        pae_two = json.dumps([{"predicted_aligned_error": [[5.0, 6.0], [7.0, 8.0]]}])
        metadata_map = {
            "P12345": [
                {
                    "uniprotAccession": "P12345",
                    "modelEntityId": "AF-P12345-F1",
                    "sequenceStart": 1,
                    "sequenceEnd": 4,
                    "latestVersion": 6,
                    "allVersions": [5, 6],
                    "modelCreatedDate": "2025-08-01T00:00:00Z",
                    "globalMetricValue": 80.0,
                    "fractionPlddtVeryLow": 0.1,
                    "fractionPlddtLow": 0.2,
                    "fractionPlddtConfident": 0.3,
                    "fractionPlddtVeryHigh": 0.4,
                    "pdbUrl": "https://example.org/AF-P12345-F1-model_v6.pdb",
                    "paeDocUrl": "https://example.org/AF-P12345-F1-pae.json",
                },
                {
                    "uniprotAccession": "P12345",
                    "modelEntityId": "AF-P12345-F2",
                    "sequenceStart": 5,
                    "sequenceEnd": 8,
                    "latestVersion": 6,
                    "allVersions": [6],
                    "modelCreatedDate": "2025-08-01T00:00:00Z",
                    "globalMetricValue": 60.0,
                    "fractionPlddtVeryLow": 0.2,
                    "fractionPlddtLow": 0.3,
                    "fractionPlddtConfident": 0.2,
                    "fractionPlddtVeryHigh": 0.3,
                    "pdbUrl": "https://example.org/AF-P12345-F2-model_v6.pdb",
                    "paeDocUrl": "https://example.org/AF-P12345-F2-pae.json",
                },
            ],
            "P99999": pipeline.FetchResult(
                status="not_found",
                records=[],
                error="not found",
                http_status=404,
            ),
        }
        client = FakeClient(
            metadata_map=metadata_map,
            artifacts={
                "https://example.org/AF-P12345-F1-model_v6.pdb": b"MODEL1\n",
                "https://example.org/AF-P12345-F2-model_v6.pdb": b"MODEL2\n",
                "https://example.org/AF-P12345-F1-pae.json": pae_one,
                "https://example.org/AF-P12345-F2-pae.json": pae_two,
            },
        )

        summary = pipeline.run_pipeline(make_args(self.tmpdir), client=client)

        self.assertEqual(summary["entries"], 2)
        self.assertEqual(summary["fragment_rows"], 2)
        fragments = pq.read_table(
            self.tmpdir / "output" / "manifests" / "alphafold_fragments.parquet"
        ).to_pylist()
        training = pq.read_table(
            self.tmpdir / "output" / "manifests" / "training_index.parquet"
        ).to_pylist()

        self.assertEqual(len(fragments), 2)
        self.assertEqual(fragments[0]["entry_id"], "P12345")
        self.assertEqual(training[0]["af_fragment_count"], 2)
        self.assertEqual(training[0]["af_model_versions"], [5, 6])
        self.assertEqual(training[0]["af_model_entity_ids"], ["AF-P12345-F1", "AF-P12345-F2"])
        self.assertAlmostEqual(training[0]["af_mean_plddt"], 70.0)
        self.assertAlmostEqual(training[0]["af_pae_mean"], 4.5)
        self.assertEqual(training[1]["af_status"], "not_found")

    def test_pipeline_records_failures_for_not_found_and_pae_parse_error(self) -> None:
        self.write_cafa_inputs()
        metadata_map = {
            "P12345": [
                {
                    "uniprotAccession": "P12345",
                    "modelEntityId": "AF-P12345-F1",
                    "sequenceStart": 1,
                    "sequenceEnd": 4,
                    "latestVersion": 6,
                    "allVersions": [6],
                    "modelCreatedDate": "2025-08-01T00:00:00Z",
                    "globalMetricValue": 90.0,
                    "fractionPlddtVeryLow": 0.0,
                    "fractionPlddtLow": 0.1,
                    "fractionPlddtConfident": 0.3,
                    "fractionPlddtVeryHigh": 0.6,
                    "pdbUrl": "https://example.org/AF-P12345-F1-model_v6.pdb",
                    "paeDocUrl": "https://example.org/AF-P12345-F1-pae.json",
                }
            ],
            "P99999": pipeline.FetchResult(
                status="not_found",
                records=[],
                error="not found",
                http_status=404,
            ),
        }
        client = FakeClient(
            metadata_map=metadata_map,
            artifacts={
                "https://example.org/AF-P12345-F1-model_v6.pdb": b"MODEL\n",
                "https://example.org/AF-P12345-F1-pae.json": "{bad json",
            },
        )

        pipeline.run_pipeline(make_args(self.tmpdir), client=client)

        with (self.tmpdir / "output" / "manifests" / "download_failures.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            failures = list(csv.DictReader(handle))
        training = pq.read_table(
            self.tmpdir / "output" / "manifests" / "training_index.parquet"
        ).to_pylist()

        statuses = {(row["entry_id"], row["status"], row["stage"]) for row in failures}
        self.assertIn(("P99999", "not_found", "metadata"), statuses)
        self.assertIn(("P12345", "parse_error", "parse_pae"), statuses)
        self.assertEqual(training[0]["af_status"], "partial_error")
        self.assertTrue(training[0]["download_error"])

    def test_resume_skips_redownloading_existing_artifacts(self) -> None:
        self.write_cafa_inputs()
        metadata_map = {
            "P12345": [
                {
                    "uniprotAccession": "P12345",
                    "modelEntityId": "AF-P12345-F1",
                    "sequenceStart": 1,
                    "sequenceEnd": 4,
                    "latestVersion": 6,
                    "allVersions": [6],
                    "modelCreatedDate": "2025-08-01T00:00:00Z",
                    "globalMetricValue": 80.0,
                    "fractionPlddtVeryLow": 0.1,
                    "fractionPlddtLow": 0.2,
                    "fractionPlddtConfident": 0.3,
                    "fractionPlddtVeryHigh": 0.4,
                    "pdbUrl": "https://example.org/AF-P12345-F1-model_v6.pdb",
                    "paeDocUrl": "https://example.org/AF-P12345-F1-pae.json",
                }
            ],
            "P99999": pipeline.FetchResult(
                status="not_found",
                records=[],
                error="not found",
                http_status=404,
            ),
        }
        client = FakeClient(
            metadata_map=metadata_map,
            artifacts={
                "https://example.org/AF-P12345-F1-model_v6.pdb": b"MODEL\n",
                "https://example.org/AF-P12345-F1-pae.json": json.dumps(
                    [{"predicted_aligned_error": [[1.0, 2.0], [3.0, 4.0]]}]
                ),
            },
        )

        pipeline.run_pipeline(make_args(self.tmpdir, resume=False), client=client)
        first_download_calls = client.download_calls
        pipeline.run_pipeline(make_args(self.tmpdir, resume=True), client=client)

        self.assertEqual(first_download_calls, 2)
        self.assertEqual(client.download_calls, 2)


if __name__ == "__main__":
    unittest.main()
