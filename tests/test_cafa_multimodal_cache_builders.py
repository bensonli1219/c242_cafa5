from __future__ import annotations

import unittest

import cafa_multimodal_cache_builders as builders


class CafaMultimodalCacheBuilderTests(unittest.TestCase):
    def test_chunk_sequence_windows_handles_overlap(self) -> None:
        windows = builders.chunk_sequence_windows(
            sequence_length=2500,
            max_residues_per_chunk=1000,
            chunk_overlap=200,
        )

        self.assertEqual(windows, [(0, 1000), (800, 1800), (1600, 2500)])

    def test_parse_freesasa_residue_number_handles_insertion_code(self) -> None:
        residue_number, insertion_code = builders.parse_freesasa_residue_number("123A")

        self.assertEqual(residue_number, 123)
        self.assertEqual(insertion_code, "A")

    def test_ss3_one_hot_maps_unknown_to_coil(self) -> None:
        self.assertEqual(builders.ss3_one_hot("H"), [1.0, 0.0, 0.0])
        self.assertEqual(builders.ss3_one_hot("E"), [0.0, 1.0, 0.0])
        self.assertEqual(builders.ss3_one_hot("T"), [0.0, 0.0, 1.0])

    def test_parse_freesasa_json_accepts_dict_style_payload(self) -> None:
        payload = {
            "structures": [
                {
                    "chains": {
                        "A": {
                            "residues": {
                                "42": {"area": {"total": 55.5}},
                                "43A": {"area": {"total": 12.0}},
                            }
                        }
                    }
                }
            ]
        }

        residues = builders.parse_freesasa_json(payload)

        self.assertEqual(residues[("A", 42, "")], 55.5)
        self.assertEqual(residues[("A", 43, "A")], 12.0)

    def test_parse_dssp_text_extracts_expected_fields(self) -> None:
        header = "  #  RESIDUE AA STRUCTURE"
        line = list(" " * 140)
        line[5:10] = list(f"{42:>5}")
        line[10] = " "
        line[11] = "A"
        line[13] = "L"
        line[16] = "H"
        line[34:38] = list(f"{123:>4}")
        line[103:109] = list(f"{-60.0:6.1f}")
        line[109:115] = list(f"{135.0:6.1f}")
        text = header + "\n" + "".join(line) + "\n"

        records = builders.parse_dssp_text(text)

        self.assertIn(("A", 42, ""), records)
        self.assertEqual(records[("A", 42, "")]["aa"], "L")
        self.assertEqual(records[("A", 42, "")]["secondary_code"], "H")
        self.assertEqual(records[("A", 42, "")]["accessibility"], 123.0)
        self.assertEqual(records[("A", 42, "")]["phi"], -60.0)
        self.assertEqual(records[("A", 42, "")]["psi"], 135.0)


if __name__ == "__main__":
    unittest.main()
