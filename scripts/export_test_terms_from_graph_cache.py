#!/usr/bin/env python3
"""Export CAFA-style ground-truth terms TSV from graph-cache test splits."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cafa_graph_dataloaders as dataloaders


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local test-split labels as a CAFA-style test_terms.tsv file."
    )
    parser.add_argument("--root", type=Path, required=True, help="Graph cache root containing metadata/entries.json.")
    parser.add_argument("--split-dir", type=Path, required=True, help="Split root containing <aspect>/test.txt files.")
    parser.add_argument(
        "--aspects",
        nargs="*",
        default=["CCO", "MFO"],
        help="Aspects to export. Defaults to the active graph-training aspects.",
    )
    parser.add_argument("--min-term-frequency", type=int, default=20, help="Min term frequency used by training.")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aspects = dataloaders.parse_aspects(args.aspects)

    entries = {}
    rows: list[tuple[str, str, str]] = []
    for aspect in aspects:
        filtered_entries, vocab = dataloaders.load_filtered_entries(
            root=args.root,
            aspect=aspect,
            min_term_frequency=args.min_term_frequency,
        )
        entry_lookup = {str(entry["entry_id"]): entry for entry in filtered_entries}
        entries.update(entry_lookup)
        vocab_set = set(vocab)
        test_ids = dataloaders.load_split_ids(args.split_dir / aspect.lower() / "test.txt")
        for entry_id in test_ids:
            entry = entry_lookup.get(entry_id)
            if entry is None:
                continue
            labels = sorted(set((entry.get("labels") or {}).get(aspect, [])) & vocab_set)
            for term in labels:
                rows.append((entry_id, term, aspect))

    rows = sorted(set(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["EntryID", "term", "aspect"])
        writer.writerows(rows)

    print(f"wrote {args.output}")
    print(f"rows={len(rows)}")
    print(f"aspects={','.join(aspects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
