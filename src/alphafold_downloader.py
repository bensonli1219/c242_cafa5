"""
alphafold_downloader.py
=======================
Utility script for downloading protein structure data from the AlphaFold
Protein Structure Database (https://alphafold.ebi.ac.uk).

Course: C242 – Machine Learning, Statistical Models, and Optimization
        for Molecular Problems
Purpose: Programmatic access to AlphaFold predicted structures and associated
         confidence metrics for use in downstream ML analyses.

---------------------------------------------------------------------------
OVERVIEW OF AVAILABLE DATA TYPES
---------------------------------------------------------------------------

For every protein entry the AlphaFold DB provides up to four file types,
all keyed on a **UniProt accession ID** (e.g. "P00520"):

  1. PDB  (.pdb)
     Classic Protein Data Bank format.  Contains 3-D atomic coordinates of
     every predicted residue.  The B-factor column is repurposed to store
     the per-residue pLDDT confidence score (0–100).  Widely supported by
     PyMOL, VMD, Biopython, MDAnalysis, etc.  Recommended for quick visual
     inspection or legacy pipelines.

  2. mmCIF  (.cif)
     PDBx/mmCIF format – the current wwPDB standard.  Richer metadata than
     PDB; no 99,999-atom limit.  Preferred for large proteins or when full
     metadata is needed.

  3. Binary CIF  (.bcif)
     Compressed binary version of mmCIF.  ~5–10× smaller; useful when
     downloading many structures.  Requires a bcif-aware parser (e.g.
     gemmi, py-mmcif).

  4. Predicted Aligned Error  (predicted_aligned_error_v4.json)
     PAE matrix: an (N × N) array of floats where entry [i][j] gives
     AlphaFold's expected error (in Å) in the position of residue j when
     residue i is used as the alignment anchor.  Low values along the
     diagonal → confident local structure; low off-diagonal values →
     confident *relative* domain placement.  Critical for identifying
     domain boundaries and for graph-based ML features.

Additional per-residue metadata (pLDDT scores) are embedded directly in the
coordinate files but are also described in the API JSON response.

---------------------------------------------------------------------------
API ENTRY POINT
---------------------------------------------------------------------------

  GET https://alphafold.ebi.ac.uk/api/prediction/{UniProt_accession}

Returns a JSON list (usually length 1) with a metadata record for each
fragment.  Long proteins (>2,700 residues) are split into overlapping
fragments: F1, F2, … The record includes direct download URLs for all
file types, so you never need to construct them manually.

Example metadata record fields:
  entryId            – e.g. "AF-P00520-F1"
  uniprotAccession   – e.g. "P00520"
  uniprotDescription – human-readable protein name
  modelCreatedDate   – ISO date string
  latestVersion      – integer (currently 4)
  allVersions        – list of all available model versions
  pdbUrl             – direct URL to .pdb file
  cifUrl             – direct URL to .cif file
  bcifUrl            – direct URL to .bcif file
  paeDocUrl          – direct URL to PAE JSON file
  paeImageUrl        – direct URL to PAE heatmap PNG
  globalMetricValue  – mean pLDDT over all residues (0–100)

---------------------------------------------------------------------------
DEPENDENCIES
---------------------------------------------------------------------------
  pip install requests tqdm
  (numpy + matplotlib optional – needed only for PAE visualisation)

"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import matplotlib.pyplot as plt
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    print("[INFO] numpy/matplotlib not found – PAE visualisation disabled.\n"
          "       Install with: pip install numpy matplotlib")

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api"
ALPHAFOLD_FILES_BASE = "https://alphafold.ebi.ac.uk/files"

# How long to wait between requests (seconds) – be polite to the server
REQUEST_DELAY = 0.5

# Retry strategy for transient HTTP errors
_RETRY_STRATEGY = Retry(
    total=4,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
)


# ===========================================================================
# SESSION FACTORY  –  creates a requests.Session with retry logic
# ===========================================================================

def _build_session() -> requests.Session:
    """Return a requests.Session with automatic retry on transient errors."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


# ===========================================================================
# 1.  METADATA QUERY
# ===========================================================================

def fetch_metadata(uniprot_accession: str) -> list[dict]:
    """
    Query the AlphaFold API for all prediction fragments for a protein.

    Parameters
    ----------
    uniprot_accession : str
        UniProt accession ID, e.g. "P00520" or "Q5VSL9".
        Case-insensitive – will be uppercased automatically.

    Returns
    -------
    list of dict
        One dictionary per fragment (most proteins have only F1).
        Each dict contains download URLs and metadata fields described in
        the module docstring.

    Raises
    ------
    ValueError
        If the accession is not found in the AlphaFold DB.
    requests.HTTPError
        On non-404 HTTP errors.
    """
    accession = uniprot_accession.strip().upper()
    url = f"{ALPHAFOLD_API_BASE}/prediction/{accession}"
    log.info(f"Querying API: {url}")

    resp = SESSION.get(url, timeout=30)

    if resp.status_code == 404:
        raise ValueError(
            f"UniProt accession '{accession}' not found in AlphaFold DB. "
            "Check the accession at https://www.uniprot.org"
        )
    resp.raise_for_status()

    data = resp.json()
    if not data:
        raise ValueError(f"Empty response for accession '{accession}'.")

    log.info(f"Found {len(data)} fragment(s) for {accession}: "
             f"{data[0].get('uniprotDescription', 'N/A')}")
    return data


def print_metadata_summary(metadata: list[dict]) -> None:
    """
    Pretty-print a summary of what's available for a given accession.

    Parameters
    ----------
    metadata : list of dict
        Output of fetch_metadata().
    """
    for frag in metadata:
        print("\n" + "="*60)
        print(f"  Entry:        {frag.get('entryId')}")
        print(f"  UniProt ID:   {frag.get('uniprotAccession')}")
        print(f"  Protein:      {frag.get('uniprotDescription')}")
        print(f"  Model date:   {frag.get('modelCreatedDate')}")
        print(f"  Version:      v{frag.get('latestVersion')}  "
              f"(all: {frag.get('allVersions')})")
        print(f"  Mean pLDDT:   {frag.get('globalMetricValue', 'N/A'):.1f}/100")
        print("\n  Available download URLs:")
        for key in ("pdbUrl", "cifUrl", "bcifUrl", "paeDocUrl", "paeImageUrl"):
            val = frag.get(key)
            if val:
                print(f"    {key:<16} {val}")
        print("="*60)


# ===========================================================================
# 2.  GENERIC FILE DOWNLOADER
# ===========================================================================

def download_file(url: str, dest_path: Path, overwrite: bool = False) -> Path:
    """
    Stream-download a file from *url* to *dest_path*.

    Parameters
    ----------
    url : str
        Direct download URL.
    dest_path : Path
        Local file path to write to.  Parent directory is created if needed.
    overwrite : bool
        If False (default), skip download when the file already exists.

    Returns
    -------
    Path
        The path of the downloaded (or pre-existing) file.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not overwrite:
        log.info(f"  [skip] {dest_path.name} already exists")
        return dest_path

    log.info(f"  Downloading → {dest_path.name}")
    resp = SESSION.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))

    if HAVE_TQDM and total:
        chunks = tqdm(
            resp.iter_content(chunk_size=8192),
            total=total // 8192 + 1,
            unit="KB",
            desc=dest_path.name,
            leave=False,
        )
    else:
        chunks = resp.iter_content(chunk_size=8192)

    with open(dest_path, "wb") as fh:
        for chunk in chunks:
            if chunk:
                fh.write(chunk)

    size_kb = dest_path.stat().st_size / 1024
    log.info(f"  Saved {dest_path.name}  ({size_kb:.1f} KB)")
    return dest_path


# ===========================================================================
# 3.  TYPE-SPECIFIC DOWNLOADERS
# ===========================================================================

def download_pdb(metadata_record: dict,
                 output_dir: Path,
                 overwrite: bool = False) -> Optional[Path]:
    """
    Download the PDB-format structure file for one fragment.

    The B-factor column in the PDB file stores per-residue pLDDT (0–100).
    Load with Biopython:
        from Bio.PDB import PDBParser
        parser = PDBParser()
        struct = parser.get_structure("protein", "AF-P00520-F1-model_v4.pdb")

    Parameters
    ----------
    metadata_record : dict
        A single fragment record from fetch_metadata().
    output_dir : Path
        Directory where the file will be saved.
    overwrite : bool
        Overwrite existing files if True.

    Returns
    -------
    Path or None
        Downloaded file path, or None if no URL available.
    """
    url = metadata_record.get("pdbUrl")
    if not url:
        log.warning("No pdbUrl in metadata record.")
        return None

    filename = url.split("/")[-1]
    return download_file(url, Path(output_dir) / filename, overwrite)


def download_mmcif(metadata_record: dict,
                   output_dir: Path,
                   overwrite: bool = False) -> Optional[Path]:
    """
    Download the mmCIF-format structure file for one fragment.

    mmCIF is the current wwPDB standard and handles proteins of any size.
    Load with gemmi:
        import gemmi
        structure = gemmi.read_structure("AF-P00520-F1-model_v4.cif")

    Parameters
    ----------
    metadata_record : dict
    output_dir : Path
    overwrite : bool

    Returns
    -------
    Path or None
    """
    url = metadata_record.get("cifUrl")
    if not url:
        log.warning("No cifUrl in metadata record.")
        return None

    filename = url.split("/")[-1]
    return download_file(url, Path(output_dir) / filename, overwrite)


def download_bcif(metadata_record: dict,
                  output_dir: Path,
                  overwrite: bool = False) -> Optional[Path]:
    """
    Download the binary CIF (.bcif) structure file for one fragment.

    Binary CIF is ~5–10× smaller than plain mmCIF and is preferred when
    batch-downloading hundreds of structures.  Parse with gemmi or py-mmcif.

    Parameters
    ----------
    metadata_record : dict
    output_dir : Path
    overwrite : bool

    Returns
    -------
    Path or None
    """
    url = metadata_record.get("bcifUrl")
    if not url:
        log.warning("No bcifUrl in metadata record.")
        return None

    filename = url.split("/")[-1]
    return download_file(url, Path(output_dir) / filename, overwrite)


def download_pae(metadata_record: dict,
                 output_dir: Path,
                 overwrite: bool = False) -> Optional[Path]:
    """
    Download the Predicted Aligned Error (PAE) JSON file for one fragment.

    The PAE matrix is an (N × N) array of floats (in Ångströms).
      - pae[i][j] = expected position error at residue j when residue i
        is used as the alignment anchor.
      - Low values along the diagonal: confident local structure (similar
        to pLDDT but pairwise).
      - Low off-diagonal block values: confident *inter-domain* geometry –
        useful for identifying rigid structural units.

    JSON structure (v4):
      {
        "pae": [[float, ...], ...],    // N×N matrix, row-major
        "residues": [int, ...],        // 1-based residue indices
        "max_predicted_aligned_error": float,
        "predicted_aligned_error": [[...], ...]  // same as "pae", alias
      }

    Parameters
    ----------
    metadata_record : dict
    output_dir : Path
    overwrite : bool

    Returns
    -------
    Path or None
    """
    url = metadata_record.get("paeDocUrl")
    if not url:
        log.warning("No paeDocUrl in metadata record.")
        return None

    filename = url.split("/")[-1]
    return download_file(url, Path(output_dir) / filename, overwrite)


def download_pae_image(metadata_record: dict,
                       output_dir: Path,
                       overwrite: bool = False) -> Optional[Path]:
    """
    Download the PAE heatmap PNG image (pre-rendered by EBI).

    This is a quick visual of the PAE matrix – useful for reports and
    notebooks without needing to render it yourself.

    Parameters
    ----------
    metadata_record : dict
    output_dir : Path
    overwrite : bool

    Returns
    -------
    Path or None
    """
    url = metadata_record.get("paeImageUrl")
    if not url:
        log.warning("No paeImageUrl in metadata record.")
        return None

    filename = url.split("/")[-1]
    return download_file(url, Path(output_dir) / filename, overwrite)


# ===========================================================================
# 4.  PAE MATRIX UTILITIES  (requires numpy + matplotlib)
# ===========================================================================

def load_pae_matrix(pae_json_path: Path) -> "np.ndarray":
    """
    Load a downloaded PAE JSON file into a NumPy array.

    Parameters
    ----------
    pae_json_path : Path
        Path to the downloaded .json file from download_pae().

    Returns
    -------
    np.ndarray  shape (N, N)
        PAE matrix in Ångströms.

    Raises
    ------
    ImportError  if numpy is not installed.
    """
    if not HAVE_NUMPY:
        raise ImportError("numpy is required: pip install numpy")

    with open(pae_json_path) as fh:
        data = json.load(fh)

    # The key is "pae" or "predicted_aligned_error" depending on API version
    matrix_key = "pae" if "pae" in data else "predicted_aligned_error"
    matrix = np.array(data[matrix_key], dtype=np.float32)
    log.info(f"Loaded PAE matrix: shape {matrix.shape}, "
             f"max_PAE={matrix.max():.1f} Å")
    return matrix


def plot_pae_matrix(pae_matrix: "np.ndarray",
                    title: str = "Predicted Aligned Error",
                    save_path: Optional[Path] = None) -> None:
    """
    Render the PAE matrix as a heatmap.

    Low values (dark blue) indicate high confidence in relative positioning.
    High values (yellow) indicate low confidence.

    Parameters
    ----------
    pae_matrix : np.ndarray  shape (N, N)
        Output of load_pae_matrix().
    title : str
        Plot title.
    save_path : Path, optional
        If given, save the figure to this path; otherwise show interactively.

    Raises
    ------
    ImportError  if matplotlib is not installed.
    """
    if not HAVE_NUMPY:
        raise ImportError("matplotlib is required: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pae_matrix, cmap="bwr", vmin=0, vmax=30, origin="upper")
    ax.set_xlabel("Scored residue", fontsize=12)
    ax.set_ylabel("Aligned residue", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, label="Expected position error (Å)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"PAE plot saved → {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# ===========================================================================
# 5.  HIGH-LEVEL CONVENIENCE FUNCTION
# ===========================================================================

def download_all_for_accession(
    uniprot_accession: str,
    output_dir: str | Path,
    file_types: tuple[str, ...] = ("pdb", "cif", "pae"),
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """
    One-call download of all requested data types for a UniProt accession.

    This function:
      1. Queries the AlphaFold API for metadata (handles multi-fragment proteins)
      2. Downloads each requested file type for every fragment
      3. Returns a dict mapping file-type → list of downloaded paths

    Parameters
    ----------
    uniprot_accession : str
        UniProt accession ID, e.g. "P00520".
    output_dir : str or Path
        Root directory for downloads.  A subdirectory named after the
        accession will be created automatically.
    file_types : tuple of str
        Any combination of: "pdb", "cif", "bcif", "pae", "pae_image".
        Default: ("pdb", "cif", "pae")
    overwrite : bool
        Re-download even if files already exist.

    Returns
    -------
    dict
        Keys are file types; values are lists of Path objects (one per
        fragment).

    Example
    -------
    >>> paths = download_all_for_accession(
    ...     "P69905",                    # Human haemoglobin subunit alpha
    ...     output_dir="./af_downloads",
    ...     file_types=("pdb", "pae"),
    ... )
    >>> print(paths["pdb"])
    """
    accession = uniprot_accession.strip().upper()
    output_dir = Path(output_dir) / accession
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: fetch metadata
    metadata = fetch_metadata(accession)
    print_metadata_summary(metadata)

    # -- Step 2: save raw metadata JSON for provenance
    meta_path = output_dir / f"{accession}_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    log.info(f"Metadata saved → {meta_path.name}")

    # -- Step 3: dispatch downloaders
    downloader_map = {
        "pdb":       download_pdb,
        "cif":       download_mmcif,
        "bcif":      download_bcif,
        "pae":       download_pae,
        "pae_image": download_pae_image,
    }

    results: dict[str, list[Path]] = {ft: [] for ft in file_types}

    for frag_record in metadata:
        frag_id = frag_record.get("entryId", accession)
        log.info(f"\nProcessing fragment: {frag_id}")

        for ft in file_types:
            fn = downloader_map.get(ft)
            if fn is None:
                log.warning(f"Unknown file type '{ft}'; "
                             "valid options: pdb, cif, bcif, pae, pae_image")
                continue

            path = fn(frag_record, output_dir, overwrite=overwrite)
            if path:
                results[ft].append(path)

        time.sleep(REQUEST_DELAY)   # be polite to the server

    return results


# ===========================================================================
# 6.  BATCH DOWNLOADER  (list of accessions)
# ===========================================================================

def batch_download(
    accession_list: list[str],
    output_dir: str | Path,
    file_types: tuple[str, ...] = ("pdb", "cif", "pae"),
    overwrite: bool = False,
    skip_errors: bool = True,
) -> dict[str, dict]:
    """
    Download data for multiple UniProt accessions.

    Parameters
    ----------
    accession_list : list of str
        UniProt accession IDs to download.
    output_dir : str or Path
        Root directory; a subfolder per accession is created.
    file_types : tuple of str
        File types to fetch.  See download_all_for_accession().
    overwrite : bool
        Re-download existing files if True.
    skip_errors : bool
        If True, log errors and continue; otherwise re-raise.

    Returns
    -------
    dict
        Mapping { accession → { file_type → [Path, ...] } }.
        Failed accessions map to {"error": str(exception)}.
    """
    results = {}
    total = len(accession_list)

    for idx, acc in enumerate(accession_list, start=1):
        log.info(f"\n[{idx}/{total}] Starting {acc}")
        try:
            results[acc] = download_all_for_accession(
                acc, output_dir, file_types=file_types, overwrite=overwrite
            )
        except Exception as exc:
            log.error(f"  Failed for {acc}: {exc}")
            if skip_errors:
                results[acc] = {"error": str(exc)}
            else:
                raise

    return results


# ===========================================================================
# 7.  DEMO / ENTRY POINT
# ===========================================================================

def main() -> None:
    """
    Demonstrate the downloader with a few well-known proteins.

    Run from the command line:
        python alphafold_downloader.py

    Or supply a custom accession:
        python alphafold_downloader.py P00520
    """
    # Demo accessions (feel free to replace with your target proteins)
    demo_accessions = [
        "P69905",   # Human haemoglobin subunit alpha – small, well-known
        "P00520",   # Proto-oncogene ABL1 – multi-domain, good for PAE demo
    ]

    # If the user passes an accession on the command line, use that instead
    if len(sys.argv) > 1:
        demo_accessions = sys.argv[1:]

    # Output directory (relative to this script's location)
    script_dir = Path(__file__).parent
    output_dir = script_dir / "data" / "alphafold_demo"

    log.info("AlphaFold Downloader – starting demo")
    log.info(f"Output directory: {output_dir.resolve()}")

    for acc in demo_accessions:
        print(f"\n{'='*60}")
        print(f" Processing: {acc}")
        print(f"{'='*60}")

        paths = download_all_for_accession(
            uniprot_accession=acc,
            output_dir=output_dir,
            file_types=("pdb", "cif", "pae", "pae_image"),
            overwrite=False,
        )

        print("\n  Downloaded files:")
        for ft, file_list in paths.items():
            for fp in file_list:
                print(f"    [{ft}]  {fp}")

        # Optional: render PAE heatmap if numpy/matplotlib available
        if HAVE_NUMPY and paths.get("pae"):
            pae_path = paths["pae"][0]
            matrix = load_pae_matrix(pae_path)
            acc_dir = output_dir / acc
            plot_pae_matrix(
                matrix,
                title=f"PAE – {acc}",
                save_path=acc_dir / f"{acc}_pae_plot.png",
            )

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
