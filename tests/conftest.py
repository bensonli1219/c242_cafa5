import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (str(SRC_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
