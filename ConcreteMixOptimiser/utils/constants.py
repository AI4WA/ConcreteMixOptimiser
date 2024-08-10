import os
from pathlib import Path

DATA = "data"

PROJECT_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

TEMP_OUTPUT_DIR = Path("/tmp")

LOG_DIR: Path = PROJECT_DIR / "logs"

REPORT_DIR = PROJECT_DIR / "reports"

DATA_DIR = PROJECT_DIR / DATA

for folder in [LOG_DIR, DATA_DIR]:
    if not folder.exists():
        folder.mkdir(exist_ok=True, parents=True)
