from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = SRC_ROOT.parent
DATA_DIR = (SRC_ROOT.parent / "data/").resolve()
