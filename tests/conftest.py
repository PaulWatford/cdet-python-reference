"""Path setup for tests so they can be run directly (python tests/test_x.py)
or via pytest (python -m pytest tests/)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
