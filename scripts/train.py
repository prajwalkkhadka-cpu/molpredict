"""Training entry point:  python scripts/train.py --config config/config.yaml"""

import sys
from pathlib import Path

# Allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from molpredict.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
