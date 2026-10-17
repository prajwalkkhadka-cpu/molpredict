"""Single-molecule predictor.

  python predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
  python predict.py --name caffeine
"""

import sys

from molpredict.predict_cli import main

if __name__ == "__main__":
    sys.exit(main())
