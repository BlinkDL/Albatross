import sys
from pathlib import Path


DEMO4_DIR = Path(__file__).resolve().parent
if str(DEMO4_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO4_DIR))

from app import main


if __name__ == "__main__":
    main()
