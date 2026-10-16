import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TRAIN_APP_DEBUG"] = "1"   # let errors surface in tests instead of the friendly error page
