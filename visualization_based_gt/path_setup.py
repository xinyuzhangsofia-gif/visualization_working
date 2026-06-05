from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)
