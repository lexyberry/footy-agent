import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("FOOTY_HOME", Path(__file__).resolve().parent.parent))


@lru_cache
def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("FOOTY_CONFIG", ROOT / "config.yaml"))
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["db_path"] = str(ROOT / cfg["db_path"])
    cfg["reports_dir"] = str(ROOT / cfg["reports_dir"])
    return cfg
