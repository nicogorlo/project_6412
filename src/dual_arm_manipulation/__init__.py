import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import logging

if not (ROOT_DIR / "output").exists():
    os.makedirs(ROOT_DIR / "output")

if not (ROOT_DIR / "output" / "method.log").exists():
    with open(ROOT_DIR / "output" / "method.log", "w") as f:
        f.write("")
        
logging.basicConfig(
    filename=ROOT_DIR / "output" / "method.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)