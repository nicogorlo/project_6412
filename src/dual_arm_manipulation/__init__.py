import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging

if not os.path.exists(os.path.join(ROOT_DIR, "output")):
    os.makedirs(os.path.join(ROOT_DIR, "output"))

if not os.path.exists(os.path.join(ROOT_DIR, "output/method.log")):
    with open(os.path.join(ROOT_DIR, "output/method.log"), "w") as f:
        f.write("")
        
logging.basicConfig(
    filename=os.path.join(ROOT_DIR, "output/method.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)