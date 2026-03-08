import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.semantic_decomposition.semantic_decomposer import SemanticDecomposer

decomposer = SemanticDecomposer()

with open(PROJECT_ROOT / "data" / "valid.json") as handle:
    data = json.load(handle)

samples = random.sample(data, 15)

for prompt in samples:

    tasks = decomposer.decompose(prompt)

    print("\nPROMPT:")
    print(prompt)

    print("\nTASKS:")
    for t in tasks:
        print(t)