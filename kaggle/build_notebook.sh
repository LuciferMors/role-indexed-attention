#!/bin/bash
# Convert run_all.py -> run_all.ipynb and inject the kernelspec metadata
# papermill (used by Kaggle) requires for notebook kernels.
set -e
cd "$(dirname "$0")"
~/Library/Python/3.12/bin/jupytext --to notebook --update run_all.py
python3 - <<'PY'
import json
p = "run_all.ipynb"
nb = json.load(open(p))
nb.setdefault("metadata", {})
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
with open(p, "w") as f:
    json.dump(nb, f, indent=1)
print(f"OK wrote {p} with kernelspec metadata")
PY
