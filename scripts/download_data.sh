#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from datasets import load_dataset

def safe_load(dataset_id, **kwargs):
    try:
        load_dataset(dataset_id, **kwargs)
        print(f"ok: {dataset_id} {kwargs}")
    except Exception as exc:
        print(f"warn: could not pre-download {dataset_id} {kwargs}: {exc}")

safe_load("GEM/xlsum", name="indonesian", split="test")
safe_load("GEM/xlsum", name="swahili", split="test")
safe_load("GEM/xlsum", name="turkish", split="test")

# Some environments require local source mapping for CoInCo.
safe_load("coinco", split="test")

print("done")
PY
