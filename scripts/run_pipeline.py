"""Run the whole refresh in order, stopping at the first failure.

    python scripts/run_pipeline.py              # clean -> check -> load -> retrain model
    python scripts/run_pipeline.py --skip-model # clean -> check -> load only

Exit codes: 0 ok, 1 a step crashed, 2 the data-quality gate blocked the load.
Used by `docker compose run --rm pipeline`.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(name: str, script: str, *args: str) -> int:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    code = subprocess.call([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT)
    print(f"--- {name}: {'ok' if code == 0 else f'FAILED (exit {code})'} in {time.time() - t0:.0f}s", flush=True)
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-clean", action="store_true", help="reuse data/processed from an earlier run")
    a = ap.parse_args()

    steps = []
    if not a.skip_clean:
        steps.append(("1. Clean the raw ORR files", "clean_data.py"))
    steps.append(("2. Check quality and load the database", "load_to_sql.py"))
    if not a.skip_model:
        steps.append(("3. Retrain the forecast model", "ml_model.py"))

    for name, script in steps:
        code = step(name, script)
        if code != 0:
            print(f"\nPipeline stopped at: {name}", flush=True)
            return code
    print("\nPipeline finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
