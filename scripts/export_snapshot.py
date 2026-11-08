"""Copy the PostgreSQL database into the SQLite file the free Streamlit Community Cloud app reads.

    DATABASE_URL=postgresql://... python scripts/export_snapshot.py

Then commit database/delays.db and database/build_info.json and push. The live demo
updates itself; PostgreSQL stays the source of truth.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import db  # noqa: E402
from scripts.load_to_sql import INDEXES, VIEWS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ["delay_causes", "station_performance", "operator_punctuality"]


def main() -> int:
    if not db.is_postgres():
        print("[ERROR] Set DATABASE_URL to the PostgreSQL database you want to export from.")
        return 1
    out = ROOT / "database" / "delays.db"
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    print(f"Exporting {db.describe()} -> {out.relative_to(ROOT)}")

    target = create_engine(f"sqlite:///{out.as_posix()}")
    with db.connect() as src, target.begin() as dst:
        for t in TABLES:
            df = pd.read_sql(f"SELECT * FROM {t}", src)
            df.to_sql(t, dst, index=False)
            print(f"  {t}: {len(df):,} rows")
        for name, sql in VIEWS.items():
            dst.execute(text(f"CREATE VIEW {name} AS {sql}"))
        for stmt in INDEXES:
            dst.execute(text(stmt))
        info = json.loads(src.execute(text("SELECT info FROM build_info")).scalar())
    (ROOT / "database" / "build_info.json").write_text(json.dumps(info, indent=2))
    print("Done. Commit database/delays.db and database/build_info.json, then push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
