# V2 setup: PostgreSQL, Docker and data-quality checks

V2 replaces "a database file inside the repo" with a real PostgreSQL database, wraps everything in Docker,
and adds a data-quality gate so bad data can never reach the dashboard.

```
 raw ORR files ──► clean_data.py ──► quality_checks.py ──► load_to_sql.py ──► PostgreSQL ──► Streamlit app
 (data/raw or S3)   (pandas)         32 checks, can BLOCK    one transaction     (Docker or RDS)   (Docker)
                                                                  │
                                                                  └──► ml_model.py (retrain) ──► models/
                                    export_snapshot.py ──► database/delays.db ──► free Streamlit Cloud demo
```

## 1. Run it locally with Docker (free, 10 minutes)

You need Docker Desktop. Put the three ORR files in `data/raw/`, then from the project folder:

```
docker compose up -d db
docker compose run --rm pipeline
docker compose up app
```

Open http://localhost:8501. Step 2 cleans the files, runs the 32 quality checks, loads PostgreSQL and retrains
the model. If a check fails, nothing is loaded and the command exits with an error.

Useful: `docker compose down` stops everything (your data stays), `docker compose down -v` also deletes the database.

## 2. Without Docker (plain Python)

```
export DATABASE_URL="postgresql://user:password@localhost:5432/traindelays"
python scripts/run_pipeline.py          # clean -> check -> load -> retrain
streamlit run app/streamlit_app.py
```

Leave `DATABASE_URL` unset and everything uses the SQLite file in `database/`, exactly like V1.

## 3. Put the database on AWS RDS (optional, costs money if left running)

Do step 1 first. Only do this once you are comfortable, and set an AWS billing alert before you start.

1. AWS console, RDS, Create database: **PostgreSQL 16**, template **Free tier** (if your account shows one),
   instance **db.t4g.micro**, storage 20 GB, database name `traindelays`, username `trains`.
   Choose a long password and keep it in a password manager. **Never paste it into a chat or commit it.**
2. Connectivity: same region as your S3 bucket. For loading from your laptop choose **Public access: Yes**, then edit
   the security group so the only inbound rule is **PostgreSQL (5432) from My IP**. Never leave 0.0.0.0/0.
3. When the status is Available, copy the endpoint and run:

```
export DATABASE_URL="postgresql://trains:YOUR_PASSWORD@YOUR-ENDPOINT:5432/traindelays?sslmode=require"
python scripts/run_pipeline.py --skip-clean
```

4. **When you are not using it, stop or delete the instance.** A stopped RDS instance restarts by itself after 7 days.
   To avoid surprise bills, delete it once you have your screenshots, and keep `docker compose` as your demo.

### Why the live Streamlit Community Cloud app does not connect to RDS

Streamlit Community Cloud has no fixed IP address, so the only way to let it in would be to open the database to the
whole internet. Do not do that. Instead the live demo reads a snapshot:

```
DATABASE_URL="postgresql://..." python scripts/export_snapshot.py
git add database/delays.db database/build_info.json
git commit -m "Refresh data" && git push
```

PostgreSQL stays the source of truth; the snapshot is a copy for the public demo. In V4 the app moves to AWS
(App Runner or ECS) inside the same network as the database, and the snapshot step disappears.

## What changed from V1

| | V1 | V2 |
|---|---|---|
| Database | SQLite file committed to git | PostgreSQL (Docker locally, RDS on AWS), SQLite still works |
| Loading | replace tables one by one | one transaction: all or nothing |
| Bad data | could reach the app | 32 checks run first; errors block the load |
| Run it | several commands | `docker compose` or `python scripts/run_pipeline.py` |
| Build info | file only | also stored in the database |

## The data-quality checks

`python scripts/quality_checks.py` prints every check and writes `data/processed/quality_report.json`.
Errors block the load, warnings are reported. Examples: one row per operator and quarter; every percentage between
0 and 100; "within 1 minute" never above "within 3 minutes"; no missing quarters; delay minutes add up to the
total; no blank-header columns (the check that would have caught a bug introduced while building V2).

## Honest limits

- The Docker files were written and checked for syntax, but the PostgreSQL loader, quality gate and export were
  tested against a real PostgreSQL 16 server, not through Docker. Run step 1 yourself and tell me if anything fails.
- Refreshing is still triggered by you adding new files. ORR does not offer a stable download link that has been
  verified here, so a fully automatic schedule (EventBridge) is the next step, not part of this release.
- The Lambda function from V1 still writes cleaned CSVs to S3. Connecting it to RDS needs a VPC setup and is planned.
