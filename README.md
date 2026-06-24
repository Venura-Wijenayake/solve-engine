# Solve Engine

Solve Engine is a ticket-triage trainer built on top of live GitHub issues. It pulls real, open
issues from public repositories, scores them for solvability and difficulty, and presents them as a
workable ticket queue so you can practice triaging and solving real-world work instead of toy
problems. This repository currently contains the project skeleton; ingestion, scoring, and the
agent are built in later phases.

## Quickstart

```powershell
# Create and activate a virtual environment
uv venv
.venv\Scripts\Activate.ps1

# Install the package with dev tooling
uv pip install -e ".[dev]"

# Run the three quality gates
ruff check .
mypy .
pytest -q
```

## Project layout (target)

The structure below is the intended end state. Most directories do not exist yet — each is built in
its own phase. The phase tag (`[P#]`) marks when a part comes online.

```
solve_engine/
  config.py      # env secrets + config.toml tunables
  models.py      # Pydantic models: Issue, Score, Run, SolveLog
  db/            # connection helper + migration runner
  ingest/        # github_client, normalize, upsert      [P1]
  score/         # solvability + skill-fit scoring        [P2]
  classify/      # issue type + difficulty                [P2]
  eval/          # golden set + metrics harness           [P3]
  agent/         # LangGraph triage core                  [P4]
  api/           # FastAPI read layer                     [P5]
  board/         # Streamlit ticket queue UI              [P5]
config.toml
tests/
docs/            # architecture.md, adr/
```
