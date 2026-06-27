# Triagent — session pickup

## Where things stand (5 phases done)
- P0 Foundation, P1 Ingestion, P2 Scoring, P3 Evaluation, P4 Agent — all merged to main.
- DB (Neon Postgres) holds ~648 issues; 23 scored at prompt_version "v1".
- Eval: 23-row human golden set committed; harness compares model vs human (type 83%, difficulty/solvability ~65%, model biased toward over-rating tractability).
- Agent: LangGraph Triage→Verify graph with a real Postgres checkpointer (checkpoints* tables exist). Writes scores at prompt_version "agent-v1".

## First moves next session
1. Re-run live scoring once Gemini daily quota has reset:
   - `python -m triagent.score.run --sleep 4`        (fills the v1 backlog, ~625 unscored)
   - `python -m triagent.agent.run --limit 5`        (live agent-v1 scores; was quota-blocked last session)
2. Then start P5 — Serve & Ship: FastAPI read layer + a real ticket-queue UI + a managed deploy (live URL, not localhost), with the Oracle Cloud migration as the hardening step.

## Useful commands
- Ingest more issues:   `python -m triagent.ingest.run`
- See the board:        `streamlit run triagent/board/app.py`
- Run the eval:         `python -m triagent.eval.run_eval`
- Agent setup (once):   `python -m triagent.agent.setup`

## Notes / known things
- Gemini free tier has a tiny daily quota — scoring runs in bursts across days; both runners stop gracefully and resume.
- uv is on PATH only after prepending its Scripts dir (see prior session memory).
- Keep .env CLOSED in the editor (DB creds). It's gitignored.
