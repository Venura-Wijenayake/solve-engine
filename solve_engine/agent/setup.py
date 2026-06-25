"""Create the LangGraph Postgres checkpointer tables (idempotent).

    python -m solve_engine.agent.setup

PostgresSaver.setup() creates its own tables (checkpoints, checkpoint_blobs,
checkpoint_writes, checkpoint_migrations) used to persist graph state. Safe to
re-run; the agent run also calls setup() on start.
"""

from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver

from solve_engine.config import get_settings


def main() -> None:
    settings = get_settings()
    with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        checkpointer.setup()
    print("checkpointer setup complete (tables created or already present).")


if __name__ == "__main__":
    main()
