"""Typed configuration: secrets from the environment, tunables from config.toml.

Secrets (``GITHUB_TOKEN``, ``DATABASE_URL``) come from the process environment,
loaded from a git-ignored ``.env`` in development. Non-secret tunables (watchlist,
language focus, score thresholds) come from a committed ``config.toml`` at the
repo root. ``get_settings()`` is the single accessor and is cached.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

# config.toml lives at the repo root, one level above this package.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class Thresholds(BaseModel):
    """Score cutoffs that control what reaches the workable queue."""

    solvability_min: float
    skill_fit_min: float


class Settings(BaseModel):
    """Resolved configuration for the whole engine."""

    # Secrets (from the environment).
    github_token: str | None
    database_url: str

    # Tunables (from config.toml).
    language_focus: list[str]
    data_tags: list[str]
    watchlist: list[str]
    thresholds: Thresholds


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings. Cached; call ``cache_clear()`` to reload.

    Raises:
        RuntimeError: if a required secret (``DATABASE_URL``) is missing.
    """
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and set it to your "
            "Neon connection string."
        )

    github_token = os.environ.get("GITHUB_TOKEN") or None

    data = _load_toml(CONFIG_PATH)
    return Settings(
        github_token=github_token,
        database_url=database_url,
        language_focus=list(data.get("language_focus", [])),
        data_tags=list(data.get("data_tags", [])),
        watchlist=list(data.get("watchlist", [])),
        thresholds=Thresholds(**data.get("thresholds", {})),
    )
