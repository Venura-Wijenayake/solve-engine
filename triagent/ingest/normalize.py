"""Normalize raw GitHub issue dicts into validated Issue models.

Raw items come from the assembly step (PR-filtered, deduped, provenance-tagged
under ``_source``). Mapping is light: derive identity, pull the obvious fields,
and retain the full raw payload so heavier cleaning can happen later at scoring
time. Records that can't be made useful are dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from triagent.ingest.query import SOURCE_KEY, issue_key
from triagent.models import Issue


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_issue(
    raw: dict[str, Any],
    *,
    language: str | None = None,
    now: datetime | None = None,
) -> Issue | None:
    """Map one raw issue dict to an Issue, or None if it isn't worth storing.

    ``language`` is applied to search-sourced items (GitHub issue objects carry
    no language); watchlist-sourced items get None.
    """
    seen_at = now or _utcnow()

    number = raw.get("number")
    repo_url = raw.get("repository_url")
    title = (raw.get("title") or "").strip()
    html_url = raw.get("html_url")
    created_at = raw.get("created_at")
    updated_at = raw.get("updated_at")

    # Hard requirements for a storable record.
    if not number or not repo_url or not title or not html_url:
        return None
    if not created_at or not updated_at:
        return None

    labels = [
        label["name"]
        for label in raw.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    ]

    body_raw = raw.get("body")
    body = (body_raw.strip() or None) if isinstance(body_raw, str) else None

    # Drop signal-free records: no body text and no labels to triage on.
    if not body and not labels:
        return None

    source = raw.get(SOURCE_KEY, "search")
    lang = language if source == "search" else None

    repo = repo_url.split("/repos/", 1)[-1]
    state = raw.get("state", "open")

    return Issue(
        repo=repo,
        number=number,
        title=title,
        body=body,
        html_url=html_url,
        state=state,
        labels=labels,
        language=lang,
        created_at=created_at,
        updated_at=updated_at,
        source=source,
        first_seen=seen_at,
        last_seen=seen_at,
        raw=raw,
    )


def normalize_issues(
    raws: list[dict[str, Any]],
    *,
    language: str | None = None,
    now: datetime | None = None,
) -> list[Issue]:
    """Normalize a batch, dropping unusable records. Dedupe defensively by key."""
    seen_at = now or _utcnow()
    out: dict[str, Issue] = {}
    for raw in raws:
        issue = normalize_issue(raw, language=language, now=seen_at)
        if issue is not None:
            out[issue_key(raw)] = issue
    return list(out.values())
