"""Ingestion: pull live, deduped, PR-filtered issues off GitHub.

This package fetches raw issue dicts only. Mapping into the Issue model and
writing to Postgres happens in a later card.
"""
