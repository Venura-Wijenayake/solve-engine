import pytest

import solve_engine.config as config


def test_get_settings_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    config.get_settings.cache_clear()

    settings = config.get_settings()

    assert settings.watchlist, "watchlist should be non-empty"
    assert "python" in settings.language_focus
    assert settings.data_tags
    assert 0.0 <= settings.thresholds.solvability_min <= 1.0
    assert 0.0 <= settings.thresholds.skill_fit_min <= 1.0

    config.get_settings.cache_clear()


def test_missing_database_url_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stop .env from supplying the value so the failure path is exercised.
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config.get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        config.get_settings()

    config.get_settings.cache_clear()
