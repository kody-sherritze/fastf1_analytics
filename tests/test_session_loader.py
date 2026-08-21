from pathlib import Path
from types import SimpleNamespace

import fastf1

from fastf1_analytics import session_loader


def test_load_session_enables_cache_and_loads_session(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    loaded = SimpleNamespace()
    loaded.load = lambda: calls.append(("load", "session"))

    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: calls.append(("cache", path)))
    monkeypatch.setattr(fastf1, "get_session", lambda year, gp, code: loaded)

    result = session_loader.load_session(2024, "Monaco", "R", cache=str(tmp_path / "cache"))

    assert result is loaded
    assert calls == [("cache", str(tmp_path / "cache")), ("load", "session")]
    assert (tmp_path / "cache").is_dir()


def test_load_session_skips_cache_when_disabled(monkeypatch) -> None:
    cache_calls: list[str] = []
    loaded = SimpleNamespace(load=lambda: None)

    monkeypatch.setattr(fastf1.Cache, "enable_cache", cache_calls.append)
    monkeypatch.setattr(fastf1, "get_session", lambda year, gp, code: loaded)

    assert session_loader.load_session(2024, "Monaco", "R", cache=None) is loaded
    assert cache_calls == []
