"""API key handling for the optional data sources.

Keys are credentials. The properties worth pinning are that they are read from
somewhere gitignored, that a missing one produces an instruction rather than a
stack trace, and that nothing here ever prints a whole key.
"""

from __future__ import annotations

import pytest

from midterms import keys


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    """No real keys and no real .env, so these never depend on the machine."""
    for name in keys.SIGNUP_URLS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_dotenv_path", lambda: tmp_path / ".env")


def test_an_unset_key_is_none():
    assert keys.get("CENSUS_API_KEY") is None


def test_the_environment_supplies_a_key(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "abc123")
    assert keys.get("CENSUS_API_KEY") == "abc123"


def test_a_dotenv_file_supplies_a_key(tmp_path):
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "\n"
        "CENSUS_API_KEY=from-file\n"
        'FEC_API_KEY="quoted-value"\n',
        encoding="utf-8",
    )
    assert keys.get("CENSUS_API_KEY") == "from-file"
    assert keys.get("FEC_API_KEY") == "quoted-value"


def test_the_environment_beats_the_file(monkeypatch, tmp_path):
    """CI injects a secret; it must not be shadowed by a stale local file."""
    (tmp_path / ".env").write_text("CENSUS_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("CENSUS_API_KEY", "from-env")
    assert keys.get("CENSUS_API_KEY") == "from-env"


def test_surrounding_whitespace_is_stripped(tmp_path):
    """A trailing space pasted from a browser must not break the key."""
    (tmp_path / ".env").write_text("CENSUS_API_KEY=  spaced  \n", encoding="utf-8")
    assert keys.get("CENSUS_API_KEY") == "spaced"


def test_an_empty_value_reads_as_unset(tmp_path):
    (tmp_path / ".env").write_text("CENSUS_API_KEY=\n", encoding="utf-8")
    assert keys.get("CENSUS_API_KEY") is None


def test_require_explains_how_to_get_the_key():
    with pytest.raises(keys.MissingKey) as caught:
        keys.require("CENSUS_API_KEY")
    message = str(caught.value)
    assert keys.SIGNUP_URLS["CENSUS_API_KEY"] in message
    assert ".env" in message
    assert "check-keys" in message


def test_require_returns_the_key_when_set(monkeypatch):
    monkeypatch.setenv("FEC_API_KEY", "live-key")
    assert keys.require("FEC_API_KEY") == "live-key"


def test_a_key_is_never_shown_in_full():
    """Status output is meant to be pasteable into a chat window."""
    secret = "abcdefghijklmnop"
    redacted = keys._redact(secret)
    assert secret not in redacted
    assert len(redacted) < len(secret)


def test_status_reports_absent_keys_without_touching_the_network():
    results = keys.status(check_network=False)
    assert {r.name for r in results} == set(keys.SIGNUP_URLS)
    assert all(not r.present for r in results)
    assert all(r.working is None for r in results)
    for entry in results:
        assert keys.SIGNUP_URLS[entry.name] in entry.detail


def test_status_does_not_leak_a_present_key(monkeypatch):
    monkeypatch.setenv("FEC_API_KEY", "supersecretvalue")
    results = {r.name: r for r in keys.status(check_network=False)}
    assert results["FEC_API_KEY"].present
    assert "supersecretvalue" not in results["FEC_API_KEY"].detail


def test_every_key_has_a_signup_url_and_a_checker():
    """A key nobody can obtain, or cannot verify, is not much use."""
    assert set(keys.CHECKS) == set(keys.SIGNUP_URLS)
    for url in keys.SIGNUP_URLS.values():
        assert url.startswith("https://")


def test_dotenv_is_gitignored():
    """The whole design assumes this file never reaches the repository."""
    from midterms.paths import REPO_ROOT

    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert any(line.strip() == ".env" for line in ignored.splitlines()), (
        ".env must be gitignored; it is where API keys live"
    )
