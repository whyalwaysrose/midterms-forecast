"""Content-hashed asset URLs, which stop a cached script meeting fresh data.

GitHub Pages serves everything with a fixed ten-minute max-age, so without this
a returning visitor could hold last deploy's app.js while fetching today's
forecast.json and trip the schema check on a healthy site.
"""

from __future__ import annotations

import re
import shutil

import pytest

from midterms import assets, paths

REF = re.compile(r'(?:src|href)="((?:js|css)/[^"]+)"')


@pytest.fixture
def site(tmp_path):
    """A throwaway copy of the real site, so tests never touch the repo's."""
    target = tmp_path / "site"
    shutil.copytree(paths.SITE_DIR, target)
    return target


def test_every_asset_reference_gets_a_hash(site):
    applied = assets.stamp(site)
    assert applied, "nothing was stamped"

    html = (site / "index.html").read_text(encoding="utf-8")
    for ref in REF.findall(html):
        assert "?v=" in ref, f"{ref} was left unstamped"
        path, _, version = ref.partition("?v=")
        assert (site / path).is_file()
        assert re.fullmatch(r"[0-9a-f]{8}", version), version


def test_the_hash_follows_the_contents(site):
    before = assets.stamp(site)
    (site / "js" / "app.js").write_text("// changed\n", encoding="utf-8")
    after = assets.stamp(site)

    assert after["js/app.js"] != before["js/app.js"], "edited file kept its hash"
    assert after["js/map.js"] == before["js/map.js"], (
        "an untouched file changed hash, which would needlessly bust its cache"
    )


def test_stamping_twice_is_stable(site):
    """Re-running must replace the version, not append a second one."""
    first = assets.stamp(site)
    html_once = (site / "index.html").read_text(encoding="utf-8")
    second = assets.stamp(site)
    html_twice = (site / "index.html").read_text(encoding="utf-8")

    assert first == second
    assert html_once == html_twice
    assert "?v=" in html_twice
    assert re.search(r"\?v=[0-9a-f]+\?v=", html_twice) is None


def test_line_endings_do_not_change_the_hash(site, tmp_path):
    """The runner is Linux and the working tree is Windows.

    If CRLF and LF hashed differently, every deploy would bust every cache and
    the whole exercise would be pointless.
    """
    app = site / "js" / "app.js"
    app.write_bytes(b"const a = 1;\nconst b = 2;\n")
    lf = assets.content_hash(app)
    app.write_bytes(b"const a = 1;\r\nconst b = 2;\r\n")
    crlf = assets.content_hash(app)
    assert lf == crlf


def test_a_missing_asset_is_an_error_not_a_silent_skip(site):
    (site / "js" / "map.js").unlink()
    with pytest.raises(FileNotFoundError, match="map.js"):
        assets.stamp(site)


def test_markup_with_no_assets_is_an_error(site):
    """A silent no-op here would ship an unstamped page and look successful."""
    (site / "index.html").write_text("<p>nothing here</p>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no js/ or css/ asset references"):
        assets.stamp(site)


def test_the_committed_page_is_not_stamped():
    """Stamping happens on the deploy artifact, so local dev stays plain."""
    html = (paths.SITE_DIR / "index.html").read_text(encoding="utf-8")
    assert "?v=" not in html, (
        "site/index.html has stamped URLs committed; stamping is a deploy step "
        "and committing it would put a stale hash in the repo"
    )
