"""Pin the dashboard's script and stylesheet URLs to their contents.

GitHub Pages serves everything with ``max-age=600`` and no way to configure it,
so for up to ten minutes after a deploy a returning visitor can hold a cached
``app.js`` while fetching a fresh ``forecast.json``. The schema check then fires
and the page shows an error, on a site that is in fact perfectly healthy.

Adding a content hash to each asset URL removes the failure rather than
recovering from it: new bytes mean a new URL, so the browser cannot pair last
week's JavaScript with today's data. Unchanged files keep their hash and stay
cached, which is the behaviour we want.

This runs against the deploy artifact, not the working tree, so the committed
``index.html`` stays clean and local development is unaffected.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# src="js/app.js" and href="css/style.css", with or without an existing ?v=.
ASSET_REF = re.compile(
    r'(?P<attr>\b(?:src|href)=")(?P<path>(?:js|css)/[A-Za-z0-9_.-]+\.(?:js|css))'
    r'(?:\?v=[0-9a-f]+)?(?P<close>")'
)


def content_hash(path: Path) -> str:
    """First 8 hex characters of the file's SHA-256.

    Newlines are normalised first. Without that the same file checked out on
    Windows and on the Linux runner hashes differently, which would bust every
    cache on every deploy and defeat the point.
    """
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()[:8]


def stamp(site_dir: Path) -> dict[str, str]:
    """Rewrite index.html's asset URLs to carry a content hash.

    Returns the mapping that was applied, so a caller can log or assert on it.
    Raises if a referenced asset is missing: a silent no-op here would deploy a
    page whose scripts 404, and the whole point of this module is to stop the
    deployed page and its data drifting apart.
    """
    index = site_dir / "index.html"
    html = index.read_text(encoding="utf-8")
    applied: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        relative = match.group("path")
        asset = site_dir / relative
        if not asset.is_file():
            raise FileNotFoundError(
                f"{index} references {relative}, which does not exist in {site_dir}"
            )
        digest = content_hash(asset)
        applied[relative] = digest
        return f"{match['attr']}{relative}?v={digest}{match['close']}"

    stamped = ASSET_REF.sub(replace, html)
    if not applied:
        raise RuntimeError(
            f"{index} has no js/ or css/ asset references to stamp — the markup "
            "changed shape and this step is now silently doing nothing"
        )

    index.write_text(stamped, encoding="utf-8", newline="")
    log.info("stamped %d assets in %s", len(applied), index.name)
    for relative, digest in sorted(applied.items()):
        log.info("  %s?v=%s", relative, digest)
    return applied
