"""Build a single self-contained HTML file from the dashboard.

The hosted dashboard fetches its data at runtime. That is right for GitHub
Pages but useless for anything you want to hand someone directly, because
browsers block ``fetch`` from ``file://`` URLs. This inlines the CSS, the JS and
the three JSON payloads into one portable file that opens anywhere.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

DATA_FILES = ("forecast", "history", "commentary", "us-states")

#: Scripts to inline, in load order. Must match the tags in site/index.html.
#:
#: Read out of the page rather than listed here. A hand-maintained list is a
#: standing invitation to add a script and forget: adding markets.js left the
#: bundle pointing at a file it had not inlined, and only the self-contained
#: assertion at the end caught it. Load order matters -- charts.js and map.js
#: call helpers defined in app.js -- and taking the order from the page is
#: exactly the guarantee that they agree.
_LOCAL_SCRIPT = re.compile(r"""<script\s+src=["']js/([A-Za-z0-9_.-]+\.js)["']""")


def local_scripts(html: str) -> tuple[str, ...]:
    """The page's own scripts, in the order it loads them."""
    found = tuple(_LOCAL_SCRIPT.findall(html))
    if not found:
        raise ValueError(
            "no local <script src=\"js/...\"> tags found in index.html; the "
            "markup changed shape and the bundler would silently inline nothing"
        )
    return found

#: Any <script> whose src points at another host, or at a protocol-relative
#: URL. Matched with DOTALL because the tag is wrapped across lines.
_EXTERNAL_SCRIPT = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*[\"']?(?:https?:)?//[^>]*>\s*</script\s*>",
    re.IGNORECASE | re.DOTALL,
)

#: The HTML comment that introduces the analytics tag, removed alongside it so
#: the bundle does not carry an explanation of something it does not do.
_ANALYTICS_COMMENT = re.compile(
    r"<!--\s*Visit counting.*?-->\s*", re.IGNORECASE | re.DOTALL
)


def strip_external_scripts(html: str) -> str:
    """Drop every script loaded from another host.

    Only the hosted page counts visits. The standalone bundle is a file people
    open from disk or are sent directly, and it must not report anything
    anywhere — both because that is the honest behaviour for a file handed over,
    and because the artifact viewer's CSP blocks the request anyway.
    """
    html = _ANALYTICS_COMMENT.sub("", html)
    return _EXTERNAL_SCRIPT.sub("", html)


def build(
    site_dir: Path | None = None,
    output: Path | None = None,
) -> Path:
    """Write a standalone HTML bundle and return its path."""
    site_dir = site_dir or paths.SITE_DIR
    output = output or paths.OUTPUTS_DIR / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    html = (site_dir / "index.html").read_text(encoding="utf-8")
    css = (site_dir / "css" / "style.css").read_text(encoding="utf-8")
    # Order comes from the page itself, so the two cannot disagree.
    scripts = local_scripts(html)
    js = "\n".join(
        (site_dir / "js" / name).read_text(encoding="utf-8") for name in scripts
    )

    data: dict[str, object] = {}
    for name in DATA_FILES:
        path = site_dir / "data" / f"{name}.json"
        if path.exists():
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            log.warning("no %s.json to embed", name)
            data[name] = None

    if data.get("forecast") is None:
        raise FileNotFoundError(
            f"{site_dir / 'data' / 'forecast.json'} not found — run `midterms run` first"
        )

    # `</script>` inside embedded JSON would close the tag early; escaping the
    # slash keeps the JSON identical to a parser while making it inert to the
    # HTML tokenizer.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    def substitute(source: str, marker: str, replacement: str, what: str) -> str:
        """Replace `marker`, failing loudly if it is not there.

        str.replace on a missing marker is a no-op and raises nothing, so a
        renamed tag silently produces a bundle that still points at external
        files — which then do not exist beside it. That shipped once already.
        """
        if marker not in source:
            raise ValueError(
                f"bundle: could not find the {what} marker in index.html. "
                f"Expected to substitute:\n{marker}"
            )
        return source.replace(marker, replacement, 1)

    html = substitute(
        html,
        '<link rel="stylesheet" href="css/style.css">',
        f"<style>\n{css}\n</style>",
        "stylesheet",
    )
    html = substitute(
        html,
        "\n".join(f'<script src="js/{name}"></script>' for name in scripts),
        f"<script>\nwindow.__FORECAST_DATA__ = {payload};\n</script>\n<script>\n{js}\n</script>",
        "script",
    )

    # Remove anything still loading from another host — in practice the
    # analytics tag. The hosted page counts visits; a file someone was handed,
    # or opened from disk, must not quietly report that back to anyone. The
    # artifact's content-security policy would block it regardless, so leaving
    # it in would only produce a console error and a broken promise.
    html = strip_external_scripts(html)

    # Nothing external may survive, or the bundle is not self-contained.
    for leftover in ('<script src=', '<link rel="stylesheet"'):
        if leftover in html:
            raise ValueError(f"bundle is not self-contained: found {leftover!r}")

    output.write_text(html, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    log.info("wrote %s (%.0f KB)", output, size_kb)
    return output
