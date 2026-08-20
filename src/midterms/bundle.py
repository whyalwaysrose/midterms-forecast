"""Build a single self-contained HTML file from the dashboard.

The hosted dashboard fetches its data at runtime. That is right for GitHub
Pages but useless for anything you want to hand someone directly, because
browsers block ``fetch`` from ``file://`` URLs. This inlines the CSS, the JS and
the three JSON payloads into one portable file that opens anywhere.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

DATA_FILES = ("forecast", "history", "commentary")


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
    js = (site_dir / "js" / "app.js").read_text(encoding="utf-8")

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

    html = html.replace(
        '<link rel="stylesheet" href="css/style.css">',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        '<script src="js/app.js"></script>',
        f"<script>\nwindow.__FORECAST_DATA__ = {payload};\n</script>\n<script>\n{js}\n</script>",
    )

    output.write_text(html, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    log.info("wrote %s (%.0f KB)", output, size_kb)
    return output
