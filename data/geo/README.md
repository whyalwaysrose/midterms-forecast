# Vendored geography

`us-states-10m.json` — U.S. state boundaries at 1:10,000,000, TopoJSON, in
unprojected WGS84 longitude/latitude.

- **Source:** [us-atlas](https://github.com/topojson/us-atlas) v3, which derives
  the geometry from the U.S. Census Bureau's cartographic boundary files.
- **License:** ISC (us-atlas). The underlying Census boundary files are U.S.
  Government works and in the public domain.
- **Why vendored:** the daily CI run must not depend on a CDN being reachable,
  and pinning the file makes the rendered map reproducible.

It is converted to projected SVG paths at build time by
`src/midterms/geo.py` (`midterms build-map`), which writes
`site/data/us-states.json`. Nothing fetches this at runtime.
