# Vendored historical polling

`raw_polls.csv.gz` — every poll FiveThirtyEight collected for its pollster ratings,
each paired with the **actual result** of the race it was polling. 20,466 polls,
1998–2022, of which 5,006 are Senate general elections.

- **Source:** [fivethirtyeight/data](https://github.com/fivethirtyeight/data),
  `pollster-ratings/raw_polls.csv`, gzipped here (4.8 MB -> ~1 MB).
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Why vendored:** FiveThirtyEight was shut down in March 2025 and its website
  is gone. The GitHub repository survives but is no longer maintained, so
  depending on it at run time would be depending on something that may vanish.

## What it is for

This is the only file in the project that can answer "how wrong are polls,
historically?" — which is what sets the width of the seat distribution, the
single most consequential number the model produces. Before it, the
`election_day_error` scales in `config/model.yaml` were asserted from the
literature. They are now fitted; see `src/midterms/calibration.py` and
`midterms calibrate`.

Key columns: `cycle`, `type_simple` (race class), `race_id`, `pollster`,
`polldate`, `electiondate`, `time_to_election`, `samplesize`, `margin_poll`,
`margin_actual`. Poll error is `margin_poll - margin_actual`, in points of
margin, signed toward the first-listed candidate.
