# Vendored pollster ratings

`pollster-ratings.csv` — FiveThirtyEight's pollster ratings: 517 polling
organisations scored on how accurate their polls turned out to be, relative to
what a poll of that type, timing and sample size should have achieved.

- **Source:** [fivethirtyeight/data](https://github.com/fivethirtyeight/data),
  `pollster-ratings/2023/pollster-ratings.csv`.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0) — the same license
  as `data/history/raw_polls.csv.gz`, and the reason both can be redistributed
  here at all.
- **Why vendored:** FiveThirtyEight was shut down in March 2025. The GitHub
  repository survives but is unmaintained, so fetching this at run time would
  make a daily forecast depend on something that may disappear without notice.

## What it is for

The model learns each pollster's **bias** from the data, as a house effect. It
cannot learn each pollster's **precision**: most pollsters have only a handful
of polls in one cycle, which is nowhere near enough to estimate a variance from.
Without an outside source, a 400-person poll from an F-rated shop and a
1,200-person poll from an A+ shop differ only by sample size — exactly the
mistake that trusting sample size alone invites.

`src/midterms/data/ratings.py` turns a rating into a multiplier on the model's
non-sampling noise term. See `polls.pollster_ratings` in `config/model.yaml`.

Columns used:

| Column | Use |
| --- | --- |
| `Pollster` | matched against our canonical pollster names |
| `Predictive Plus-Minus` | error relative to a comparable poll, in points; negative is better. Mean-reverted, so a shop with four lucky polls is not read as excellent |
| `Simple Expected Error` | the denominator that turns a plus-minus in points into a proportional multiplier |
| `Banned by 538` | pollsters 538 refused to accept polls from, mostly for suspected fabrication |
| `538 Grade` | shown in the dashboard; not used by the model |
| `Mean-Reverted Bias` | **not currently used** — the model estimates house effects itself |

## Two limitations worth knowing

**It stops in 2023.** 538 never published another ratings file before shutting
down, so pollsters that emerged since — Quantus Insights, Verasight, Focaldata,
The Bullfinch Group, Tavern Research, ActiVote among them — have no rating and
cannot get one. They are treated as exactly average, which is if anything
generous: a pollster with no track record is riskier than one with a good one.
Around 21% of polls in the current feed are unrated; each run logs the count.

**It rates the organisation, not the poll.** A pollster that has improved (or
degraded) since 2023 carries its old record. The house effect the model fits
from current data is the part that adapts; the quality multiplier is not.
