"""Hierarchical Bayesian forecasting for U.S. midterm elections.

The package is organised as a one-way pipeline:

    data/       fetch and normalise polls from the VoteHub API
    fundamentals.py
                turn structural race facts into a prior on each race baseline
    model/      the PyMC hierarchical dynamic model, plus the simulation layer
                that turns its posterior into a chamber seat distribution
    outputs.py  serialise a forecast run to JSON for the dashboard
    commentary.py
                diff consecutive runs into a human-readable changelog
    bundle.py   inline the dashboard into one shareable HTML file
    backtest.py calibration diagnostics

Nothing above `model/` knows that the chamber is the Senate: races are supplied
by config, so adding the House or the governorships is a config change plus a
control-arithmetic rule, not a rewrite.
"""

__version__ = "0.1.0"
