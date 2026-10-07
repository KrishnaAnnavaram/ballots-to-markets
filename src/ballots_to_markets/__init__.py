"""ballots-to-markets: event study, causality tests and walk-forward forecasts on a trading-day index."""

__version__ = "0.1.0"

# Short asset name -> Yahoo Finance ticker.
ASSETS = {"sp500": "^GSPC", "dow": "^DJI", "gold": "GC=F", "wti": "CL=F", "brent": "BZ=F"}
MARKET = "sp500"
