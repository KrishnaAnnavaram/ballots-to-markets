# data/

Git does not track the files in this folder, except this README. Do not commit vendor price
files, downloaded texts or caches.

## Expected layout

```
data/
├── prices/
│   ├── sp500.csv    # Date,Close  (^GSPC)
│   ├── dow.csv      # Date,Close  (^DJI)
│   ├── gold.csv     # Date,Close  (GC=F)
│   ├── wti.csv      # Date,Close  (CL=F)
│   └── brent.csv    # Date,Close  (BZ=F)
├── fed_funds.csv    # date,value  (FRED series DFF)
└── texts.csv        # timestamp,tweet_text[,sentiment][,candidate]
```

## Sources

| File | Source | Terms | How to get it |
|---|---|---|---|
| `prices/*.csv` | Yahoo Finance through `yfinance` | Yahoo terms of use. Personal research use. Do not redistribute | `ballots-to-markets fetch` (extra `fetch`) |
| `fed_funds.csv` | FRED, Federal Reserve Bank of St. Louis, series `DFF` (<https://fred.stlouisfed.org/series/DFF>) | Public domain (Board of Governors data). Cite FRED | `ballots-to-markets fetch` with `FRED_API_KEY` set, or the CSV download on the FRED page |
| `texts.csv` | Your own documented text collection (news headlines or social posts) | The terms of the source platform | Write the CSV yourself |

`ballots-to-markets fetch` uses `B2M_START` and `B2M_END` (default 2023-11-01 to 2025-02-10) and
keeps a cached file unless you add `--refresh`.

## The 2024 election text sample of the prototype

The prototype used the Kaggle dataset "2024 U.S. Election Sentiment on X" (599 rows, columns
`tweet_id, user_handle, timestamp, tweet_text, candidate, party, retweets, likes, sentiment`).
Its handles are placeholders and its texts look synthetic. It covers about 120 days. You can load
it with `--texts train.csv val.csv test.csv`, but its coverage is below the 60% limit, so the
causality tests on text are refused. It can still validate the sentiment model on its labels.

## Event calendar

The built-in calendar `src/ballots_to_markets/data/events_2024.csv` lists the 2024 debates,
campaign events, election day and the FOMC decision dates (source: the Federal Reserve FOMC
calendar and the public news record). Use `--events my_events.csv` for another calendar.

## No download

`ballots-to-markets synth --out data/synthetic` writes synthetic files in this layout, and the
`--synthetic` flag makes the same data in memory. The tests use only synthetic data.
