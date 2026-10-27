# lsm-live

Live training dashboard for **LSM-30M**, a ~33M-parameter byte-level
latent-space language model trained on HTML corpora with the **Ultron**
optimizer (RTX 3060, local Windows box).

Static site — no build step. Open `index.html` or deploy the directory as-is
(Vercel: framework = *Other*).

## What it renders

- run status pill (`training` / `idle` / `done`), step progress, tokens seen,
  tok/s, train/EMA/val losses, best-val callout
- loss curve chart (train raw, train EMA, val; best-event diamonds,
  checkpoint ticks) + throughput chart (ECharts)
- run config table, checkpoint list
- run report rendered from `live/report.md` (marked.js)
- raw log tail viewer (`live/logtail.txt`, last ~200 lines, auto-scroll)
- auto-refresh every 30 s

## Data flow

The training box publishes JSON into `live/` and pushes to `main`. The site
fetches `live/status.json`, `live/history.jsonl`, `live/logtail.txt` and
`live/report.md` from `raw.githubusercontent.com` (cache-busted), falling
back to same-origin copies. The contract is documented in
[`live/README.md`](live/README.md); `tools/gen_data.py` generates the files
from a training log.
