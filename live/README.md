# live/ — dashboard data contract

The dashboard (`index.html`) fetches these files every 30 s, first from
`https://raw.githubusercontent.com/<OWNER>/<REPO>/main/live/...` (with a
cache-busting query param), falling back to same-origin `live/...` copies.
A publisher script on the training machine writes them and commits/pushes.

## Files

### `status.json`
Snapshot of the run. Example:

```json
{
  "updated_at": "2026-10-28T00:00:00Z",
  "state": "training",
  "run": "ultron_webmix_websight_webgen_30m_b4s512_eager",
  "step": 28583,
  "target_step": 28583,
  "tokens_seen": 58537984,
  "train_loss": 0.4299,
  "train_ema": 0.4039,
  "val_total": 0.40745,
  "best_val": 0.40745,
  "best_step": 28583,
  "tok_per_s": 9191,
  "checkpoint": "E:\\lsm30_out\\..._last.pt"
}
```

- `state` — `"training"` | `"idle"` | `"done"` (drives the status pill)
- `config` (optional object) — arbitrary key/value pairs rendered in the
  RUN CONFIG table
- extra keys are ignored; missing keys render as `—`

### `history.jsonl`
One JSON object per line, appended over time:

```json
{"step":28583,"ts":"...","train":0.43,"ema":0.404,"val":0.4075,"tok_s":9191,"event":"log"}
```

- `event` — `"log"` | `"best"` | `"checkpoint"` | `"complete"` | `"early-stop"`
- `log` rows drive the train / EMA / val / tok·s⁻¹ series
- `best` rows (with `val`) are drawn as diamond markers
- `checkpoint` rows may carry `checkpoint` (path/name) and `kind` (`"best"`|`"last"`) and feed the checkpoint list
- malformed lines are skipped by the parser

### `logtail.txt`
Last ~200 lines of the raw training log. Rendered verbatim in the LOG TAIL
panel (tail 200 lines, auto-scroll).

### `report.md`
Markdown run report — config, milestones, notes. Rendered client-side via
marked.js; falls back to a `<pre>` block if the CDN is unreachable.

## Publisher sketch

```python
# append to history.jsonl, overwrite status.json/logtail.txt/report.md,
# then:  git add live && git commit -m "update live metrics" && git push
```

See `tools/gen_data.py` for a working example that parses a training log.
