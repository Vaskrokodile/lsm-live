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
  "state": "done",
  "run": "addon_webmix_20261028_b_continue3",
  "step": 84608,
  "target_step": 84608,
  "tokens_seen": 283510784,
  "train_loss": 0.5138,
  "train_ema": 0.5712,
  "val_total": 0.637001,
  "val_total_source": "addon_webmix",
  "best_val": 0.634516,
  "best_step": 83858,
  "best_val_source": "addon_webmix",
  "tok_per_s": 18579.963,
  "checkpoint": "last.pt"
}
```

- `state` — `"training"` | `"idle"` | `"done"` (drives the status pill)
- `config` (optional object) — publisher emits only validated allowlisted
  training metadata to the RUN CONFIG table
- extra keys are ignored; missing keys render as `—`

### `history.jsonl`
One JSON object per line. Validation series are corpus-tagged (`val_v2`/`val_v3a`, `val_lm_v2`/`val_lm_v3a`, or `val_total_source`). Log-only legacy inputs use `val_legacy` and are visibly labeled as lacking corpus provenance.

```json
{"step":84608,"train":0.514,"ema":0.571,"val_total":0.637001,"val_total_source":"addon_webmix","val_lm":0.214586,"tok_s":18579.96,"event":"complete"}
```

- `event` — `"log"` | `"best"` | `"checkpoint"` | `"complete"` | `"early-stop"`
- `log` rows drive train / EMA / validation / throughput series; publisher-parsed
  legacy values use `val_legacy` because the old log has no corpus provenance
- `best` rows (with `val_v2`, `val_v3a`, or provenance-uncertain `val_legacy`) are drawn as diamond markers
- `checkpoint` rows may carry `checkpoint` (path/name) and `kind` (`"best"`|`"last"`) and feed the checkpoint list
- malformed lines are skipped by the parser

### `logtail.txt`
A compact tail of allowlisted metric events. Arbitrary raw log text is never
published; the dashboard displays this as metric history for diagnosis.

### `generation_eval.json`
Allowlisted held-out HTML-generation structural metrics and paired checkpoint
comparison. The publisher omits malformed rows and incomplete before/after pairs;
raw prompts, samples, and generated HTML are not published. This is separate from
validation loss and does not constitute rendered-page acceptance.

### `report.md`
Markdown run report — config, milestones, notes and dataset attribution. Rendered
client-side via marked.js; falls back to a `<pre>` block if the CDN is unreachable.

## Publisher sketch

```python
# Local-only dry runs use tools/publish_live.py --once with trainer telemetry.
# Public metrics are allowlisted; report.md is maintained and reviewed in-repo.
```

See `tools/gen_data.py` for a working example that parses a training log.
