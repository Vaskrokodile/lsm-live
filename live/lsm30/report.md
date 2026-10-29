# LSM-30M — run report (continuous training)

This report is maintained manually between runs; live numbers on the page
update from `status.json` / `history.jsonl` every ~30 s.

## Latest leg — leg 5 (SFT, reasoning head on)
- Resumed from leg 4 checkpoint at step 31,583 (~64.7M tokens)
- Objective: `lm + 0.15*(close + depth)` on webmix v2 corpus
- Val components logged as `(total, lm, close, depth)`

## Campaign history

| leg | ckpt chain | steps | val total | val lm | note |
|---|---|---|---|---|---|
| webmix v1 | `ultron_webmix_websight_webgen_30m` | →28,583 | 0.4075 (lm) | 0.4075 | 3 WebSight shards + WebGen, no reasoning head |
| leg 3 reason | `ultron_webmixv2_reason_30m` | →30,783 | 0.6639 | 0.3807 | corpus v2 (15 shards + WebGen + Stack v2 HTML), reasoning head on, stopped early |
| leg 4 | `ultron_reason_leg4_sft` | →31,583 | ~0.68 | ~0.37 | partial (driver restart), salvaged |
| leg 5+ | `ultron_reason_legN_{sft,rl}` | running | — | — | continuous auto-leg loop; every 3rd leg = RL reward leg |

## RL legs (rl_ultron.py)

REINFORCE-style updates on sampled HTML continuations. Reward =
0.30 tag-stack validity + 0.30 structural landmarks + 0.25 request
adherence + 0.15 fluency (low n-gram repetition), ×(1 - 0.5·junk).
Anchor LM loss on corpus windows keeps the model grounded.

- **LM loss** on val stays the primary apples-to-apples metric.
- Sample dumps land in `E:\lsm30_out\samples\legN.txt` per leg.
