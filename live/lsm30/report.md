# LSM-30M — run report (continuous campaign)

Live numbers update from `status.json` / `history.jsonl` every ~30 s.
Metric note: `val lm` is the primary quality signal; `val total` includes
close/depth reasoning aux losses. The validation set changed at leg 7
(union of webmix v2 + v3, which adds high-entropy CommonPile HTML) —
pre-leg-7 val numbers are not comparable to later ones.

## Campaign history

| leg | type | corpus | steps | outcome |
|---|---|---|---|---|
| webmix v1 | sft | v1 (3 shards) | →28,583 | val lm 0.4075 |
| leg 3 | sft+reason | v2 | →30,783 | val lm 0.3807, stopped early |
| leg 4 | sft | v2 | →31,583 | partial, salvaged |
| leg 5 | sft | v2 | →46,583 | **v2-val lm 0.308** (8-batch eval) |
| leg 6 | **rl** | v2 | →46,883 | reward EMA 0.34→0.46, best 0.59; v2-val lm 0.392 (RL traded likelihood for structure) |
| leg 7 | sft | v3 | →49,883 | corpus switch; union val lm ~1.09 baseline |
| leg 8 | sft | v3 | →53,083 | early-stopped (patience); union val lm ~1.08 |
| leg 9 | **rl** | v3 | →53,383 | reward EMA ~0.47, best 0.60; struct 0.6–0.75 |
| leg 10+ | sft/rl | v3 | running | alternating; every 3rd leg is RL |

## RL leg design (rl_ultron.py)

REINFORCE on sampled HTML continuations, per-update K=4 samples from real
corpus prompts, advantage = normalized reward, +0.5 LM anchor on corpus
windows, Ultron optimizer (lr 5e-5). Reward = 0.30 tag-stack validity +
0.30 structural landmarks + 0.25 request adherence + 0.15 fluency,
×(1 − 0.5·junk-markers).

Observed so far: repetition nearly eliminated (flu →1.0), structural
landmark rate up (struct 0.4→0.7), tag-balance up early then traded for
structure. Weak axis: prompt adherence (~0.1) — keyword coverage barely
moves; likely needs longer training and/or a learned adherence signal.

## Checkpoints

Best checkpoints per leg sit next to `_last.pt` files in `E:\lsm30_out\`.
The chain always resumes from the newest `_last.pt` with optimizer state.
