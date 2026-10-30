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
| leg 9 | **rl** | v3 | →53,383 | reward EMA 0.509, best 0.608; struct 0.6–0.75, flu ~1.0 |
| leg 10 | sft | v3 | →57,983 | early-stopped; union val lm ~1.10; v2-val 0.828 (partial recovery) |
| leg 11 | sft | v3 @ seq 1024 | →~77k | full-document windows; union best total 1.378 |
| leg 12 | rl | v3 | →75k | reward EMA 0.544, best 0.660 |
| leg 15 | rl | v3 | →~81k | reward EMA 0.420, best 0.557 |
| leg 18 | rl | v3 | →~87k | reward EMA 0.537, best 0.664 |
| leg 21 | rl | v3 | →92,983 | **collapse**: ema →0.24, flu→0.2 (repetition drift) |
| rollback | — | — | 91,083 | rewound to leg-20 best; v2/union verified clean |
| leg 4–5 (r2) | sft | v3 @ 1024 | →96,083 | re-consolidated on rollback |
| leg 6 (r2) | **rl v2** | v3 | running | hardened RL (see below) |

## RL pipeline v2 (hardened after leg-21 collapse)

REINFORCE on sampled HTML continuations, K=4 samples from real corpus
prompts, normalized advantage, **KL penalty vs frozen reference model**
(β=0.02), +0.7 LM anchor, lr 3e-5, temp 0.85. **Collapse guard**: leg
aborts if reward EMA <0.22 after 50 updates. Reward =
0.28 validity (`html.parser` stack match, +bonus for `</html>` terminal)
+ 0.28 structure + 0.22 adherence + 0.22 fluency (16-byte ngrams **and**
line-level repetition), ×0.5 for corpus-marker junk, ×len/300 for
sub-300-byte outputs.

Observed pre-collapse: repetition nearly eliminated (flu →1.0), struct
0.4→0.7. Weak axis: prompt adherence (~0.1) — needs a learned signal or
longer runs.

## Infra notes

- E:\ is a Storage Spaces pool: thin-provisioning reported ~10GB "free"
  while physical slabs were exhausted → torch.save partial-write crashes.
  Fixed by deleting stale reasonlabel caches (~14GB).
- auto_leg retries a failed leg 3× then exits; legs resume from newest
  `*_last.pt` by mtime (rollback file is deliberately newest).

## Checkpoints

Best checkpoints per leg sit next to `_last.pt` files in `E:\lsm30_out\`.
The chain always resumes from the newest `_last.pt` with optimizer state.
