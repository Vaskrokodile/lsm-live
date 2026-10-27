# LSM-30M · Ultron run report

## ultron_webmix_websight_webgen_30m_b4s512_eager

Byte-level latent-space language model (~33M parameters) trained on the
`webmix` HTML corpus blend (websight + webgen) with the **Ultron** optimizer.
Run resumed from step 18,583 (38.06M tokens) and **completed** at step 28,583.

## Outcome

| metric | value |
|---|---|
| final step | 28,583 / 28,583 |
| tokens seen | 58,537,984 (+20.48M this session) |
| best validation loss | **0.40745** @ step 28,583 |
| final train EMA | 0.4039 |
| throughput | ~9.1k tok/s (avg 9,085, peak 9,237) |
| checkpoint | `ultron_webmix_websight_webgen_30m_b4s512_eager_last.pt` |

## Optimizer

```
optimizer=ultron  lr=1e-4  betas=(0.9, 0.9841)
beta2_reference_tokens=None  factored_beta1=0.0
batch=4  accum=1  seq=512  amp=True  compile_loss=False  eval_batches=1
```

## Milestones

- val 0.5233 → 0.4967 in the first 1k steps after resume (warmup)
- steady monotone improvement: new best checkpoint written at nearly every
  200-step eval cadence
- val crossed **0.45** at step ~23,383 and **0.41** at step ~27,983
- best val 0.407451 recorded on the final eval (step 28,583)

## Notes

- Validation loss still decreasing at run end — no early-stop triggered.
- Throughput held ~8.6–9.2k tok/s on the RTX 3060; optimizer state ~47.5 MiB.
- Tokenizer: byte-level (no BPE); the model reads raw HTML bytes directly.
- Next step: extend target_step or resume the `_last.pt` checkpoint for a
  longer budget — the loss curve has not plateaued.
