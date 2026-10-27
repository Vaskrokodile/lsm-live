#!/usr/bin/env python3
"""Parse the LSM-30M Ultron training log into dashboard live/ data files.

Emits: live/history.jsonl, live/logtail.txt, live/status.json
Contract documented in live/README.md.
"""
import json
import os
import re
from datetime import datetime, timezone

LOG = r"E:\lsm30_out\lsm-run\ultron_webmix_30m_resume18583.log"
OUT = r"E:\lsm30_out\dashboard\live"

RUN = "ultron_webmix_websight_webgen_30m_b4s512_eager"
START_STEP = 18583
START_TOKENS = 38057984

step_re = re.compile(r"^step (\d+): train ([\d.]+) ema ([\d.]+)")
val_re = re.compile(r"val=\(([\d.eE+-]+)")
toks_re = re.compile(r"(\d+) tok/s")
best_re = re.compile(r"^new best validation total=([\d.]+) at step (\d+)")
ckpt_re = re.compile(r"^checkpoint step (\d+): (.+)$")
complete_re = re.compile(
    r"^complete: (\d+) tok/s, (\d+) new tokens, (\d+) total; val=\(([\d.eE+-]+)"
)

with open(LOG, "r", encoding="utf-16", errors="replace") as f:
    lines = f.read().splitlines()

os.makedirs(OUT, exist_ok=True)

# logtail: last ~200 lines
with open(os.path.join(OUT, "logtail.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines[-200:]) + "\n")

hist = []
last = {}
for ln in lines:
    ln = ln.strip()
    m = step_re.match(ln)
    if m:
        step, tr, ema = m.groups()
        rec = {"step": int(step), "train": float(tr), "ema": float(ema), "event": "log"}
        vm = val_re.search(ln)
        tm = toks_re.search(ln)
        if vm:
            rec["val"] = float(vm.group(1))
        if tm:
            rec["tok_s"] = int(tm.group(1))
        hist.append(rec)
        last = rec
        continue
    m = best_re.match(ln)
    if m:
        v, s = m.groups()
        hist.append({"step": int(s), "val": float(v), "event": "best"})
        continue
    m = ckpt_re.match(ln)
    if m:
        s, p = m.groups()
        hist.append({
            "step": int(s),
            "checkpoint": os.path.basename(p.strip()),
            "kind": "best" if "_best.pt" in p else "last",
            "event": "checkpoint",
        })
        continue
    m = complete_re.match(ln)
    if m:
        tps, new_tok, tot, v = m.groups()
        hist.append({
            "step": last.get("step"),
            "tok_s": int(tps),
            "tokens_seen": int(tot),
            "val": float(v),
            "event": "complete",
        })

deduped = []
for rec in hist:
    if deduped and deduped[-1] == rec:
        continue
    deduped.append(rec)
hist = deduped

with open(os.path.join(OUT, "history.jsonl"), "w", encoding="utf-8") as f:
    for rec in hist:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")

# status.json
bests = [r for r in hist if r.get("event") == "best"]
best = bests[-1] if bests else {}
done = any(r.get("event") == "complete" for r in hist)
mtime = datetime.fromtimestamp(os.path.getmtime(LOG), tz=timezone.utc)

status = {
    "updated_at": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "state": "done" if done else "training",
    "run": RUN,
    "step": last.get("step", 0),
    "target_step": 28583,
    "start_step": START_STEP,
    "start_tokens": START_TOKENS,
    "tokens_seen": 58537984,
    "train_loss": last.get("train"),
    "train_ema": last.get("ema"),
    "val_total": last.get("val"),
    "best_val": best.get("val"),
    "best_step": best.get("step"),
    "tok_per_s": last.get("tok_s", 9085),
    "avg_tok_per_s": 9085,
    "checkpoint": r"E:\lsm30_out\ultron_webmix_websight_webgen_30m_b4s512_eager_last.pt",
    "config": {
        "model": "LSM byte-level latent LM",
        "parameters": "~33M",
        "optimizer": "ultron",
        "lr": 0.0001,
        "betas": "(0.9, 0.9841)",
        "beta2_reference_tokens": "None",
        "factored_beta1": 0.0,
        "batch_size": 4,
        "grad_accum": 1,
        "seq_len": 512,
        "amp": True,
        "compile_loss": False,
        "reason": False,
        "eval_batches": 1,
        "dataset": "webmix (websight + webgen HTML corpora)",
        "hardware": "RTX 3060 (local, Windows)",
        "precision": "AMP (bf16/fp16)",
        "resumed_from": "step 18583 / 38,057,984 tokens",
    },
}
with open(os.path.join(OUT, "status.json"), "w", encoding="utf-8") as f:
    json.dump(status, f, indent=2)

print(f"history rows: {len(hist)}  bests: {len(bests)}  done: {done}")
print("status:", json.dumps({k: status[k] for k in ("state", "step", "best_val", "best_step")}))
