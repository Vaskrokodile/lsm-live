#!/usr/bin/env python3
"""Watch LSM training log(s) and publish live metrics to the dashboard repo.

LSM-30M-specific publisher: parses train_ultron.py/rl_ultron.py logs into a
namespaced live dir (default live/lsm30) so multiple runs can publish to the
same repo without colliding. Commits and pushes on change; the site reads
these files via raw.githubusercontent.com, so no redeploy is needed.

Usage:
    python publish_lsm30.py --repo E:\lsm30_out\dashboard \
        --run <run_name> --start-step 0 --start-tokens 0 \
        --target-step 200000 --log old.log --logs-glob "logs\\leg*.log" \
        --config leg_config.json --interval 60
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

STEP_RE = re.compile(r"^step (\d+): train ([\d.]+) ema ([\d.]+)")
VAL_RE = re.compile(
    r"val=\(([\d.eE+-]+|nan)(?:,\s*([\d.eE+-]+|nan))?"
    r"(?:,\s*([\d.eE+-]+|nan))?(?:,\s*([\d.eE+-]+|nan))?\)")
TOKS_RE = re.compile(r"(\d+) tok/s")
BEST_RE = re.compile(r"^new best validation total=([\d.]+) at step (\d+)")
CKPT_RE = re.compile(r"^checkpoint step (\d+): (.+)$")
COMPLETE_RE = re.compile(
    r"^complete: (\d+) tok/s, (\d+) new tokens, (\d+) total; val=\(([\d.eE+-]+)")
RL_COMPLETE_RE = re.compile(r"^rl complete: (\d+) updates")
_TAG_RE = re.compile(rb"</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>")
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
         "link", "meta", "param", "source", "track", "wbr"}


def sample_metrics(text):
    """Rough generation-quality metrics for the HTML eval widget."""
    raw = text.encode("utf-8", errors="replace")
    lower = text.lower()
    stack, closes, matched = [], 0, 0
    for m in _TAG_RE.finditer(raw):
        name = m.group(1).decode("ascii", "replace").lower()
        if m.group(0).startswith(b"</"):
            closes += 1
            if stack and stack[-1] == name:
                stack.pop()
                matched += 1
            elif name in stack:
                while stack and stack.pop() != name:
                    pass
                matched += 1
        elif name not in _VOID and not m.group(0).endswith(b"/>"):
            stack.append(name)
    return {
        "balance": round(matched / closes, 4) if closes else None,
        "balanced": int(closes > 0 and matched == closes and not stack),
        "doctype": int("<!doctype" in lower),
        "html_pair": int("<html" in lower and "</html>" in lower),
        "body_pair": int("<body" in lower and "</body>" in lower),
    }


def extract_generated(text):
    i = text.find("===== END WEBSITE =====")
    if i > 0:
        text = text[:i]
    j = text.rfind("ASSISTANT PROJECT FILES:")
    return text[j + len("ASSISTANT PROJECT FILES:"):] if j >= 0 else text
EARLY_RE = re.compile(r"^early stopping at step (\d+); best step=(\d+) "
                      r"val total=([\d.]+)")
RL_RE = re.compile(
    r"^step (\d+): upd (\d+): reward ([\d.]+) ema ([\d.]+) best ([\d.]+); "
    r"struct/valid/adhere/flu=\(([\d.eE+-]+),\s*([\d.eE+-]+),"
    r"\s*([\d.eE+-]+),\s*([\d.eE+-]+)\);\s*logp ([\d.eE+-]+|nan)")


def read_lines(path):
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read().splitlines()
        except (UnicodeError, ValueError):
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace").splitlines()


def num(value):
    if value is None or value == "nan":
        return None
    return float(value)


def parse_logs(paths, start_step):
    hist = []
    last = {}
    finished = False
    for path in paths:
        finished = False
        for raw in read_lines(path):
            ln = raw.strip()
            m = STEP_RE.match(ln)
            if m:
                step, tr, ema = m.groups()
                rec = {"step": int(step), "train": float(tr),
                       "ema": float(ema), "event": "log"}
                vm = VAL_RE.search(ln)
                tm = TOKS_RE.search(ln)
                if vm:
                    rec["val"] = num(vm.group(1))
                    rec["val_legacy"] = rec["val"]
                    rec["val_total"] = rec["val"]
                    for key, group in (("val_lm", 2), ("val_close", 3),
                                       ("val_depth", 4)):
                        v = num(vm.group(group))
                        if v is not None:
                            rec[key] = v
                    if rec.get("val_lm") is not None:
                        rec["val_lm_v2"] = rec["val_lm"]
                if tm:
                    rec["tok_s"] = int(tm.group(1))
                hist.append(rec)
                last = rec
                continue
            m = RL_RE.match(ln)
            if m:
                (step, upd, rew, ema, best, s1, s2, s3, s4, lp) = m.groups()
                rec = {"step": int(step),
                       "reward": float(rew), "ema": float(ema),
                       "best_reward": float(best),
                       "event": "log", "leg": "rl"}
                tm = TOKS_RE.search(ln)
                if tm:
                    rec["tok_s"] = int(tm.group(1))
                hist.append(rec)
                last = rec
                continue
            m = BEST_RE.match(ln)
            if m:
                v, s = m.groups()
                hist.append({"step": int(s), "val": float(v),
                             "event": "best"})
                continue
            m = CKPT_RE.match(ln)
            if m:
                s, p = m.groups()
                hist.append({
                    "step": int(s),
                    "checkpoint": os.path.basename(p.strip()),
                    "kind": "best" if "_best.pt" in p else "last",
                    "event": "checkpoint"})
                continue
            m = EARLY_RE.match(ln)
            if m:
                s, bs, bv = m.groups()
                hist.append({"step": int(s), "best_step": int(bs),
                             "val": float(bv), "event": "early-stop"})
                finished = True
                continue
            if RL_COMPLETE_RE.match(ln):
                finished = True
                continue
            m = COMPLETE_RE.match(ln)
            if m:
                tps, new_tok, tot, v = m.groups()
                hist.append({
                    "step": last.get("step"), "tok_s": int(tps),
                    "tokens_seen": int(tot), "val": float(v),
                    "event": "complete"})
                finished = True
    deduped = []
    for rec in hist:
        if deduped and deduped[-1] == rec:
            continue
        deduped.append(rec)
    return deduped, last, finished


def git(repo, *argv):
    return subprocess.run(["git", "-C", repo, *argv],
                          capture_output=True, text=True, timeout=120)


def publish(repo, payload, out_subdir):
    status_path = os.path.join(repo, ".publish_lsm30_state.json")
    previous = None
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                previous = f.read()
        except OSError:
            pass
    if previous == payload:
        return
    with open(status_path, "w", encoding="utf-8") as f:
        f.write(payload)
    git(repo, "add", out_subdir, ".publish_lsm30_state.json")
    diff = git(repo, "diff", "--cached", "--quiet")
    if diff.returncode == 0:
        return
    state = json.loads(payload)["status"]
    msg = f"lsm30: {state['run']} step {state.get('step')} ({state['state']})"
    git(repo, "commit", "-m", msg)
    push = git(repo, "push")
    if push.returncode != 0:
        git(repo, "pull", "--rebase", "--autostash")
        push = git(repo, "push")
    if push.returncode != 0:
        print(f"[publish] push failed: {push.stderr.strip()}", flush=True)
    else:
        print(f"[publish] pushed: {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out-subdir", default="live/lsm30")
    p.add_argument("--run", required=True)
    p.add_argument("--log", action="append", default=[])
    p.add_argument("--logs-glob", action="append", default=[])
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--start-tokens", type=int, default=0)
    p.add_argument("--target-step", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--samples-glob", default=None,
                   help="glob for generated sample dumps; newest file "
                        "feeds the generation_eval widget")
    p.add_argument("--interval", type=float, default=60.0)
    args = p.parse_args()

    out = os.path.join(args.repo, args.out_subdir.replace("/", os.sep))
    os.makedirs(out, exist_ok=True)
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    print(f"[publish] lsm30 -> {args.out_subdir}, repo={args.repo}, "
          f"interval={args.interval}s", flush=True)
    while True:
        log_paths = list(args.log)
        for pat in args.logs_glob:
            log_paths += glob.glob(pat)
        seen = set()
        log_paths = [x for x in log_paths
                     if not (os.path.abspath(x) in seen
                             or seen.add(os.path.abspath(x)))]
        log_paths.sort(key=lambda x: os.path.getmtime(x)
                       if os.path.exists(x) else 0)
        if not log_paths:
            print("[publish] no logs matched", flush=True)
            time.sleep(args.interval)
            continue
        try:
            hist, last, finished = parse_logs(log_paths, args.start_step)
        except OSError as exc:
            print(f"[publish] log read error: {exc}", flush=True)
            time.sleep(args.interval)
            continue

        last_log = log_paths[-1]
        try:
            tail = read_lines(last_log)[-200:]
        except OSError:
            tail = []
        with open(os.path.join(out, "logtail.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(tail) + "\n")

        with open(os.path.join(out, "history.jsonl"), "w",
                  encoding="utf-8") as f:
            for rec in hist:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

        bests = [r for r in hist if r.get("event") == "best"]
        best = bests[-1] if bests else {}
        complete = [r for r in hist if r.get("event") == "complete"]
        if finished and complete:
            tokens = complete[-1].get("tokens_seen")
        else:
            tokens = args.start_tokens + \
                (last.get("step", args.start_step) - args.start_step) * \
                (cfg.get("batch_size", 1) * cfg.get("seq_len", 1) *
                 cfg.get("grad_accum", 1))
        try:
            mtime = datetime.fromtimestamp(
                os.path.getmtime(last_log), tz=timezone.utc)
        except OSError:
            mtime = datetime.now(timezone.utc)
        status = {
            "updated_at": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "state": "done" if finished else "training",
            "run": args.run,
            "step": last.get("step", args.start_step),
            "target_step": args.target_step,
            "start_step": args.start_step,
            "start_tokens": args.start_tokens,
            "tokens_seen": tokens,
            "train_loss": last.get("train"),
            "train_ema": last.get("ema"),
            "reward": last.get("reward"),
            "val_total": last.get("val"),
            "val_lm": last.get("val_lm"),
            "val_close": last.get("val_close"),
            "val_depth": last.get("val_depth"),
            "best_val": best.get("val"),
            "best_step": best.get("step"),
            "tok_per_s": last.get("tok_s"),
            "checkpoint": args.checkpoint,
            "config": cfg,
        }
        if args.samples_glob:
            samps = [x for x in glob.glob(args.samples_glob)
                     if os.path.exists(x)
                     and os.path.getsize(x) > 512]
            if samps:
                newest = max(samps, key=os.path.getmtime)
                try:
                    gen = extract_generated(
                        "\n".join(read_lines(newest)))
                    m = sample_metrics(gen)
                    status["generation_eval"] = {
                        **m, "samples": 1,
                        "step": status["step"],
                        "source": os.path.basename(newest),
                        "settings": {"temperature": 0.6, "top_k": 50,
                                     "max_new_bytes": 4096}}
                except OSError:
                    pass
        with open(os.path.join(out, "status.json"), "w",
                  encoding="utf-8") as f:
            json.dump(status, f, indent=2)

        payload = json.dumps(
            {"status": status, "history_rows": len(hist),
             "tail": tail[-50:]}, sort_keys=True)
        try:
            publish(args.repo, payload, args.out_subdir)
        except Exception as exc:
            print(f"[publish] error: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
