#!/usr/bin/env python3
"""Publish allowlisted LSM run metrics to the public lsm-live repository.

The dashboard is public. This publisher strips machine-local paths and secret-like
values, writes live files atomically, and commits only the live/ payload.
"""
import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone

STEP_RE = re.compile(r"^step (\d+): train ([\d.eE+-]+|nan) ema ([\d.eE+-]+|nan)")
VAL_RE = re.compile(
    r"val=\(([-\d.eE+]+|nan)(?:,\s*([-\d.eE+]+|nan))?"
    r"(?:,\s*([-\d.eE+]+|nan))?(?:,\s*([-\d.eE+]+|nan))?\)")
TOKS_RE = re.compile(r"(\d+) tok/s")
BEST_RE = re.compile(r"^new best validation total=([\d.eE+-]+) at step (\d+)")
CKPT_RE = re.compile(r"^checkpoint step (\d+): (.+)$")
COMPLETE_RE = re.compile(
    r"^complete: (\d+) tok/s, (\d+) new tokens, (\d+) total; val=\(([-\d.eE+]+|nan)")
EARLY_RE = re.compile(r"^early stopping at step (\d+); best step=(\d+) val total=([\d.eE+-]+)")
LOCAL_PATH_RE = re.compile(
    r"(?i)(?:[A-Z]:[/\\\\]|\\\\\\\\[^\\\s]+\\[^\\\s]+|"
    r"/(?:kaggle/(?:input|working)|home|Users|root|workspace|mnt|tmp|var|opt|srv|etc|private|Volumes)/)[^\s\"']*")
SECRET_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|"
    r"kaggle_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|"
    r"AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"npm_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._~+/-]{16,})(?![A-Za-z0-9])")
SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(?:token|key|secret|password|credential|authorization)(?:$|[_-])")
PUBLIC_HISTORY_NUMERIC_FIELDS = {
    "train", "ema", "tok_s", "tokens_seen", "new_tokens",
    "val_v2", "val_v3a", "val_legacy", "val_lm_v2", "val_lm_v3a",
    "val_lm_legacy", "val_close_v2", "val_close_v3a", "val_close_legacy",
    "val_depth_v2", "val_depth_v3a", "val_depth_legacy", "best_step",
    "val_total", "val_total_v2", "val_total_v3a", "val_total_legacy",
    "val", "val_lm", "val_close", "val_depth",
}
PUBLIC_CONFIG_NUMERIC_FIELDS = {
    "parameters", "learning_rate", "factored_beta1", "batch_size", "grad_accum",
    "seq_len", "chunk_size", "think_steps", "aux_w", "eval_batches",
    "train_bytes", "validation_bytes", "train_tokens", "validation_tokens",
    "optimizer_state_mib", "warmup_steps",
}
PUBLIC_STATUS_NUMERIC_FIELDS = {
    "step", "target_step", "start_step", "start_tokens", "tokens_seen",
    "train_loss", "train_ema", "val_total", "val_lm", "val_close",
    "val_depth", "best_val", "best_step", "tok_per_s",
}
PUBLIC_EVAL_ROW_FIELDS = {
    "balance", "balanced", "doctype", "html_pair", "body", "body_pair",
    "mismatch", "unclosed", "hit_cap", "mean_generated_bytes",
    "step", "samples", "checkpoint_step",
}
PUBLIC_EVAL_SETTINGS_FIELDS = {
    "prompts_per_checkpoint", "max_new_bytes", "max_new", "temperature",
    "top_k", "prompts", "samples_per_prompt",
}
DISPLAY_EVAL_METRIC_FIELDS = ("balance", "balanced", "doctype", "html_pair", "body_pair")


def read_lines(path):
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                return f.read().splitlines()
        except (UnicodeError, ValueError):
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace").splitlines()


def number(value):
    if value is None or value.lower() == "nan":
        return None
    return float(value)


def finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def parse_logs(paths):
    history = []
    last = {}
    finished = False
    for path in paths:
        for raw in read_lines(path):
            line = raw.strip()
            match = STEP_RE.match(line)
            if match:
                step, train, ema = match.groups()
                rec = {"step": int(step), "train": number(train),
                       "ema": number(ema), "event": "log"}
                val = VAL_RE.search(line)
                rate = TOKS_RE.search(line)
                if val:
                    # The log format lacks corpus labels. Keep these metrics explicitly
                    # legacy rather than guessing v2/v3a from context.
                    rec["val_legacy"] = number(val.group(1))
                    for key, group in (("val_lm_legacy", 2), ("val_close_legacy", 3),
                                       ("val_depth_legacy", 4)):
                        value = number(val.group(group))
                        if value is not None:
                            rec[key] = value
                if rate:
                    rec["tok_s"] = int(rate.group(1))
                history.append(rec)
                last = rec
                continue
            match = BEST_RE.match(line)
            if match:
                value, step = match.groups()
                history.append({"step": int(step), "val_legacy": float(value),
                                "event": "best"})
                continue
            match = CKPT_RE.match(line)
            if match:
                step, path_value = match.groups()
                history.append({"step": int(step),
                                "checkpoint": os.path.basename(path_value.strip()),
                                "kind": "best" if "_best.pt" in path_value else "last",
                                "event": "checkpoint"})
                continue
            match = EARLY_RE.match(line)
            if match:
                step, best_step, value = match.groups()
                history.append({"step": int(step), "best_step": int(best_step),
                                "val_legacy": float(value), "event": "early-stop"})
                finished = True
                continue
            match = COMPLETE_RE.match(line)
            if match:
                rate, new_tokens, total, val = match.groups()
                history.append({"step": last.get("step"), "tok_s": int(rate),
                                "new_tokens": int(new_tokens),
                                "tokens_seen": int(total), "val_legacy": number(val),
                                "event": "complete"})
                finished = True
    return history, last, finished


def unique_rows(rows):
    seen = set()
    out = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key not in seen:
            seen.add(key)
            out.append(row)
    out.sort(key=lambda row: (
        row.get("step") if isinstance(row.get("step"), int)
        and not isinstance(row.get("step"), bool) else -1,
        row.get("event") if isinstance(row.get("event"), str) else ""))
    return out


def safe_source_name(value):
    if (not isinstance(value, str) or SECRET_RE.search(value)
            or LOCAL_PATH_RE.search(value)
            or not re.fullmatch(r"[a-z0-9_-]{1,48}", value)):
        return None
    return value


def safe_public_text(value, *, max_length, allowed_pattern):
    if (not isinstance(value, str) or len(value) > max_length
            or SECRET_RE.search(value) or LOCAL_PATH_RE.search(value)
            or not re.fullmatch(allowed_pattern, value)):
        return None
    return value


def public_value(value, key=""):
    if SENSITIVE_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): public_value(v, str(k)) for k, v in value.items()
                if str(k).lower() not in {"prompt", "prompts", "sample", "samples",
                                          "completion", "completions", "generated_text",
                                          "html", "content"}}
    if isinstance(value, list):
        return [public_value(item, key) for item in value[-100_000:]]
    if isinstance(value, str):
        value = SECRET_RE.sub("[REDACTED]", value)
        value = LOCAL_PATH_RE.sub("[local path]", value)
        value = re.sub(
            r"(?im)((?:authorization|api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*)"
            r"(?!\[REDACTED\])[^\s,;]+", r"\1[REDACTED]", value)
        return value[:500_000]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


def fallback_validation_metric(row):
    if not isinstance(row, dict):
        return None, None
    source = safe_source_name(row.get("val_total_source"))
    total = row.get("val_total")
    if source is not None and finite_number(total):
        return total, source
    for key, source in (("val_v3a", "v3a"), ("val_v2", "v2"),
                        ("val_legacy", "legacy"), ("val", "legacy")):
        value = row.get(key)
        if finite_number(value):
            return value, source
    return None, None


def apply_history_provenance(status, history):
    result = dict(status)
    current = next((row for row in reversed(history)
                    if row.get("event") in {"log", "eval"}), {})
    metric, source = fallback_validation_metric(current)
    if (source and finite_number(result.get("val_total"))
            and result["val_total"] == metric
            and safe_source_name(result.get("val_total_source")) is None):
        result["val_total_source"] = source
    best_rows = [row for row in history if row.get("event") == "best"]
    best_metric, best_source = fallback_validation_metric(best_rows[-1] if best_rows else {})
    if (best_source and finite_number(result.get("best_val"))
            and result["best_val"] == best_metric
            and safe_source_name(result.get("best_val_source")) is None):
        result["best_val_source"] = best_source
    return result


def public_history(rows):
    safe = []
    numeric_fields = PUBLIC_HISTORY_NUMERIC_FIELDS
    valid_events = {"log", "eval", "best", "checkpoint", "complete", "early-stop"}
    for row in rows:
        event = row.get("event") if isinstance(row, dict) else None
        if not isinstance(row, dict) or event not in valid_events:
            continue
        step = row.get("step")
        if (not isinstance(step, int) or isinstance(step, bool) or step < 0):
            if event != "complete":
                continue
        item = {"event": event}
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
            item["step"] = step
        for field in numeric_fields:
            value = row.get(field)
            if finite_number(value):
                item[field] = value
        source = safe_source_name(row.get("val_total_source"))
        if source is not None:
            item["val_total_source"] = source
        if row.get("kind") in {"best", "last"}:
            item["kind"] = row["kind"]
        checkpoint = safe_public_name(row.get("checkpoint"), None)
        if checkpoint:
            item["checkpoint"] = checkpoint
        safe.append(item)
    return safe


def first_metric(row, keys):
    if not isinstance(row, dict):
        return None
    return next((row[key] for key in keys if finite_number(row.get(key))), None)


def public_config(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    positive_fields = {"parameters", "learning_rate", "batch_size", "grad_accum",
                       "seq_len", "chunk_size", "think_steps", "eval_batches"}
    nonnegative_fields = PUBLIC_CONFIG_NUMERIC_FIELDS - positive_fields
    for key in PUBLIC_CONFIG_NUMERIC_FIELDS:
        item = value.get(key)
        valid = finite_number(item)
        if valid and key in positive_fields:
            valid = item > 0
        elif valid and key in nonnegative_fields:
            valid = item >= 0
        if valid:
            result[key] = item
    if isinstance(value.get("compiled"), bool):
        result["compiled"] = value["compiled"]
    if isinstance(value.get("tf32"), bool):
        result["tf32"] = value["tf32"]
    for key in {"model", "optimizer", "precision", "dataset", "hardware"}:
        item = safe_public_text(value.get(key), max_length=120,
                                allowed_pattern=r"[A-Za-z0-9 _().,+-]+")
        if item is not None:
            result[key] = item
    betas = value.get("betas")
    if isinstance(betas, list) and len(betas) == 2 and all(
            finite_number(item) for item in betas):
        result["betas"] = betas
    return result


def public_eval_row(value):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in PUBLIC_EVAL_ROW_FIELDS
            if key in value and finite_number(value[key])}


def display_eval_row(value):
    row = public_eval_row(value)
    samples = row.get("samples")
    if (not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0
            or any(not finite_number(row.get(key)) for key in DISPLAY_EVAL_METRIC_FIELDS)):
        return None
    return row


def public_eval_settings(value):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in PUBLIC_EVAL_SETTINGS_FIELDS
            if key in value and finite_number(value[key])}


def safe_public_name(value, fallback):
    if not isinstance(value, str) or SECRET_RE.search(value) or LOCAL_PATH_RE.search(value):
        return fallback
    name = value.strip()
    if (len(name) > 80 or "/" in name or "\\" in name
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name in {".", ".."}):
        return fallback
    return name


def generation_settings_valid(value):
    settings = public_eval_settings(value)
    temperature = settings.get("temperature")
    top_k = settings.get("top_k")
    cap = settings.get("max_new_bytes", settings.get("max_new"))
    prompts = settings.get("prompts_per_checkpoint", settings.get("prompts"))
    return (finite_number(temperature) and 0 < temperature <= 5
            and finite_number(top_k) and top_k >= 0
            and isinstance(cap, int) and not isinstance(cap, bool) and cap > 0
            and isinstance(prompts, int) and not isinstance(prompts, bool) and prompts > 0)


def public_generation(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    run = safe_public_name(value.get("run"), None)
    if run:
        result["run"] = run
    for key in ("total_samples", "balanced_samples", "doctype_samples"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    settings = value.get("settings")
    if isinstance(settings, dict):
        result["settings"] = public_eval_settings(settings)
    evaluations = []
    source_evaluations = value.get("evaluations")
    if isinstance(source_evaluations, list):
        evaluations = [row for raw in source_evaluations[-10_000:]
                       if (row := display_eval_row(raw)) is not None]
    latest = display_eval_row(value.get("latest"))
    if latest is None and evaluations:
        latest = evaluations[-1]
    if latest is not None:
        result["latest"] = latest
    if evaluations:
        result["evaluations"] = evaluations
    comparison = value.get("paired_checkpoint_comparison")
    if isinstance(comparison, dict):
        safe_comparison = public_comparison(comparison)
        if display_eval_row(safe_comparison.get("before")) is not None and display_eval_row(
                safe_comparison.get("after")) is not None:
            result["paired_checkpoint_comparison"] = safe_comparison
    return result


def generation_has_metrics(value):
    """Require complete dashboard metrics and a positive sample count."""
    if not isinstance(value, dict):
        return False
    if display_eval_row(value.get("latest")) is not None:
        return True
    evaluations = value.get("evaluations")
    if isinstance(evaluations, list) and any(
            display_eval_row(row) is not None for row in evaluations):
        return True
    comparison = value.get("paired_checkpoint_comparison")
    return (isinstance(comparison, dict)
            and display_eval_row(comparison.get("before")) is not None
            and display_eval_row(comparison.get("after")) is not None)


def public_comparison(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("method", "caveat"):
        text = safe_public_text(value.get(key), max_length=500,
                                allowed_pattern=r"[A-Za-z0-9 ,;:.()\-→]+")
        if text is not None:
            result[key] = text
    settings = value.get("settings")
    if isinstance(settings, dict):
        result["settings"] = public_eval_settings(settings)
    for key in ("before", "after"):
        row = display_eval_row(value.get(key))
        if row is not None:
            result[key] = row
    return result


def public_status(status):
    if not isinstance(status, dict):
        return {}
    result = {}
    for key in PUBLIC_STATUS_NUMERIC_FIELDS:
        value = status.get(key)
        if finite_number(value):
            result[key] = value
    state = status.get("state")
    if isinstance(state, str) and state in {"training", "idle", "done"}:
        result["state"] = state
    updated_at = status.get("updated_at")
    if isinstance(updated_at, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", updated_at):
        result["updated_at"] = updated_at
    result["run"] = safe_public_name(status.get("run"), "local-run")
    for field in ("checkpoint", "best_checkpoint"):
        if field in status:
            result[field] = safe_public_name(status[field], None)
    if isinstance(status.get("config"), dict):
        result["config"] = public_config(status["config"])
    if isinstance(status.get("generation_eval"), dict):
        eval_row = display_eval_row(status["generation_eval"])
        if eval_row is not None:
            if isinstance(status["generation_eval"].get("settings"), dict):
                eval_row["settings"] = public_eval_settings(
                    status["generation_eval"]["settings"])
            result["generation_eval"] = eval_row
    if isinstance(status.get("generation_comparison"), dict):
        comparison = public_comparison(status["generation_comparison"])
        if (display_eval_row(comparison.get("before")) is not None
                and display_eval_row(comparison.get("after")) is not None):
            result["generation_comparison"] = comparison
    for key in ("val_total_source", "best_val_source"):
        source = safe_source_name(status.get(key))
        if source is not None:
            result[key] = source
    return result


def atomic_write(path, text):
    temp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def git(repo, *argv):
    return subprocess.run(["git", "-C", repo, *argv],
                          capture_output=True, text=True, timeout=120)


PUBLISHER_LIVE_PATHS = (
    "live/status.json", "live/history.jsonl", "live/logtail.txt",
    "live/generation_eval.json",
)


def preflight_once(repo):
    """Reject pre-existing publisher target changes without remote-branch checks."""
    result = git(repo, "status", "--porcelain", "--untracked-files=all",
                 "--", *PUBLISHER_LIVE_PATHS)
    if result.returncode != 0:
        raise RuntimeError("cannot inspect publisher-owned live paths")
    if result.stdout.strip():
        raise RuntimeError("refusing one-shot write over pre-existing live metric changes")


def preflight_publish(repo, *, allow_metric_changes=False):
    branch = git(repo, "branch", "--show-current")
    if branch.returncode != 0 or not branch.stdout.strip():
        raise RuntimeError("refusing to publish from a detached HEAD")
    branch_name = branch.stdout.strip()
    if branch_name != "main":
        raise RuntimeError("refusing to publish outside main")
    upstream = git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if upstream.returncode != 0:
        raise RuntimeError("refusing to publish without an upstream tracking branch")
    upstream_name = upstream.stdout.strip()
    if upstream_name != f"origin/{branch_name}":
        raise RuntimeError("refusing to publish outside matching origin branch")
    remote = git(repo, "remote", "get-url", "--push", "origin")
    if remote.returncode != 0 or not re.fullmatch(
            r"(?:https://github\.com/Vaskrokodile/lsm-live(?:\.git)?|"
            r"git@github\.com:Vaskrokodile/lsm-live(?:\.git)?)", remote.stdout.strip()):
        raise RuntimeError("refusing to publish to an unexpected origin push URL")
    staged = git(repo, "diff", "--cached", "--quiet")
    if staged.returncode != 0:
        raise RuntimeError("refusing to publish with pre-existing staged changes")
    relation = git(repo, "rev-list", "--left-right", "--count", f"{upstream_name}...HEAD")
    if relation.returncode != 0:
        raise RuntimeError("cannot verify upstream commit relationship")
    try:
        behind_count, ahead_count = map(int, relation.stdout.split())
    except (ValueError, TypeError) as exc:
        raise RuntimeError("cannot parse upstream commit relationship") from exc
    if behind_count or ahead_count:
        raise RuntimeError("refusing to publish unless branch is exactly at upstream")
    if not allow_metric_changes:
        dirty_targets = git(repo, "status", "--porcelain", "--untracked-files=all",
                            "--", *PUBLISHER_LIVE_PATHS)
        if dirty_targets.returncode != 0:
            raise RuntimeError("cannot inspect live metric working-tree state")
        if dirty_targets.stdout.strip():
            raise RuntimeError("refusing to overwrite pre-existing live metric changes")
    return branch_name, upstream_name


def publish(repo, message, include_generation=False):
    # Check checkout and upstream state before staging or committing anything.
    branch_name, upstream_name = preflight_publish(repo, allow_metric_changes=True)
    paths = ["live/status.json", "live/history.jsonl", "live/logtail.txt"]
    if include_generation:
        paths.append("live/generation_eval.json")
    result = git(repo, "add", "--", *paths)
    if result.returncode != 0:
        raise RuntimeError("git add of publisher-owned files failed")
    staged = git(repo, "diff", "--cached", "--quiet", "--", *paths)
    if staged.returncode not in (0, 1):
        raise RuntimeError("cannot inspect publisher-owned staged files")
    if staged.returncode == 0:
        print("[publish] no metric changes; push skipped", flush=True)
        return
    result = git(repo, "commit", "--only", "-m", message, "--", *paths)
    if result.returncode != 0:
        raise RuntimeError("publisher-scoped commit failed")
    relation = git(repo, "rev-list", "--left-right", "--count", f"{upstream_name}...HEAD")
    if relation.returncode != 0 or relation.stdout.split() != ["0", "1"]:
        raise RuntimeError("refusing to push: HEAD is not exactly one publisher commit ahead")
    result = git(repo, "push", "origin", f"HEAD:refs/heads/{branch_name}")
    if result.returncode != 0:
        raise RuntimeError("git push failed")
    print(f"[publish] pushed {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--history-seed", default=None,
                        help="prior JSONL history to retain across run legs")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--start-tokens", type=int, default=0)
    parser.add_argument("--target-step", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true",
                        help="write one local snapshot and exit without pushing")
    parser.add_argument("--telemetry", default=None,
                        help="optional atomic JSON snapshot with status/history/log_tail")
    parser.add_argument("--generation-eval", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 5:
        parser.error("interval must be a finite value of at least 5 seconds")
    if not args.log and not args.telemetry:
        parser.error("at least one --log or --telemetry source is required")
    if not os.path.isdir(args.repo) or not os.path.isdir(os.path.join(args.repo, ".git")):
        parser.error("repo must be an existing Git checkout")
    if args.once and not args.telemetry:
        parser.error("--once requires --telemetry")
    if args.once:
        try:
            preflight_once(args.repo)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            parser.error(f"one-shot local preflight failed: {type(exc).__name__}")
    else:
        try:
            preflight_publish(args.repo)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            parser.error(f"public publish preflight failed: {type(exc).__name__}")
    out = os.path.join(args.repo, "live")
    try:
        config = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("config root must be an object")
        seed = []
        if args.history_seed:
            seed = [json.loads(line) for line in read_lines(args.history_seed) if line.strip()]
            if any(not isinstance(row, dict) for row in seed):
                raise ValueError("history seed rows must be objects")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if args.once:
            parser.error(f"one-shot input error: {type(exc).__name__}")
        raise

    try:
        os.makedirs(out, exist_ok=True)
    except OSError as exc:
        if args.once:
            parser.error(f"one-shot output initialization failed: {type(exc).__name__}")
        raise

    def invalid_input(message):
        if args.once:
            parser.error(f"one-shot {message}")
        print(f"[publish] {message}; retrying", flush=True)
        time.sleep(args.interval)

    print(f"[publish] run={args.run} interval={args.interval:g}s "
          f"logs={len(args.log)} telemetry={bool(args.telemetry)}", flush=True)
    while True:
        try:
            parsed, last, finished = parse_logs(args.log) if args.log else ([], {}, False)
            telemetry = None
            if args.telemetry:
                with open(args.telemetry, "r", encoding="utf-8") as f:
                    telemetry = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            invalid_input(f"input error ({type(exc).__name__})")
            continue

        if telemetry is not None and not isinstance(telemetry, dict):
            invalid_input("invalid telemetry root")
            continue
        if (telemetry is not None and "status" in telemetry
                and not isinstance(telemetry["status"], dict)):
            invalid_input("invalid telemetry status")
            continue
        history = telemetry.get("history", parsed) if telemetry else parsed
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            invalid_input("invalid history snapshot")
            continue
        try:
            # Project untrusted telemetry before canonicalization: NaN/Inf and
            # attacker-controlled non-scalar fields must not break JSON deduping.
            history = unique_rows(public_history(seed + history))
        except (TypeError, ValueError, KeyError) as exc:
            invalid_input(f"invalid history values ({type(exc).__name__})")
            continue
        if not isinstance(history, list):
            invalid_input("invalid sanitized history")
            continue
        safe_generation = None
        generation_valid = False

        bests = [row for row in history if row.get("event") == "best"]
        complete = [row for row in history if row.get("event") == "complete"]
        if telemetry and isinstance(telemetry.get("status"), dict):
            status = dict(telemetry["status"])
            if not isinstance(status.get("config"), dict):
                status["config"] = {}
        else:
            current = next((row for row in reversed(history)
                            if row.get("event") in {"log", "eval"}), last)
            best_row = bests[-1] if bests else {}

            val_total, val_source = fallback_validation_metric(current)
            best_val, best_source = fallback_validation_metric(best_row)
            if complete:
                tokens = complete[-1].get("tokens_seen")
            else:
                step_for_tokens = current.get("step", args.start_step)
                tokens = args.start_tokens + (step_for_tokens - args.start_step) * int(
                    config.get("batch_size", 1) * config.get("seq_len", 1)
                    * config.get("grad_accum", 1))
            mtime = datetime.now(timezone.utc)
            if args.log:
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(args.log[-1]), tz=timezone.utc)
                except OSError:
                    pass
            status = {
                "updated_at": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "state": "done" if finished else "training",
                "run": args.run,
                "step": current.get("step", args.start_step),
                "target_step": args.target_step,
                "start_step": args.start_step,
                "start_tokens": args.start_tokens,
                "tokens_seen": tokens,
                "train_loss": current.get("train"),
                "train_ema": current.get("ema"),
                "val_total": val_total,
                "val_total_source": val_source,
                "val_lm": first_metric(current, ("val_lm_v3a", "val_lm_v2", "val_lm_legacy", "val_lm")),
                "val_close": first_metric(current, ("val_close_v3a", "val_close_v2", "val_close_legacy", "val_close")),
                "val_depth": first_metric(current, ("val_depth_v3a", "val_depth_v2", "val_depth_legacy", "val_depth")),
                "best_val": best_val,
                "best_val_source": best_source,
                "best_step": best_row.get("step"),
                "tok_per_s": current.get("tok_s"),
                "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else None,
                "config": config,
            }
        if not isinstance(status.get("config"), dict):
            status["config"] = {}
        status.setdefault("updated_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        status.setdefault("run", args.run)
        if args.config:
            status["config"].update(config)
        if args.generation_eval:
            try:
                with open(args.generation_eval, "r", encoding="utf-8") as f:
                    generation = json.load(f)
                if not isinstance(generation, dict):
                    raise ValueError("generation evaluation must be an object")
                safe_generation = public_generation(generation)
                if not generation_has_metrics(safe_generation):
                    raise ValueError("generation evaluation has no valid metric rows")
                generation_valid = True
                latest_eval = safe_generation.get("latest")
                if isinstance(latest_eval, dict):
                    latest_eval = dict(latest_eval)
                    latest_eval["settings"] = safe_generation.get("settings", {})
                status["generation_eval"] = latest_eval
                status["generation_comparison"] = safe_generation.get("paired_checkpoint_comparison")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"[publish] generation eval error ({type(exc).__name__}); skipping snapshot", flush=True)
                if args.once:
                    parser.error(f"one-shot generation evaluation invalid: {type(exc).__name__}")
                time.sleep(args.interval)
                continue
        status = public_status(apply_history_provenance(status, history))
        if not isinstance(status, dict):
            invalid_input("invalid status snapshot")
            continue
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            invalid_input("invalid sanitized history")
            continue
        # Construct the tail only from validated public history. This rejects string
        # steps/newlines before they can become additional public lines.
        tail = []
        for row in history[-30:]:
            step, event = row.get("step"), row["event"]
            fields = [f"{key}={row[key]}" for key in (
                "train", "ema", "val_v2", "val_v3a", "val_legacy",
                "val_total", "val_lm", "val_close", "val_depth", "tok_s")
                if finite_number(row.get(key))]
            if event == "checkpoint" and row.get("kind") in {"best", "last"}:
                tail.append(f"checkpoint step {step}: {row['kind']}")
            elif fields and event in {"log", "eval", "best", "complete"}:
                tail.append(f"{event} step {step if step is not None else 'unknown'}: " + " ".join(fields))
        try:
            status_text = json.dumps(status, indent=2, allow_nan=False) + "\n"
            history_text = "".join(
                json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                for row in history)
            atomic_write(os.path.join(out, "status.json"), status_text)
            atomic_write(os.path.join(out, "history.jsonl"), history_text)
            atomic_write(os.path.join(out, "logtail.txt"), "\n".join(tail) + ("\n" if tail else ""))
            if generation_valid:
                atomic_write(os.path.join(out, "generation_eval.json"),
                             json.dumps(safe_generation, indent=2, allow_nan=False) + "\n")
        except (TypeError, ValueError, OSError) as exc:
            if args.once:
                parser.error(f"one-shot snapshot write failed: {type(exc).__name__}")
            print("[publish] snapshot serialization/write failed; retrying", flush=True)
            time.sleep(args.interval)
            continue
        if args.once:
            print(f"[publish] wrote local snapshot for step {status.get('step', '?')}; push disabled",
                  flush=True)
            return
        try:
            step = status.get("step", "?")
            publish(args.repo,
                    f"live: {status.get('run', args.run)} step {step} ({status.get('state', 'training')})",
                    include_generation=generation_valid)
        except Exception as exc:
            print(f"[publish] error ({type(exc).__name__}); details suppressed", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
