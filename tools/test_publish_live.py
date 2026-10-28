import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import publish_live as publisher


class PublicProjectionTests(unittest.TestCase):
    def test_config_allows_only_valid_public_values(self):
        result = publisher.public_config({
            "model": "LSM byte-level latent LM",
            "dataset": "Webmix v3a",
            "hardware": "RTX 3060",
            "parameters": 32_860_000,
            "learning_rate": 0.0001,
            "batch_size": 4,
            "seq_len": 512,
            "compiled": False,
            "betas": [0.9, 0.9841],
            "credential": "do-not-publish",
            "report_path": "C:/private/train.log",
            "unexpected": "drop-me",
        })
        self.assertEqual(result["parameters"], 32_860_000)
        self.assertEqual(result["learning_rate"], 0.0001)
        self.assertEqual(result["batch_size"], 4)
        self.assertEqual(result["seq_len"], 512)
        self.assertEqual(result["dataset"], "Webmix v3a")
        self.assertNotIn("credential", result)
        self.assertNotIn("report_path", result)
        self.assertNotIn("unexpected", result)
        self.assertNotIn("report_path", json.dumps(result))

    def test_history_filters_bad_rows_and_keeps_complete_without_step(self):
        projected = publisher.public_history([
            {"event": "log", "step": "1\nforged", "train": 0.2},
            {"event": "log", "step": 1, "train": float("inf"), "val_v3a": 0.4,
             "generated_text": "<script>bad</script>"},
            {"event": "complete", "step": None, "tokens_seen": 123},
        ])
        self.assertEqual(projected, [
            {"event": "log", "step": 1, "val_v3a": 0.4},
            {"event": "complete", "tokens_seen": 123},
        ])

    def test_generation_display_requires_complete_metrics_and_positive_samples(self):
        complete = {"step": 5, "samples": 4, "balance": 0.2, "balanced": 0.1,
                    "doctype": 0.2, "html_pair": 0.3, "body_pair": 0.4}
        self.assertTrue(publisher.generation_has_metrics(
            publisher.public_generation({"latest": complete})))
        self.assertEqual(publisher.public_generation({"latest": complete})["latest"], complete)
        for missing in ("samples", "balance", "balanced", "doctype", "html_pair", "body_pair"):
            with self.subTest(missing=missing):
                row = dict(complete)
                row.pop(missing)
                self.assertFalse(publisher.generation_has_metrics(
                    publisher.public_generation({"latest": row})))
        for samples in (0, -1, 2.5, True):
            with self.subTest(samples=samples):
                row = dict(complete, samples=samples)
                self.assertFalse(publisher.generation_has_metrics(
                    publisher.public_generation({"latest": row})))
        self.assertFalse(publisher.generation_has_metrics(
            publisher.public_generation({"latest": {"step": 5, "mismatch": 0}})))
        self.assertFalse(publisher.generation_has_metrics(
            publisher.public_generation({"evaluations": [{"step": 5}]})))
        self.assertFalse(publisher.generation_has_metrics(
            publisher.public_generation({"paired_checkpoint_comparison": {
                "caveat": "no valid metric rows"}})))

    def test_generation_allowlist_preserves_comparison_provenance_only(self):
        projected = publisher.public_generation({
            "run": "run-1", "total_samples": 72,
            "generated_text": "<html>secret sample</html>",
            "settings": {"prompts_per_checkpoint": 4, "temperature": 0.8,
                         "prompts": ["private prompt"]},
            "latest": {"step": 82608, "samples": 4, "balance": 0.2,
                       "balanced": 0.1, "doctype": 0.05, "html_pair": 0.25,
                       "body_pair": 0.25, "html": "<html>raw</html>"},
            "paired_checkpoint_comparison": {
                "method": "same four prompts and decoding configuration",
                "settings": {"prompts": 4, "samples_per_prompt": 8,
                             "max_new_bytes": 768, "temperature": 0.8, "top_k": 50},
                "before": {"checkpoint_step": 30783, "balance": 0.12,
                           "samples": 32, "balanced": 0.08, "doctype": 0.06,
                           "html_pair": 0.03, "body_pair": 0.02},
                "after": {"checkpoint_step": 82608, "balance": 0.15,
                          "samples": 32, "balanced": 0.1, "doctype": 0.0,
                          "html_pair": 0.04, "body_pair": 0.03},
                "caveat": "Small stochastic sample; all samples hit the 768-byte generation cap.",
                "generated_text": "drop",
            },
        })
        comparison = projected["paired_checkpoint_comparison"]
        self.assertEqual(comparison["before"]["checkpoint_step"], 30783)
        self.assertEqual(comparison["after"]["checkpoint_step"], 82608)
        self.assertEqual(comparison["before"]["samples"], 32)
        self.assertIn("caveat", comparison)
        self.assertNotIn("generated_text", comparison)
        self.assertNotIn("prompts", projected["settings"])
        self.assertNotIn("generated_text", projected)
        self.assertNotIn("html", projected["latest"])

    def test_incomplete_comparison_omitted_even_with_valid_latest_row(self):
        latest = {"step": 10, "samples": 4, "balance": 0.5, "balanced": 0.25,
                  "doctype": 0.5, "html_pair": 0.5, "body_pair": 0.25}
        projected = publisher.public_generation({
            "latest": latest,
            "paired_checkpoint_comparison": {
                "before": {"checkpoint_step": 5, "samples": 4, "balance": 0.1},
                "after": {"checkpoint_step": 10, "samples": 4, "balance": 0.5,
                          "balanced": 0.25, "doctype": 0.5,
                          "html_pair": 0.5, "body_pair": 0.25},
            },
        })
        self.assertTrue(publisher.generation_has_metrics(projected))
        self.assertIn("latest", projected)
        self.assertNotIn("paired_checkpoint_comparison", projected)

    def test_text_allowlists_reject_secret_tokens_and_local_paths(self):
        secrets = ["ghp_" + "a" * 28, "sk-" + "b" * 24, "Bearer " + "c" * 24]
        for candidate in secrets + ["runs /home/alice/model", "C:/Users/alice/run"]:
            with self.subTest(candidate=candidate):
                self.assertIsNone(publisher.safe_public_text(
                    candidate, max_length=500,
                    allowed_pattern=r"[A-Za-z0-9 ,;:.()\\-→]+"))
        result = publisher.public_config({"model": "safe " + secrets[0],
                                          "dataset": "/home/alice/data"})
        self.assertEqual(result, {})
        self.assertIsNone(publisher.safe_public_name("C:/Users/alice/run", None))

    def test_status_omits_incomplete_generation_eval_and_comparison(self):
        projected = publisher.public_status({
            "generation_eval": {"samples": 8, "balance": 0.2},
            "generation_comparison": {
                "before": {"samples": 8, "balance": 0.2},
                "after": {"samples": 8, "balance": 0.3, "balanced": 0.1,
                          "doctype": 0.2, "html_pair": 0.2, "body_pair": 0.1},
            },
        })
        self.assertNotIn("generation_eval", projected)
        self.assertNotIn("generation_comparison", projected)

    def test_status_drops_paths_and_keeps_sources(self):
        result = publisher.public_status({
            "run": "safe-run", "state": "done", "step": 8,
            "val_total": 0.5, "val_total_source": "legacy",
            "best_val": 0.4, "best_val_source": "v3a",
            "note": "C:/private/path", "unknown": "drop",
        })
        self.assertEqual(result["val_total_source"], "legacy")
        self.assertEqual(result["best_val_source"], "v3a")
        self.assertNotIn("note", result)
        self.assertNotIn("unknown", result)

    def test_embedded_tokens_in_allowlisted_names_and_sources_are_rejected(self):
        token = "ghp_" + "a" * 28
        self.assertIsNone(publisher.safe_public_name(f"run_{token}", None))
        self.assertIsNone(publisher.safe_source_name(f"run_{token}"))
        self.assertIsNone(publisher.safe_public_name(f"checkpoint_{token}", None))
        self.assertIsNone(publisher.safe_public_text(
            f"prefix_{token}", max_length=500,
            allowed_pattern=r"[A-Za-z0-9 _().,+-]+"))
        projected = publisher.public_status({
            "run": f"run_{token}", "checkpoint": f"checkpoint_{token}",
            "val_total_source": f"run_{token}",
        })
        self.assertEqual(projected["run"], "local-run")
        self.assertIsNone(projected["checkpoint"])
        self.assertNotIn("val_total_source", projected)

    def test_windows_and_unix_local_paths_are_redacted(self):
        for path in ("C:\\private\\train.log", "C:/private/train.log",
                     "/tmp/private/train.log", "/var/tmp/private/train.log"):
            with self.subTest(path=path):
                self.assertNotIn(path, publisher.public_value(path))

    def test_history_provenance_fills_missing_sources_without_overriding_explicit_source(self):
        history = publisher.public_history([
            {"event": "log", "step": 4, "val_total": 0.6,
             "val_total_source": "addon_webmix"},
            {"event": "best", "step": 3, "val_total": 0.5,
             "val_total_source": "addon_webmix"},
        ])
        status = publisher.apply_history_provenance(
            {"val_total": 0.6, "best_val": 0.5}, history)
        self.assertEqual(status["val_total_source"], "addon_webmix")
        self.assertEqual(status["best_val_source"], "addon_webmix")
        explicit = publisher.apply_history_provenance(
            {"val_total": 0.6, "val_total_source": "v3a"}, history)
        self.assertEqual(explicit["val_total_source"], "v3a")

    def test_fallback_status_selects_labeled_validation_and_keeps_legacy_unlabeled(self):
        generic = publisher.fallback_validation_metric({
            "val_total": 0.3, "val_total_source": "addon_webmix"})
        self.assertEqual(generic, (0.3, "addon_webmix"))
        cases = [
            ({"step": 4, "event": "log", "val_v3a": 0.4}, 0.4, "v3a"),
            ({"step": 4, "event": "log", "val_v2": 0.5}, 0.5, "v2"),
            ({"step": 4, "event": "log", "val_legacy": 0.6}, 0.6, "legacy"),
        ]
        for row, expected, source in cases:
            with self.subTest(source=source):
                history = publisher.public_history([row])
                metric, observed_source = publisher.fallback_validation_metric(history[-1])
                self.assertEqual(metric, expected)
                self.assertEqual(observed_source, source)


class PublisherCliTests(unittest.TestCase):
    def test_once_writes_sanitized_full_snapshot_and_never_calls_git(self):
        with tempfile.TemporaryDirectory(prefix="lsm-pub-test-") as tmp:
            repo = Path(tmp) / "repo"
            live = repo / "live"
            live.mkdir(parents=True)
            (repo / ".git").mkdir()
            telemetry = Path(tmp) / "telemetry.json"
            generation = Path(tmp) / "generation.json"
            telemetry.write_text(json.dumps({
                "status": {
                    "updated_at": "2026-01-01T00:00:00Z", "state": "done",
                    "run": "fixture-run", "step": 5,
                    "config": {"model": "fixture model", "batch_size": 2,
                               "secret": "discard"},
                },
                "history": [
                    {"event": "log", "step": 5, "val_v3a": 0.5,
                     "train": 0.6, "generated_text": "<html>private</html>"},
                    {"event": "checkpoint", "step": 5,
                     "checkpoint": "C:/private/best.pt", "kind": "best"},
                    {"event": "log", "step": "6\\nforged", "train": 0.2},
                ],
            }), encoding="utf-8")
            generation.write_text(json.dumps({
                "run": "fixture-run", "total_samples": 4,
                "settings": {"prompts_per_checkpoint": 4, "max_new_bytes": 768,
                             "temperature": 0.8, "top_k": 50},
                "latest": {"step": 5, "samples": 4, "balance": 0.25,
                           "balanced": 0.0, "doctype": 0.0,
                           "html_pair": 0.25, "body_pair": 0.25},
            }), encoding="utf-8")
            argv = ["publish_live.py", "--repo", str(repo), "--run", "fixture-run",
                    "--telemetry", str(telemetry), "--generation-eval", str(generation),
                    "--once"]
            with patch.object(sys, "argv", argv), patch.object(
                    publisher, "preflight_publish", side_effect=AssertionError("public preflight in --once")), patch.object(
                    publisher, "publish", side_effect=AssertionError("publish in --once")), patch.object(
                    publisher, "git", return_value=subprocess.CompletedProcess(
                        [], 0, "", "")) as git_mock:
                publisher.main()
            git_mock.assert_called_once_with(
                str(repo), "status", "--porcelain", "--untracked-files=all",
                "--", *publisher.PUBLISHER_LIVE_PATHS)
            status = json.loads((live / "status.json").read_text(encoding="utf-8"))
            history = [json.loads(line) for line in
                       (live / "history.jsonl").read_text(encoding="utf-8").splitlines()]
            tail = (live / "logtail.txt").read_text(encoding="utf-8")
            public_generation = json.loads(
                (live / "generation_eval.json").read_text(encoding="utf-8"))
            self.assertEqual(status["config"]["batch_size"], 2)
            self.assertNotIn("secret", status["config"])
            self.assertEqual(status["step"], 5)
            log_row = next(row for row in history if row["event"] == "log")
            self.assertEqual(log_row["val_v3a"], 0.5)
            self.assertEqual(len(tail.splitlines()), 2)
            self.assertTrue(all("forged" not in line for line in tail.splitlines()))
            self.assertEqual(public_generation["latest"]["samples"], 4)
            self.assertEqual(status["generation_eval"]["samples"], 4)
            serialized = "".join([json.dumps(status), json.dumps(history), tail,
                                  json.dumps(public_generation)])
            self.assertNotIn("private", serialized)
            self.assertNotIn("generated_text", serialized)

    def test_once_preflight_rejects_dirty_target_before_writing(self):
        for status_text in (" M live/status.json\\n", "?? live/generation_eval.json\\n"):
            with self.subTest(status=status_text), tempfile.TemporaryDirectory(
                    prefix="lsm-pub-once-dirty-") as tmp:
                repo = Path(tmp) / "repo"
                live = repo / "live"
                live.mkdir(parents=True)
                (repo / ".git").mkdir()
                telemetry = Path(tmp) / "telemetry.json"
                telemetry.write_text(json.dumps({"status": {}, "history": []}),
                                     encoding="utf-8")
                argv = ["publish_live.py", "--repo", str(repo), "--run", "fixture-run",
                        "--telemetry", str(telemetry), "--once"]
                with patch.object(sys, "argv", argv), patch.object(
                        publisher, "git", return_value=subprocess.CompletedProcess(
                            [], 0, status_text, "")):
                    with self.assertRaises(SystemExit) as raised:
                        publisher.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(list(live.iterdir()), [])

    def test_once_derives_legacy_status_provenance_from_history(self):
        with tempfile.TemporaryDirectory(prefix="lsm-pub-legacy-status-") as tmp:
            repo = Path(tmp) / "repo"
            live = repo / "live"
            live.mkdir(parents=True)
            (repo / ".git").mkdir()
            telemetry = Path(tmp) / "telemetry.json"
            telemetry.write_text(json.dumps({
                "history": [{"event": "log", "step": 9, "val_legacy": 0.42}],
            }), encoding="utf-8")
            argv = ["publish_live.py", "--repo", str(repo), "--run", "fixture-run",
                    "--telemetry", str(telemetry), "--once"]
            with patch.object(sys, "argv", argv), patch.object(
                    publisher, "git", return_value=subprocess.CompletedProcess(
                        [], 0, "", "")) as git_mock:
                publisher.main()
            git_mock.assert_called_once_with(
                str(repo), "status", "--porcelain", "--untracked-files=all",
                "--", *publisher.PUBLISHER_LIVE_PATHS)
            status = json.loads((live / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["val_total"], 0.42)
            self.assertEqual(status["val_total_source"], "legacy")

    def test_preflight_rejects_non_main_before_payload_changes(self):
        with patch.object(publisher, "git") as git_mock:
            git_mock.return_value = subprocess.CompletedProcess([], 0, "feature\n", "")
            with self.assertRaisesRegex(RuntimeError, "outside main"):
                publisher.preflight_publish("unused")
            self.assertEqual(git_mock.call_count, 1)

    def test_preflight_rejects_untracked_metric_files(self):
        responses = [
            subprocess.CompletedProcess([], 0, "main\n", ""),
            subprocess.CompletedProcess([], 0, "origin/main\n", ""),
            subprocess.CompletedProcess([], 0,
                "https://github.com/Vaskrokodile/lsm-live.git\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "0\t0\n", ""),
            subprocess.CompletedProcess([], 0, "?? live/generation_eval.json\n", ""),
        ]
        with patch.object(publisher, "git", side_effect=responses) as git_mock:
            with self.assertRaisesRegex(RuntimeError, "pre-existing live metric"):
                publisher.preflight_publish("unused")
        self.assertEqual(git_mock.call_count, len(responses))

    def test_once_with_invalid_generation_preserves_existing_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="lsm-pub-invalid-generation-") as tmp:
            repo = Path(tmp) / "repo"
            live = repo / "live"
            live.mkdir(parents=True)
            (repo / ".git").mkdir()
            generation = Path(tmp) / "generation.json"
            generation.write_text(json.dumps({"run": "empty", "latest": {"step": 5}}),
                                  encoding="utf-8")
            previous = live / "generation_eval.json"
            previous.write_text("previous local snapshot\\n", encoding="utf-8")
            telemetry = Path(tmp) / "telemetry.json"
            telemetry.write_text(json.dumps({"status": {"step": 5}, "history": []}),
                                 encoding="utf-8")
            argv = ["publish_live.py", "--repo", str(repo), "--run", "fixture-run",
                    "--telemetry", str(telemetry), "--generation-eval", str(generation),
                    "--once"]
            with patch.object(sys, "argv", argv), patch.object(
                    publisher, "preflight_once"), patch.object(
                    publisher, "git", side_effect=AssertionError("raw Git invoked after mocked preflight")):
                with self.assertRaises(SystemExit) as raised:
                    publisher.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(previous.read_text(encoding="utf-8"), "previous local snapshot\\n")
            self.assertFalse((live / "status.json").exists())


if __name__ == "__main__":
    unittest.main()
