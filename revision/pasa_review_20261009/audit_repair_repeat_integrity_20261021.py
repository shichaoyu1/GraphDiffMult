"""Read-only audit of the returned repair repeat campaign; emit no patient data.

Run from any directory with the project Python environment. The only write is
the JSON receipt beside this script. Source hashes use git blob bytes, not the
Windows working tree, so CRLF conversion cannot appear as scientific drift.
"""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "output/20261019_AAAI_review"
REFERENCE = EVIDENCE / "pasa_repair_pilot_20261020_095259_432520"
REPEATS = EVIDENCE / "pasa_repair_repeats_20261020_171817_445198"
RECEIPT = Path(__file__).parent / "repair_repeats_audit_20261021/integrity_checks.json"
SCIENCE_COMMIT = "86a8e6f432b05f134923ecee574e0b8e688f54ce"
COORDINATOR_COMMIT = "5ce2b3b8643f9af1a4da848e74250eb5334f1314"
METHODS = ["whole_roi_mass", "whole_roi_field", "region_pool_mass", "region_pool_field"]


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def blob_digest(commit, filename):
    data = subprocess.check_output(["git", "show", commit + ":" + filename], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def main():
    checks = []

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    config = read(REPEATS / "repeat_config.json")
    protocol = read(REFERENCE / "protocol.json")
    combined = read(REPEATS / "summary.json")
    reference_names = {"protocol.json", "inputs.json", "anchor_vocab.json", "complete.json", "summary.json"}
    reference_hashes = config["identity"]["reference_sha256"]
    check("reference_artifact_set", set(reference_hashes) == reference_names, count=len(reference_hashes))
    for name, expected in reference_hashes.items():
        actual = digest(REFERENCE / name)
        check("reference_hash:" + name, actual == expected, expected_sha256=expected, actual_sha256=actual)
    check("science_commit_and_source_count", protocol["git_head"] == SCIENCE_COMMIT
          and len(protocol["source_sha256"]) == 11, count=len(protocol["source_sha256"]))
    for name, expected in protocol["source_sha256"].items():
        actual = blob_digest(SCIENCE_COMMIT, name)
        check("science_blob_hash:" + name, actual == expected, expected_sha256=expected, actual_sha256=actual)
    actual = blob_digest(COORDINATOR_COMMIT, "tools/repeat_pasa_repair_pilot.py")
    expected = config["identity"]["coordinator_sha256"]
    check("coordinator_blob_hash", config["git_head"] == COORDINATOR_COMMIT and actual == expected,
          expected_sha256=expected, actual_sha256=actual)
    check("repeat_identity", config["identity"]["version"] == "pasa_repair_repeats_v1"
          and config["identity"]["seeds"] == [43, 44])
    check("repeat_completion", read(REPEATS / "complete.json") == {
        "version": "pasa_repair_repeats_v1", "new_seeds": [43, 44], "new_runs": 8, "test_evaluated": False})
    attempts = list(REPEATS.glob("seed_*/attempt_*"))
    check("two_first_attempts_only", set(attempts) == {
        REPEATS / "seed_43/attempt_001", REPEATS / "seed_44/attempt_001"}, count=len(attempts))

    reference_records = read(REFERENCE / METHODS[0] / "validation_epoch_001.json")["records"]
    reference_masks = {key: value for key, value in reference_records.items() if key != "scores"}
    reference_prior = read(REFERENCE / METHODS[0] / "frequency_baseline_validation.json")
    run_count = epoch_count = checkpoint_count = 0
    for seed, base in [(42, REFERENCE), (43, REPEATS / "seed_43/attempt_001"),
                       (44, REPEATS / "seed_44/attempt_001")]:
        current = read(base / "protocol.json")
        manifest = read(base / "inputs.json")
        differences = sorted(key for key in set(current) | set(protocol)
                             if current.get(key) != protocol.get(key))
        expected_differences = [] if seed == 42 else ["repeat_reference", "seed"]
        check(f"seed_{seed}:protocol_difference", differences == expected_differences
              and current["seed"] == seed, differing_keys=differences)
        if seed != 42:
            check(f"seed_{seed}:reference_binding", current["repeat_reference"] == {
                "path": config["identity"]["reference"],
                "protocol_sha256": reference_hashes["protocol.json"]})
        for name in ["inputs.json", "anchor_vocab.json"]:
            actual = digest(base / name)
            check(f"seed_{seed}:shared_{name}", actual == reference_hashes[name], actual_sha256=actual)
        splits = current["splits"]
        train, val = splits["train"], splits["val"]
        check(f"seed_{seed}:split_integrity", set(splits) == {"train", "val"}
              and len(train) == len(set(train)) == 435 and len(val) == len(set(val)) == 63
              and not set(train) & set(val), train_count=len(train), validation_count=len(val))
        check(f"seed_{seed}:manifest_membership_and_presence",
              set(manifest) == set(train + val)
              and all(manifest[subject]["split"] == split and any(manifest[subject]["region_present"])
                      for split, subjects in splits.items() for subject in subjects), count=len(manifest))
        check(f"seed_{seed}:common_exclusions", current["excluded"] == protocol["excluded"]
              and len(current["excluded"]) == 3, count=len(current["excluded"]))
        check(f"seed_{seed}:frozen_methods_and_budget", current["methods"] == METHODS
              and current["epochs"] == 10 and current["batch_size"] == 2)
        seed_summary = read(base / "summary.json")
        for method in METHODS:
            run_count += 1
            prefix = f"seed_{seed}:{method}:"
            arm = base / method
            arm_config = read(arm / "config.json")
            history = read(arm / "history.json")
            complete = read(arm / "complete.json")
            best = read(arm / "validation_best_auc.json")
            check(prefix + "configuration", arm_config == {
                "method": method, "protocol": current, "parameter_count": 409056})
            check(prefix + "ten_epochs", [row["epoch"] for row in history] == list(range(1, 11)),
                  count=len(history))
            first_auc = max(history, key=lambda row: row["validation"]["biomarker_macro_auc"])
            first_map = max(history, key=lambda row: row["validation"]["map"])
            check(prefix + "completion_and_first_strict_max",
                  complete["method"] == method and complete["training_steps"] == 2180
                  and complete["selected_epoch"] == first_auc["epoch"]
                  and complete["validation_auc"] == first_auc["validation"]["biomarker_macro_auc"]
                  and complete["validation_map"] == first_auc["validation"]["map"]
                  and complete["validation_replay_equal"] is True and complete["test_evaluated"] is False,
                  selected_epoch=first_auc["epoch"], training_steps=complete["training_steps"])
            check(prefix + "summary_traceability", complete in seed_summary
                  and {"seed": seed, **complete} in combined["runs"])
            expected_views = 435 if method.startswith("whole_roi") else sum(
                sum(manifest[subject]["region_present"]) for subject in train)
            check(prefix + "encoded_view_count", all(row["encoded_training_views"] == expected_views
                                                      for row in history), per_epoch_count=expected_views)
            for row in history:
                epoch_count += 1
                saved = read(arm / f'validation_epoch_{row["epoch"]:03d}.json')
                records = saved["records"]
                masks = {key: value for key, value in records.items() if key != "scores"}
                metrics = saved["metrics"]
                check(prefix + f'epoch_{row["epoch"]:03d}', metrics == row["validation"]
                      and masks == reference_masks and len(records["scores"]) == 189
                      and set(records["subject_ids"]) == set(val)
                      and all(len(scores) == 26 and all(math.isfinite(value) for value in scores)
                              for scores in records["scores"])
                      and metrics["scored_queries"] == 120 and metrics["skipped_no_positive"] == 69,
                      row_count=len(records["scores"]), known_query_count=metrics["scored_queries"],
                      missing_query_count=metrics["skipped_no_positive"])
            selected = read(arm / f'validation_epoch_{first_auc["epoch"]:03d}.json')
            check(prefix + "saved_validation_replay", best["metrics"] == selected["metrics"]
                  and best["records"] == selected["records"] and best["selected_epoch"] == first_auc["epoch"])
            check(prefix + "common_frequency_baseline", read(arm / "frequency_baseline_validation.json")
                  == reference_prior)
            for name, row, metric in [("best_auc.pt", first_auc, "biomarker_macro_auc"),
                                       ("best_map.pt", first_map, "map"),
                                       ("last.pt", history[-1], "biomarker_macro_auc")]:
                checkpoint_count += 1
                checkpoint = torch.load(arm / name, map_location="cpu", weights_only=True)
                parameter_count = sum(value.numel() for key, value in checkpoint["model"].items()
                                      if not key.endswith(("running_mean", "running_var", "num_batches_tracked")))
                check(prefix + name, checkpoint["epoch"] == row["epoch"]
                      and checkpoint["protocol_version"] == current["version"]
                      and checkpoint["selection_metric_name"] == metric
                      and checkpoint["selection_metric"] == row["validation"][metric]
                      and parameter_count == 409056, epoch=checkpoint["epoch"],
                      selection_metric_name=checkpoint["selection_metric_name"],
                      selection_metric=checkpoint["selection_metric"], parameter_count=parameter_count,
                      sha256=digest(arm / name))

    strict_count = invalid_count = test_result_count = lock_count = failed_count = 0
    for base in [REFERENCE, REPEATS]:
        for path in base.rglob("*"):
            if not path.is_file() or ".ipynb_checkpoints" in path.parts:
                continue
            test_result_count += int("test" in path.name.lower())
            lock_count += int(path.suffix == ".lock")
            failed_count += int(path.name == "failed.json")
            if path.suffix == ".json":
                strict_count += 1
                try:
                    read(path)
                except (ValueError, UnicodeError):
                    invalid_count += 1
    check("strict_json", strict_count == 198 and invalid_count == 0,
          count=strict_count, invalid_count=invalid_count)
    check("no_test_named_results", test_result_count == 0, count=test_result_count)
    check("no_locks_or_failed_attempts", lock_count == 0 and failed_count == 0,
          lock_count=lock_count, failed_count=failed_count)
    check("total_artifact_counts", run_count == 12 and epoch_count == 120 and checkpoint_count == 36,
          runs=run_count, new_runs=run_count - 4, epoch_records=epoch_count, checkpoints=checkpoint_count)
    runner = subprocess.check_output(["git", "show", SCIENCE_COMMIT + ":tools/run_pasa_repair_pilot.py"],
                                     cwd=ROOT).decode("utf-8")
    coordinator = subprocess.check_output(["git", "show", COORDINATOR_COMMIT + ":tools/repeat_pasa_repair_pilot.py"],
                                          cwd=ROOT).decode("utf-8")
    check("frozen_code_train_validation_loader_guards",
          "for name in ('train','val')}" in runner
          and "groups={name:[lookup[s] for s in ids] for name,ids in protocol['splits'].items()}" in runner
          and "datasets={name:RepairDataset(cases,manifest,args.roi_size,args.z_slices,augment=name=='train')" in runner
          and "if set(splits)!={'train','val'}" in coordinator)
    receipt = {
        "audit_version": "repair_repeat_integrity_20261021_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "reference_directory": REFERENCE.relative_to(ROOT).as_posix(),
        "repeat_directory": REPEATS.relative_to(ROOT).as_posix(),
        "method": {"science_commit": SCIENCE_COMMIT, "coordinator_commit": COORDINATOR_COMMIT,
                   "source_hash_input": "raw bytes from git show <pinned-commit>:<file>; not working-tree text",
                   "artifact_hash_input": "raw downloaded file bytes",
                   "checkpoint_load": "torch.load(map_location=cpu, weights_only=True)",
                   "validation_comparison": "exact saved records, masks, metrics, and first strict maximum",
                   "images_read_or_training_run": False, "original_results_modified": False},
        "summary": {"passed": all(row["passed"] for row in checks), "check_count": len(checks),
                    "failed_check_count": sum(not row["passed"] for row in checks),
                    "run_count": run_count, "new_run_count": run_count - 4, "epoch_record_count": epoch_count,
                    "checkpoint_count": checkpoint_count, "strict_json_count": strict_count,
                    "common_validation_mask_sha256": object_digest(reference_masks)},
        "limitations": ["No independent MRI inference replay was performed.",
                        "Raw MRI voxels are not hashed in the returned manifests.",
                        "Saved replay equality does not independently rerun the GPU computation.",
                        "No test-named artifacts plus frozen loader guards do not prove all external server actions.",
                        "Three training seeds reuse one development validation split; no independent generalization claim.",
                        "Equal parameter and optimizer-step counts do not imply equal FLOPs or encoded-view counts."],
        "checks": checks,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(RECEIPT), **receipt["summary"]}, ensure_ascii=False))
    return 0 if receipt["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
