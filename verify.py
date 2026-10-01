"""Replay the executed study, conversion, compact caches, samples, and tests."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cachecraft import Engine, load_checkpoint, pool_kv_heads, generate_text
from study import read_data, evaluate, windows, error_windows

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from case_ids(item)
        else:
            yield item.id()


def check_stats(values, expected, count):
    if len(values) != count or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("invalid or missing timing observations")
    for percentile in (50, 95):
        np.testing.assert_allclose(np.percentile(values, percentile), expected[f"p{percentile}_ms"])


def check_error_windows(actual, expected, path="error_analysis"):
    """Require exact examples/structure, allowing float32 reduction roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"error analysis structure differs: {path}")
        return max((check_error_windows(actual[key], value, f"{path}.{key}")
                    for key, value in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"error analysis examples differ: {path}")
        return max((check_error_windows(left, right, f"{path}[{index}]")
                    for index, (left, right) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0, err_msg=path)
        return abs(float(actual) - expected)
    if actual != expected:
        raise ValueError(f"error analysis example differs: {path}")
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = ROOT / "audit.json"
    if output.exists() and not args.overwrite:
        parser.error("audit exists; --overwrite explicitly replaces only this audit")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    manifest, (train, validation, test) = read_data()
    metrics = json.loads((ROOT / "runs/metrics.json").read_text())
    benchmark = json.loads((ROOT / "runs/benchmark.json").read_text())
    raw = (ROOT / "data/input.txt").read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != manifest["source_git_blob_sha"] or benchmark["source_sha256"] != manifest["sha256"]:
        raise ValueError("corpus provenance differs")
    rng = np.random.default_rng(metrics["seed"])
    expected_starts = rng.integers(0, len(train) - metrics["context"], size=(metrics["steps"], metrics["batch"]))
    np.testing.assert_array_equal(np.load(ROOT / "runs/training_starts.npy"), expected_starts)
    counts = np.bincount(train, minlength=len(manifest["vocabulary"])).astype(float) + 1
    _, targets = windows(test, metrics["context"])
    baseline = float(-np.log((counts/counts.sum())[targets]).mean())
    np.testing.assert_allclose(baseline, metrics["training_only_unigram"]["test_nll"])
    checked, losses = {}, {}
    for name, recorded in metrics["models"].items():
        model = load_checkpoint(ROOT / "runs" / name)
        engine = Engine(model)
        if model.count_params() != recorded["parameters"]:
            raise ValueError("parameter count differs")
        validation_replay, _ = evaluate(model, validation)
        replay, window_losses = evaluate(model, test)
        test_difference = abs(replay["nll_nats_per_character"] - recorded["test"]["nll_nats_per_character"])
        validation_difference = abs(validation_replay["nll_nats_per_character"] - recorded["validation"]["nll_nats_per_character"])
        per_window_difference = float(np.max(np.abs(np.load(ROOT / "runs" / name / "test_window_losses.npy") - window_losses)))
        if max(test_difference, validation_difference) > 1e-6 or per_window_difference > 1e-5:
            raise ValueError(f"checkpoint scores differ: {name}")
        losses[name] = window_losses
        expected_sample = (ROOT / "runs" / name / "sample.txt").read_text().removesuffix("\n")
        for cached in (False, True):
            sample = generate_text(engine, manifest["vocabulary"], "ROMEO:\n", 240, 123, cached=cached)
            if sample != expected_sample:
                raise ValueError(f"sample replay differs: {name}, cached={cached}")
        observed = benchmark["models"][name]
        for length in (16, 64, 128):
            session = engine.start(test[:length])
            measurements = observed["prefill"][str(length)]
            payload = session.cache_bytes()
            if payload != model.config.cache_payload_bytes(1, length) or payload != measurements["actual_kv_payload_bytes"]:
                raise ValueError("cache formula does not match real compact tensors")
            shapes = [[tensor.shape.as_list() for tensor in pair] for pair in session.caches]
            if shapes != measurements["cache_shapes"]:
                raise ValueError("cache shape differs")
            check_stats(measurements["samples_ms"], measurements, benchmark["measured_trials_per_mode"])
        maximum, greedy_match = 0., True
        for workload, inputs in benchmark["workloads"].items():
            length = 32 if workload == "growing-prefix" else 128
            continuation_length = 64 if workload == "growing-prefix" else 32
            np.testing.assert_array_equal(inputs["prompt_ids"], test[:length])
            np.testing.assert_array_equal(inputs["continuation_ids"], test[length:length+continuation_length])
            left = engine.start(inputs["prompt_ids"])
            right = engine.start(inputs["prompt_ids"], cached=False)
            for step in range(continuation_length + 1):
                a, b = left.logits.numpy(), right.logits.numpy()
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
                maximum = max(maximum, float(np.max(np.abs(a-b))))
                greedy_match &= bool(np.array_equal(a.argmax(-1), b.argmax(-1)))
                if step < continuation_length:
                    token = [[inputs["continuation_ids"][step]]]
                    left.advance(token)
                    right.advance(token)
            item = observed["requests"][workload]
            if left.resets != item["cache_resets"] or left.cache_bytes() != item["final_kv_payload_bytes"]:
                raise ValueError("window-reset accounting differs")
            for mode in ("reference", "cached"):
                check_stats(item[mode]["samples_ms"], item[mode], benchmark["measured_trials_per_mode"])
            np.testing.assert_allclose(item["reference"]["p50_ms"] / item["cached"]["p50_ms"], item["reference_over_cached_median_ratio"])
        if not greedy_match or engine.traces() != {"full": 1, "prefill": 1, "decode": 1}:
            raise ValueError("greedy predictions or graph trace count differ")
        if model.config.cache_payload_bytes(1, 128) != recorded["float32_kv_payload_bytes_batch1_context128"]:
            raise ValueError("recorded full-window payload differs")
        checked[name] = {"test_nll_difference": test_difference, "validation_nll_difference": validation_difference,
                         "maximum_window_nll_difference": per_window_difference,
                         "cached_and_reference_seeded_sample_match": True,
                         "maximum_absolute_cached_logit_difference": maximum,
                         "all_checked_greedy_predictions_match": greedy_match, "graph_traces": engine.traces()}
        print(json.dumps({"model": name, **checked[name]}), flush=True)
    source = load_checkpoint(ROOT / "runs/mha-4")
    converted = pool_kv_heads(source, 2)
    stored = load_checkpoint(ROOT / "runs/gqa-pooled")
    for actual, expected in zip(converted.get_weights(), stored.get_weights()):
        np.testing.assert_array_equal(actual, expected)
    conversion = json.loads((ROOT / "runs/gqa-pooled/conversion.json").read_text())
    if conversion["source_checkpoint_sha256"] != sha256(ROOT / "runs/mha-4/model.weights.h5"):
        raise ValueError("conversion points to a different source checkpoint")
    expected_errors = error_windows(test, manifest, losses, metrics["context"])
    stored_errors = json.loads((ROOT / "runs/error_analysis.json").read_text())
    error_window_difference = check_error_windows(expected_errors, stored_errors)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    ids = list(case_ids(suite))
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    if not outcome.wasSuccessful() or outcome.skipped:
        raise ValueError("architecture test failed or was skipped")
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    if environment.strip() != (ROOT / "environment.lock.txt").read_text().strip():
        raise ValueError("executed environment differs from lock")
    readme = (ROOT / "README.md").read_text()
    images = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", readme)
    if not readme.startswith("# CacheCraft GQA\n") or len(images) != 6 or any(not (ROOT / image).is_file() for image in images):
        raise ValueError("README title or figure links invalid")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                   and ".git" not in path.parts and path.name not in {"input.txt", "results.partial.json", "audit.json"})
    audit = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
             "tensorflow": tf.__version__, "source_sha256": manifest["sha256"], "source_git_blob_sha": blob,
             "checkpoint_replay": checked, "conversion_weight_replay_exact": True,
             "sampled_training_windows_replay_exact": True,
             "error_window_examples_replay_exact": True,
             "error_window_nll_absolute_tolerance": 1e-5,
             "maximum_error_window_nll_difference": error_window_difference,
             "benchmark_statistics_recomputed": True, "cache_payload_formula_matches_tensors": True,
             "tests": {"passed": outcome.testsRun, "skipped": len(outcome.skipped), "test_ids": ids},
             "artifact_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files},
             "attribution": "AI-assisted implementation and automated study verification for Saud Alotaibi."}
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"verified": True, "tests_passed": outcome.testsRun, "artifact_count": len(files)}), flush=True)


if __name__ == "__main__":
    main()
