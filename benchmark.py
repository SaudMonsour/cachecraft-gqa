"""Benchmark actual compact cache tensors and synchronized CPU inference."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cachecraft import Engine, load_checkpoint
from study import read_data

ROOT = Path(__file__).resolve().parent


def stats(samples):
    return {"p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples}


def request(engine, prompt, continuation, cached):
    started = time.perf_counter_ns()
    session = engine.start(prompt, cached)
    session.logits.numpy()
    for token in continuation:
        session.advance([[int(token)]]).numpy()
    return (time.perf_counter_ns() - started) / 1e6, session


def check(engine, prompt, continuation):
    cached, reference = engine.start(prompt), engine.start(prompt, False)
    maximum, greedy_match = 0., True
    for step in range(len(continuation) + 1):
        left, right = cached.logits.numpy(), reference.logits.numpy()
        np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-5)
        maximum = max(maximum, float(np.max(np.abs(left - right))))
        greedy_match &= bool(np.array_equal(left.argmax(-1), right.argmax(-1)))
        if step < len(continuation):
            token = [[int(continuation[step])]]
            cached.advance(token)
            reference.advance(token)
    if not greedy_match:
        raise ValueError("greedy predictions differ on a benchmark workload")
    return {"maximum_absolute_logit_difference": maximum, "all_greedy_predictions_match": greedy_match,
            "predictions_checked": len(continuation) + 1, "cache_resets": cached.resets,
            "final_kv_payload_bytes": cached.cache_bytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.trials < 5:
        parser.error("at least five measured trials are required")
    output = ROOT / "runs/benchmark.json"
    if output.exists() and not args.overwrite:
        parser.error("benchmark exists; use --overwrite to replace only this benchmark")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    result = json.loads((ROOT / "runs/metrics.json").read_text())
    manifest, (_, _, test) = read_data()
    workloads = {"growing-prefix": (test[:32], test[32:96]),
                 "full-window": (test[:128], test[128:160])}
    measured = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "models": {},
                "source_sha256": manifest["sha256"], "measured_trials_per_mode": args.trials,
                "warmup_requests_per_mode": 3,
                "method": "Whole request, teacher-forced identical IDs, CPU batch 1. Includes prefill, "
                          "Python orchestration and synchronized last logits for every update. Excludes loading, "
                          "graph tracing, tokenizer and sampling. Reference emits full prefix logits; "
                          "cached prefill emits last logits and compact K/V tensors. Modes alternate first position.",
                "cache_measurement": "Sum of actual float32 K/V tensor element counts times scalar size; "
                                     "not process RSS, device allocation, weights, attention workspaces or temporary copies.",
                "workloads": {name: {"prompt_ids": prompt.tolist(), "continuation_ids": continuation.tolist()}
                              for name, (prompt, continuation) in workloads.items()},
                "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__,
                            "intra_threads": 2, "inter_threads": 2, "oneDNN": False,
                            "platform": platform.platform(), "devices": [str(device) for device in tf.config.list_physical_devices()]}}
    for name in result["models"]:
        engine = Engine(load_checkpoint(ROOT / "runs" / name))
        item = {"kv_heads": engine.model.config.kv_heads, "prefill": {}, "requests": {}}
        for length in (16, 64, 128):
            prompt = test[:length]
            for _ in range(5):
                last, _ = engine.prefill(prompt[None])
                last.numpy()
            samples = []
            for _ in range(args.trials):
                start = time.perf_counter_ns()
                last, caches = engine.prefill(prompt[None])
                last.numpy()
                samples.append((time.perf_counter_ns() - start) / 1e6)
            actual_bytes = sum(int(tf.size(tensor)) * tensor.dtype.size for pair in caches for tensor in pair)
            expected = engine.model.config.cache_payload_bytes(1, length)
            if actual_bytes != expected:
                raise ValueError("compact cache tensor payload differs from formula")
            item["prefill"][str(length)] = {**stats(samples), "actual_kv_payload_bytes": actual_bytes,
                                           "formula_kv_payload_bytes": expected,
                                           "cache_shapes": [[tensor.shape.as_list() for tensor in pair] for pair in caches]}
        for workload, (prompt, continuation) in workloads.items():
            equivalence = check(engine, prompt, continuation)
            for _ in range(3):
                for cached in (False, True):
                    request(engine, prompt, continuation, cached)
            samples = {"reference": [], "cached": []}
            for trial in range(args.trials):
                for cached in ((False, True) if trial % 2 == 0 else (True, False)):
                    milliseconds, _ = request(engine, prompt, continuation, cached)
                    samples["cached" if cached else "reference"].append(milliseconds)
            item["requests"][workload] = {"reference": stats(samples["reference"]),
                                          "cached": stats(samples["cached"]), **equivalence,
                                          "reference_over_cached_median_ratio": float(np.median(samples["reference"]) / np.median(samples["cached"]))}
            print(json.dumps({"model": name, "workload": workload,
                              "reference_ms": item["requests"][workload]["reference"]["p50_ms"],
                              "cached_ms": item["requests"][workload]["cached"]["p50_ms"], **equivalence}), flush=True)
        item["graph_traces"] = engine.traces()
        measured["models"][name] = item
    measured["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(measured, indent=2) + "\n")


if __name__ == "__main__":
    main()
