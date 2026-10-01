"""Train three attention variants on identical windows, then evaluate once."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cachecraft import Config, Decoder, Engine, generate_text, load_checkpoint, pool_kv_heads

ROOT = Path(__file__).resolve().parent


def read_data():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    raw = (ROOT / "data/input.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("corpus SHA-256 does not match the pinned source")
    text = raw.decode("utf-8")
    lookup = {character: i for i, character in enumerate(manifest["vocabulary"])}
    boundaries = manifest["split_offsets"]
    pieces = tuple(np.array([lookup[character] for character in text[start:end]], np.int32)
                   for start, end in zip(boundaries[:-1], boundaries[1:]))
    return manifest, pieces


def windows(piece, context):
    starts = np.arange(0, len(piece) - context, context)
    indices = starts[:, None] + np.arange(context)[None, :]
    return piece[indices], piece[indices + 1]


def evaluate(model, piece, limit=None):
    inputs, targets = windows(piece, model.config.context)
    if limit is not None:
        inputs, targets = inputs[:limit], targets[:limit]
    @tf.function(input_signature=[tf.TensorSpec([None, model.config.context], tf.int32)])
    def forward(ids):
        return model(ids, training=False)
    losses, correct = [], []
    for start in range(0, len(inputs), 16):
        logits = forward(inputs[start:start+16])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets[start:start+16], logits=logits)
        losses.extend(tf.reduce_mean(loss, axis=1).numpy().tolist())
        correct.extend(np.mean(logits.numpy().argmax(-1) == targets[start:start+16], axis=1).tolist())
    mean = float(np.mean(losses))
    return {"nll_nats_per_character": mean, "perplexity": float(np.exp(mean)),
            "next_character_accuracy": float(np.mean(correct)),
            "evaluated_characters": int(inputs.size), "windows": len(inputs),
            "window_nll_sd": float(np.std(losses, ddof=1))}, np.array(losses)


def error_windows(test, manifest, block_losses, context):
    vocabulary = manifest["vocabulary"]
    def row(index):
        begin = int(index) * context
        return {"window": int(index), "source_character_offset": manifest["split_offsets"][2] + begin,
                "input": "".join(vocabulary[token] for token in test[begin:begin+context]),
                "nll_by_model": {name: float(values[index]) for name, values in block_losses.items()}}
    gqa = block_losses["gqa-2"]
    degradation = gqa - block_losses["mha-4"]
    return {"highest_gqa_nll": [row(index) for index in np.argsort(gqa)[-5:][::-1]],
            "largest_gqa_minus_mha_nll": [dict(row(index), gqa_minus_mha_nll=float(degradation[index]))
                                        for index in np.argsort(degradation)[-5:][::-1]],
            "note": "Worst windows are diagnostic examples, not an independent test or IID confidence interval."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch) < 1:
        parser.error("positive steps and batch required")
    output = ROOT / "runs"
    if output.exists():
        parser.error("runs/ already exists; this study never overwrites an earlier experiment")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    output.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    manifest, (train, validation, test) = read_data()
    context = 128
    rng = np.random.default_rng(args.seed)
    schedule = rng.integers(0, len(train) - context, size=(args.steps, args.batch))
    np.save(output / "training_starts.npy", schedule)
    histories, metrics = [], {}
    variants = (("mha-4", 4), ("gqa-2", 2), ("mqa-1", 1))
    for name, heads in variants:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        config = Config(vocabulary=len(manifest["vocabulary"]), kv_heads=heads)
        model = Decoder(config)
        model(tf.zeros([1, context], tf.int32))
        directory = output / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        optimizer = tf.keras.optimizers.Adam(learning_rate=.001, global_clipnorm=1.)
        optimizer.build(model.trainable_variables)
        @tf.function(input_signature=[tf.TensorSpec([args.batch, context], tf.int32),
                                      tf.TensorSpec([args.batch, context], tf.int32)])
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        best, best_step = float("inf"), 0
        began = time.perf_counter()
        for step, starts in enumerate(schedule, 1):
            indices = starts[:, None] + np.arange(context)[None, :]
            loss = float(train_step(train[indices], train[indices + 1]))
            if step == 1 or step % 100 == 0 or step == args.steps:
                subset, _ = evaluate(model, validation, limit=128)
                value = subset["nll_nats_per_character"]
                row = {"model": name, "step": step, "training_nll": loss,
                       "validation_subset_nll": value, "elapsed_seconds": time.perf_counter() - began}
                histories.append(row)
                print(json.dumps(row), flush=True)
                if value < best:
                    best, best_step = value, step
                    model.save_weights(directory / "model.weights.h5")
        seconds = time.perf_counter() - began
        model.load_weights(directory / "model.weights.h5")
        full_validation, _ = evaluate(model, validation)
        metrics[name] = {"kind": "trained from scratch", "kv_heads": heads,
                         "parameters": model.count_params(), "selected_step": best_step,
                         "training_steps": args.steps, "training_characters": args.steps * args.batch * context,
                         "training_seconds_including_validation_and_saving": seconds,
                         "validation": full_validation}
        (output / "results.partial.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Pool the selected MHA checkpoint, with no optimizer steps or test-set choice.
    source = load_checkpoint(output / "mha-4")
    converted = pool_kv_heads(source, 2)
    converted.save_checkpoint(output / "gqa-pooled")
    conversion = {"source_checkpoint": "runs/mha-4", "source_checkpoint_sha256": hashlib.sha256((output / "mha-4/model.weights.h5").read_bytes()).hexdigest(),
                  "source_kv_heads": 4, "target_kv_heads": 2,
                  "method": "Contiguous K/V projection head mean pooling; other weights unchanged.",
                  "additional_training_steps": 0,
                  "warning": "Not uptrained. This is a conversion diagnostic, not a trained GQA substitute."}
    (output / "gqa-pooled/conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    pooled_validation, _ = evaluate(converted, validation)
    metrics["gqa-pooled"] = {"kind": "MHA checkpoint conversion, no uptraining", "kv_heads": 2,
                              "parameters": converted.count_params(), "inherited_selected_step": metrics["mha-4"]["selected_step"],
                              "additional_training_steps": 0, "validation": pooled_validation}
    selected = min((name for name, _ in variants), key=lambda name: metrics[name]["validation"]["nll_nats_per_character"])
    # Only now evaluate every fixed checkpoint on the final held-out split.
    block_losses = {}
    for name in metrics:
        model = load_checkpoint(output / name)
        heldout, losses = evaluate(model, test)
        np.save(output / name / "test_window_losses.npy", losses)
        block_losses[name] = losses
        engine = Engine(model)
        sample = generate_text(engine, manifest["vocabulary"], "ROMEO:\n", count=240, seed=123)
        (output / name / "sample.txt").write_text(sample + "\n")
        cached = engine.start(np.ones([1, context], np.int32))
        actual_payload = cached.cache_bytes()
        formula = model.config.cache_payload_bytes(1, context)
        if actual_payload != formula:
            raise ValueError("cache formula and actual tensor payload differ")
        metrics[name].update({"test": heldout, "float32_kv_payload_bytes_batch1_context128": actual_payload})
        print(json.dumps({"model": name, "validation": metrics[name]["validation"], "test": heldout,
                          "parameters": model.count_params(), "actual_kv_bytes": actual_payload}), flush=True)
    counts = np.bincount(train, minlength=len(manifest["vocabulary"])).astype(float) + 1
    probabilities = counts / counts.sum()
    _, labels = windows(test, context)
    baseline = float(-np.log(probabilities[labels]).mean())
    result = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "steps": args.steps, "batch": args.batch, "context": context,
              "source_sha256": manifest["sha256"], "selected_from_scratch_model_by_full_validation": selected,
              "models": metrics, "training_only_unigram": {"test_nll": baseline, "test_perplexity": float(np.exp(baseline))},
              "experimental_controls": "Same sampled windows/order, seed, width, query heads, layers, FFN, optimizer and steps. "
                                       "KV count changes parameter count; this is not a parameter-matched or compute-matched comparison.",
              "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__, "platform": platform.platform(),
                          "intra_threads": 2, "inter_threads": 2, "oneDNN": False,
                          "devices": [str(device) for device in tf.config.list_physical_devices()]},
              "limitations": ["One seed, character-level corpus, 128-character context.",
                              "Contiguous held-out windows are correlated, not IID replicates.",
                              "Smaller cache payload is not total RAM usage or a guaranteed latency improvement.",
                              "Head pooling without uptraining can seriously damage quality."]}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "error_analysis.json").write_text(json.dumps(error_windows(test, manifest, block_losses, context), indent=2) + "\n")
    with (output / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    print(json.dumps({"selected_by_validation": selected, "completed_at_utc": result["completed_at_utc"]}), flush=True)


if __name__ == "__main__":
    main()
