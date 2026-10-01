"""Generate from a local trained attention variant."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import json
from pathlib import Path
import tensorflow as tf
from cachecraft import Engine, load_checkpoint, generate_text

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/gqa-2")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--characters", type=int, default=240)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=("cached", "reference"), default="cached")
    args = parser.parse_args()
    if args.characters < 1 or not args.prompt:
        parser.error("a nonempty prompt and positive character count are required")
    vocabulary = json.loads((ROOT / "data/manifest.json").read_text())["vocabulary"]
    if set(args.prompt) - set(vocabulary):
        parser.error("prompt contains unseen characters")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print(generate_text(Engine(load_checkpoint(args.checkpoint)), vocabulary, args.prompt,
                        args.characters, args.seed, cached=args.mode == "cached"))


if __name__ == "__main__":
    main()
