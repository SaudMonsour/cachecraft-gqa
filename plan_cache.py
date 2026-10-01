"""Estimate KV tensor payload for a checkpoint, not total device memory."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
import json
from pathlib import Path
from cachecraft import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "runs/gqa-2")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    config = Config(**json.loads((args.checkpoint / "config.json").read_text()))
    try:
        payload = config.cache_payload_bytes(args.batch, args.tokens)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"batch": args.batch, "tokens": args.tokens, "kv_heads": config.kv_heads,
                      "float32_kv_tensor_bytes": payload, "float32_kv_tensor_KiB": payload / 1024,
                      "excludes": ["model weights", "attention scores", "temporary tensors", "graph storage",
                                   "allocator overhead", "other requests"]}, indent=2))


if __name__ == "__main__":
    main()
