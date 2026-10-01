"""Create a separate checkpoint with mean-pooled KV projections."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import argparse
import json
from pathlib import Path
from cachecraft import load_checkpoint, pool_kv_heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--kv-heads", type=int, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination exists; conversion never replaces a checkpoint")
    source = load_checkpoint(args.source)
    try:
        converted = pool_kv_heads(source, args.kv_heads)
    except ValueError as error:
        parser.error(str(error))
    converted.save_checkpoint(args.destination)
    (args.destination / "conversion.json").write_text(json.dumps({
        "source": str(args.source), "source_kv_heads": source.config.kv_heads,
        "target_kv_heads": converted.config.kv_heads,
        "method": "Contiguous K/V head mean pooling; other weights unchanged; no uptraining.",
        "warning": "Quality is not preserved by this conversion; evaluate before use."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
