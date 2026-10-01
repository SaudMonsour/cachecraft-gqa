"""Download and verify a pinned, unmodified public character-language corpus."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
SOURCE = f"https://raw.githubusercontent.com/karpathy/char-rnn/{SOURCE_COMMIT}/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
BLOB_SHA = "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"


def main():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    path = directory / "input.txt"
    if not path.exists():
        with urlopen(SOURCE, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("download differs from the pinned source")
        path.write_bytes(data)
    raw = path.read_bytes()
    git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if hashlib.sha256(raw).hexdigest() != SHA256 or git_blob != BLOB_SHA:
        raise ValueError("corpus does not match the pinned GitHub blob and SHA-256")
    text = raw.decode("utf-8")
    train_end, validation_end = int(.8 * len(text)), int(.9 * len(text))
    vocabulary = sorted(set(text[:train_end]))
    if set(text[train_end:]) - set(vocabulary):
        raise ValueError("held-out characters outside the training-only vocabulary")
    manifest = {"name": "Tiny Shakespeare", "source": SOURCE, "source_commit": SOURCE_COMMIT,
                "source_git_blob_sha": BLOB_SHA, "sha256": SHA256,
                "bytes": len(raw), "characters": len(text), "vocabulary": vocabulary,
                "split_offsets": [0, train_end, validation_end, len(text)],
                "split_policy": "Contiguous 80/10/10; no input/target window crosses a split boundary.",
                "limitations": "Repeated phrases across plays can occur; this is not a deduplicated split.",
                "redistribution": "Corpus is not included; pinned original Shakespeare text is downloaded unchanged."}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source_verified": True, "sha256": SHA256, "git_blob_sha": git_blob,
                      "characters": len(text), "vocabulary": len(vocabulary)}, indent=2))


if __name__ == "__main__":
    main()
