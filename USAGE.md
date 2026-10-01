# Command reference

Use Python 3.12 in an isolated environment. `requirements.txt` pins TensorFlow CPU and Matplotlib; `environment.lock.txt` records the complete executed environment.

```sh
python -m pip install -r requirements.txt
python prepare.py
python generate.py --checkpoint runs/gqa-2 --prompt "ROMEO:" --characters 240
```

Only training-vocabulary characters are accepted. Add `--mode reference` to disable the KV cache. Temperature is 0.8 and the default sampling seed is 123. Prompts longer than 128 characters are cropped for inference, while the original prompt stays in the returned text.

Create a new converted checkpoint without replacing the source:

```sh
python convert_checkpoint.py runs/mha-4 converted-gqa --kv-heads 2
```

The destination must not exist. Mean pooling is not quality preserving: the recorded pooled checkpoint failed badly without uptraining. Evaluate a converted model before using it.

Estimate compact cache payload:

```sh
python plan_cache.py --checkpoint runs/gqa-2 --batch 8 --tokens 128
```

This estimates float32 K/V tensor bytes, not total RAM or maximum safe serving capacity. It refuses context lengths beyond the trained configuration.

The Python runtime can be reused without the sampler:

```python
from cachecraft import Engine, load_checkpoint

engine = Engine(load_checkpoint("runs/gqa-2"))
session = engine.start([30, 27, 25, 17, 27, 10, 0])  # ROMEO:\n
next_token = int(session.logits[0].numpy().argmax())
following_logits = session.advance([[next_token]])
```

`engine.start` accepts a nonempty integer `[time]` or `[batch, time]` array. A session consumes one token per batch member and returns `[batch, vocabulary]` logits. Create a separate mutable session for each request; sessions are not shared or thread-safe. Equal-length batches work, but padding and variable-length batches are not supported. At the context limit, a session re-prefills the full latest window to preserve the reference decoder's semantics.

The conversion function is available separately:

```python
from cachecraft import load_checkpoint, pool_kv_heads

original = load_checkpoint("runs/mha-4")
converted = pool_kv_heads(original, kv_heads=2)
```

Run verification and inspect the recorded study:

```sh
python -m unittest discover -s tests -v
python verify.py
python inspect_results.py
python benchmark.py --trials 20
```

The verifier and benchmark refuse to replace existing audit/benchmark outputs unless `--overwrite` is explicit. Plot generation reads the measured artifacts; it does not train or change checkpoints.

The training command is:

```sh
python study.py --steps 1000 --batch 16 --seed 42
```

The trainer refuses to overwrite an existing `runs/` directory. Preserve the included results and use a separate working checkout for another experiment. Configuration changes create a different comparison; do not mix their scores with the published results.

Only load checkpoints from sources you trust. The included models use local files and require no credentials or inference service.
