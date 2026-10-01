"""Compiled, request-local inference with compact grouped-query caches."""
import numpy as np
import tensorflow as tf


def token_array(ids, vocabulary):
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("expected a nonempty integer [batch, time] array")
    if np.any(ids < 0) or np.any(ids >= vocabulary):
        raise ValueError("token ID outside vocabulary")
    return ids.astype(np.int32, copy=False)


class Engine:
    def __init__(self, model):
        self.model = model
        config = model.config
        if not model.built:
            model(tf.zeros([1, 1], tf.int32))
        if model.compute_dtype != "float32":
            raise ValueError("this runtime is tested with float32")
        prefix = tf.TensorSpec([None, None], tf.int32)
        token = tf.TensorSpec([None, 1], tf.int32)
        kv = tf.TensorSpec([None, config.kv_heads, None, config.head_width], tf.float32)
        caches = tuple((kv, kv) for _ in model.blocks)
        self.full = tf.function(lambda ids: model(ids, training=False)[:, -1], input_signature=[prefix])
        self.prefill = tf.function(model.prefill, input_signature=[prefix])
        self.decode = tf.function(model.decode_step, input_signature=[token, caches])

    def start(self, ids, cached=True):
        ids = token_array(ids, self.model.config.vocabulary)
        return Session(self, ids[:, -self.model.config.context:], bool(cached))

    def traces(self):
        return {name: function.experimental_get_tracing_count()
                for name, function in (("full", self.full), ("prefill", self.prefill), ("decode", self.decode))}


class Session:
    """One mutable decoding request; separate sessions do not share KV tensors."""
    def __init__(self, engine, ids, cached):
        self.engine, self.window, self.cached = engine, ids.copy(), cached
        self.resets = 0
        if cached:
            self.logits, self.caches = engine.prefill(self.window)
        else:
            self.logits, self.caches = engine.full(self.window), None

    def advance(self, ids):
        ids = token_array(ids, self.engine.model.config.vocabulary)
        if ids.shape != (len(self.window), 1):
            raise ValueError("consume exactly one token for each batch member")
        context = self.engine.model.config.context
        full = self.window.shape[1] == context
        window = np.concatenate([self.window, ids], axis=1)[:, -context:]
        if not self.cached:
            logits, caches = self.engine.full(window), None
        elif full:
            # Evicting KV alone would retain information from outside the window.
            logits, caches = self.engine.prefill(window)
        else:
            logits, caches = self.engine.decode(ids, self.caches)
        self.window, self.logits, self.caches = window, logits, caches
        if self.cached and full:
            self.resets += 1
        return logits

    def cache_bytes(self):
        if self.caches is None:
            return 0
        return sum(int(tf.size(tensor)) * tensor.dtype.size for pair in self.caches for tensor in pair)


def generate_text(engine, vocabulary, prompt, count=240, seed=123, temperature=.8, cached=True):
    if not prompt or type(count) is not int or count < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonempty prompt, nonnegative integer count and positive temperature required")
    if len(vocabulary) != engine.model.config.vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary does not match model")
    lookup = {character: i for i, character in enumerate(vocabulary)}
    if set(prompt) - set(lookup):
        raise ValueError("prompt includes characters outside the training vocabulary")
    ids = [lookup[character] for character in prompt]
    if not count:
        return prompt
    session = engine.start(ids, cached)
    rng = np.random.default_rng(seed)
    for step in range(count):
        logits = session.logits[0].numpy() / temperature
        logits -= logits.max()
        probabilities = np.exp(logits) / np.exp(logits).sum()
        token = int(rng.choice(len(vocabulary), p=probabilities))
        ids.append(token)
        if step + 1 < count:
            session.advance([[token]])
    return "".join(vocabulary[token] for token in ids)
