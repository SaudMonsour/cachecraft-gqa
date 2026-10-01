"""Grouped-query attention with compact KV tensors, RoPE, and a dense decoder."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import tensorflow as tf


@dataclass(frozen=True)
class Config:
    vocabulary: int = 65
    context: int = 128
    width: int = 64
    query_heads: int = 4
    kv_heads: int = 2
    layers: int = 2
    hidden: int = 128

    def __post_init__(self):
        dimensions = (self.vocabulary, self.context, self.width, self.query_heads,
                      self.kv_heads, self.layers, self.hidden)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("dimensions must be positive integers")
        if self.width % self.query_heads or self.head_width % 2:
            raise ValueError("query heads must divide width into even-dimensional heads")
        if self.query_heads % self.kv_heads:
            raise ValueError("KV heads must divide query heads")

    @property
    def head_width(self):
        return self.width // self.query_heads

    def cache_payload_bytes(self, batch, tokens, bytes_per_scalar=4):
        """K and V tensor payload only; no weights, workspaces, or allocator costs."""
        if any(type(value) is not int or value < 1 for value in (batch, tokens, bytes_per_scalar)):
            raise ValueError("batch, tokens and scalar size must be positive integers")
        if tokens > self.context:
            raise ValueError("token count exceeds the trained context")
        return 2 * self.layers * batch * tokens * self.kv_heads * self.head_width * bytes_per_scalar


def rotary(value, offset=0):
    """Adjacent-pair rotation on [batch, head, token, head_width]."""
    dimension = tf.shape(value)[-1]
    positions = tf.cast(tf.range(tf.shape(value)[2]) + offset, value.dtype)
    frequencies = tf.pow(tf.cast(10000., value.dtype),
                         -tf.cast(tf.range(0, dimension, 2), value.dtype) / tf.cast(dimension, value.dtype))
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = tf.cos(angles), tf.sin(angles)
    even, odd = value[..., ::2], value[..., 1::2]
    pairs = tf.stack([even * cosine - odd * sine, even * sine + odd * cosine], -1)
    return tf.reshape(pairs, tf.shape(value))


class GroupedAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.query = tf.keras.layers.Dense(config.width, use_bias=False, name="query")
        self.key = tf.keras.layers.Dense(config.kv_heads * config.head_width, use_bias=False, name="key")
        self.value = tf.keras.layers.Dense(config.kv_heads * config.head_width, use_bias=False, name="value")
        self.output_projection = tf.keras.layers.Dense(config.width, use_bias=False, name="output")

    def project(self, x, offset=0):
        config = self.config
        batch, time = tf.shape(x)[0], tf.shape(x)[1]
        def split(projection, heads):
            return tf.transpose(tf.reshape(projection(x), [batch, time, heads, config.head_width]), [0, 2, 1, 3])
        query = rotary(split(self.query, config.query_heads), offset)
        key = rotary(split(self.key, config.kv_heads), offset)
        value = split(self.value, config.kv_heads)
        return query, key, value

    def attend(self, query, keys, values, offset=0):
        config = self.config
        batch, time = tf.shape(query)[0], tf.shape(query)[2]
        groups = config.query_heads // config.kv_heads
        # Query groups share compact K/V heads; there is no tf.repeat on K/V.
        grouped_query = tf.reshape(query, [batch, config.kv_heads, groups, time, config.head_width])
        scores = tf.einsum("bhgtd,bhsd->bhgts", grouped_query, keys) * (config.head_width ** -.5)
        query_positions = tf.range(time) + offset
        key_positions = tf.range(tf.shape(keys)[2])
        mask = key_positions[None, :] <= query_positions[:, None]
        scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        attended = tf.einsum("bhgts,bhsd->bhgtd", tf.nn.softmax(scores, axis=-1), values)
        attended = tf.reshape(attended, [batch, config.query_heads, time, config.head_width])
        attended = tf.reshape(tf.transpose(attended, [0, 2, 1, 3]), [batch, time, config.width])
        return self.output_projection(attended)

    def call(self, x, cache=None):
        offset = tf.shape(cache[0])[2] if cache is not None else 0
        query, key, value = self.project(x, offset)
        if cache is not None:
            key = tf.concat([cache[0], key], axis=2)
            value = tf.concat([cache[1], value], axis=2)
        return self.attend(query, key, value, offset), (key, value)


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.up = tf.keras.layers.Dense(config.hidden, use_bias=False)
        self.gate = tf.keras.layers.Dense(config.hidden, use_bias=False)
        self.down = tf.keras.layers.Dense(config.width, use_bias=False)

    def call(self, x):
        return self.down(self.up(x) * tf.nn.silu(self.gate(x)))


class Block(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = GroupedAttention(config)
        self.ffn = SwiGLU(config)

    def call(self, x, cache=None):
        delta, updated = self.attention(self.norm_attention(x), cache=cache)
        x = x + delta
        return x + self.ffn(self.norm_ffn(x)), updated


class Decoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocabulary, config.width)
        self.blocks = [Block(config, name=f"block_{i}") for i in range(config.layers)]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def validate_ids(self, ids):
        tf.debugging.assert_rank(ids, 2)
        tf.debugging.assert_positive(tf.shape(ids)[0], message="empty batch")
        tf.debugging.assert_positive(tf.shape(ids)[1], message="empty sequence")
        tf.debugging.assert_greater_equal(ids, 0)
        tf.debugging.assert_less(ids, self.config.vocabulary)

    def hidden(self, ids, caches=None):
        x = self.embedding(ids)
        updated = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cache=None if caches is None else caches[i])
            updated.append(cache)
        return x, tuple(updated)

    def logits(self, x):
        return tf.einsum("btd,vd->btv", self.final_norm(x), self.embedding.embeddings)

    def call(self, ids, training=False):
        self.validate_ids(ids)
        tf.debugging.assert_less_equal(tf.shape(ids)[1], self.config.context)
        x, _ = self.hidden(ids)
        return self.logits(x)

    def prefill(self, ids):
        self.validate_ids(ids)
        tf.debugging.assert_less_equal(tf.shape(ids)[1], self.config.context)
        if not self.built:
            self(ids)
        x, caches = self.hidden(ids)
        return self.logits(x[:, -1:])[:, 0], caches

    def decode_step(self, ids, caches):
        self.validate_ids(ids)
        tf.debugging.assert_equal(tf.shape(ids)[1], 1, message="decode exactly one token")
        if len(caches) != self.config.layers:
            raise ValueError("one KV pair per block is required")
        length = tf.shape(caches[0][0])[2]
        tf.debugging.assert_positive(length)
        tf.debugging.assert_less(length, self.config.context, message="context full; prefill a fresh window")
        for keys, values in caches:
            tf.debugging.assert_rank(keys, 4)
            tf.debugging.assert_rank(values, 4)
            tf.debugging.assert_equal(tf.shape(keys), tf.shape(values))
            tf.debugging.assert_equal(tf.shape(keys), [tf.shape(ids)[0], self.config.kv_heads,
                                                       length, self.config.head_width])
        x, updated = self.hidden(ids, caches)
        return self.logits(x)[:, 0], updated

    def save_checkpoint(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        self.save_weights(directory / "model.weights.h5")


def load_checkpoint(directory):
    directory = Path(directory)
    model = Decoder(Config(**json.loads((directory / "config.json").read_text())))
    model(tf.zeros([1, 1], tf.int32))
    model.load_weights(directory / "model.weights.h5")
    return model
