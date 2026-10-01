"""Mean-pool a decoder's K/V projection heads; leave other weights unchanged."""
from dataclasses import asdict
import numpy as np
import tensorflow as tf
from .model import Config, Decoder


def pool_kv_heads(source, kv_heads):
    """Return a new decoder, not a mutation or a quality-preserving conversion.

    Original KV heads must be divisible by the desired count. Contiguous heads
    are averaged in both projections. Fine-tuning/uptraining may be necessary.
    """
    if type(kv_heads) is not int or kv_heads < 1 or source.config.kv_heads % kv_heads:
        raise ValueError("new KV count must be a positive divisor of the source KV count")
    if not source.built:
        raise ValueError("build or load the source model before converting")
    config = Config(**dict(asdict(source.config), kv_heads=kv_heads))
    target = Decoder(config)
    target(tf.zeros([1, 1], tf.int32))
    target.embedding.set_weights(source.embedding.get_weights())
    target.final_norm.set_weights(source.final_norm.get_weights())
    old_heads, new_heads, dimension = source.config.kv_heads, config.kv_heads, config.head_width
    for old, new in zip(source.blocks, target.blocks):
        new.norm_attention.set_weights(old.norm_attention.get_weights())
        new.norm_ffn.set_weights(old.norm_ffn.get_weights())
        new.ffn.set_weights(old.ffn.get_weights())
        new.attention.query.set_weights(old.attention.query.get_weights())
        new.attention.output_projection.set_weights(old.attention.output_projection.get_weights())
        for original, pooled in ((old.attention.key, new.attention.key),
                                  (old.attention.value, new.attention.value)):
            weights = original.get_weights()[0]
            grouped = weights.reshape(config.width, new_heads, old_heads // new_heads, dimension)
            pooled.set_weights([np.mean(grouped, axis=2).reshape(config.width, new_heads * dimension)])
    return target
