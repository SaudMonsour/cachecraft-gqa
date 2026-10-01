import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from cachecraft import Config, Decoder, Engine, generate_text, load_checkpoint, pool_kv_heads
from cachecraft.model import GroupedAttention, rotary


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)

    def config(self, heads=2):
        return Config(vocabulary=12, context=8, width=32, query_heads=4,
                      kv_heads=heads, layers=2, hidden=48)

    def test_grouped_attention_matches_repeated_head_reference(self):
        for heads in (1, 2, 4):
            config = self.config(heads)
            layer = GroupedAttention(config)
            x = tf.random.normal([2, 7, config.width])
            actual, (keys, values) = layer(x)
            queries, _, _ = layer.project(x)
            repeated_keys = tf.repeat(keys, config.query_heads // heads, axis=1)
            repeated_values = tf.repeat(values, config.query_heads // heads, axis=1)
            scores = tf.matmul(queries, repeated_keys, transpose_b=True) * (config.head_width ** -.5)
            mask = tf.linalg.band_part(tf.ones([7, 7], tf.bool), -1, 0)
            attention = tf.matmul(tf.nn.softmax(tf.where(mask, scores, -1e9), -1), repeated_values)
            reference = layer.output_projection(tf.reshape(tf.transpose(attention, [0, 2, 1, 3]), [2, 7, config.width]))
            np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)
            self.assertEqual(keys.shape[1], heads)

    def test_future_tokens_do_not_change_past_logits(self):
        for heads in (1, 2, 4):
            model = Decoder(self.config(heads))
            left = model(tf.constant([[1, 2, 3, 4, 5]]))
            right = model(tf.constant([[1, 2, 3, 10, 11]]))
            np.testing.assert_allclose(left[:, :3], right[:, :3], atol=1e-6)

    def test_rotary_offsets_match_full_rotation_and_preserve_norm(self):
        x = tf.random.normal([2, 4, 8, 8])
        np.testing.assert_allclose(rotary(x)[:, :, 3:], rotary(x[:, :, 3:], 3), atol=1e-7)
        np.testing.assert_allclose(tf.reduce_sum(x*x, -1), tf.reduce_sum(rotary(x)**2, -1), rtol=1e-5)

    def test_cached_logits_match_full_prefix_in_batches(self):
        ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], np.int32)
        for heads in (1, 2, 4):
            engine = Engine(Decoder(self.config(heads)))
            session = engine.start(ids[:, :2])
            np.testing.assert_allclose(session.logits, engine.full(ids[:, :2]), atol=3e-6)
            for end in range(3, 9):
                np.testing.assert_allclose(session.advance(ids[:, end-1:end]), engine.full(ids[:, :end]), atol=3e-6)
            self.assertEqual(session.cache_bytes(), engine.model.config.cache_payload_bytes(2, 8))

    def test_window_resets_preserve_original_context_semantics(self):
        for heads in (1, 2, 4):
            engine = Engine(Decoder(self.config(heads)))
            cached = engine.start([1, 2, 3, 4, 5, 6, 7, 8, 9])
            reference = engine.start([1, 2, 3, 4, 5, 6, 7, 8, 9], cached=False)
            for token in (10, 11, 1, 2):
                np.testing.assert_allclose(cached.advance([[token]]), reference.advance([[token]]), atol=3e-6)
            self.assertEqual(cached.resets, 4)

    def test_cache_payload_reduction_is_real_tensor_storage(self):
        payloads = []
        for heads in (4, 2, 1):
            model = Decoder(self.config(heads))
            session = Engine(model).start(np.ones([2, 8], np.int32))
            payloads.append(session.cache_bytes())
            self.assertEqual(payloads[-1], model.config.cache_payload_bytes(2, 8))
        self.assertEqual(payloads[0], 2 * payloads[1])
        self.assertEqual(payloads[0], 4 * payloads[2])

    def test_query_key_and_value_receive_finite_gradients(self):
        model = Decoder(self.config())
        ids = tf.constant([[1, 2, 3, 4, 5]])
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ids, logits=model(ids)))
        layers = (model.blocks[0].attention.query, model.blocks[0].attention.key, model.blocks[0].attention.value)
        gradients = tape.gradient(loss, [layer.kernel for layer in layers])
        for gradient in gradients:
            self.assertTrue(np.isfinite(gradient.numpy()).all())
            self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0.)

    def test_head_pooling_preserves_other_weights_and_source(self):
        source = Decoder(self.config(4))
        ids = tf.constant([[1, 2, 3]])
        before = source(ids).numpy()
        target = pool_kv_heads(source, 2)
        for old, new in zip(source.blocks, target.blocks):
            expected = old.attention.key.kernel.numpy().reshape(32, 2, 2, 8).mean(2).reshape(32, 16)
            np.testing.assert_array_equal(new.attention.key.kernel.numpy(), expected)
            expected_v = old.attention.value.kernel.numpy().reshape(32, 2, 2, 8).mean(2).reshape(32, 16)
            np.testing.assert_array_equal(new.attention.value.kernel.numpy(), expected_v)
            for original, copied in ((old.ffn, new.ffn), (old.norm_attention, new.norm_attention),
                                     (old.norm_ffn, new.norm_ffn), (old.attention.query, new.attention.query),
                                     (old.attention.output_projection, new.attention.output_projection)):
                for left, right in zip(original.get_weights(), copied.get_weights()):
                    np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(source.embedding.get_weights()[0], target.embedding.get_weights()[0])
        np.testing.assert_array_equal(source.final_norm.get_weights()[0], target.final_norm.get_weights()[0])
        np.testing.assert_array_equal(source(ids).numpy(), before)

    def test_identity_conversion_is_exact(self):
        source = Decoder(self.config(4))
        ids = tf.constant([[1, 2, 3]])
        before = source(ids).numpy()
        target = pool_kv_heads(source, 4)
        np.testing.assert_array_equal(target(ids).numpy(), before)

    def test_checkpoint_round_trip(self):
        for heads in (1, 2, 4):
            model = Decoder(self.config(heads))
            ids = tf.constant([[1, 2, 3]])
            original = model(ids).numpy()
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "checkpoint"
                model.save_checkpoint(directory)
                loaded = load_checkpoint(directory)
                np.testing.assert_array_equal(loaded(ids).numpy(), original)
                np.testing.assert_allclose(Engine(loaded).start([1, 2, 3]).logits, original[:, -1], atol=1e-6)

    def test_graphs_do_not_retrace_for_cache_growth_or_batch_size(self):
        engine = Engine(Decoder(self.config()))
        for batch in (1, 2):
            session = engine.start(np.ones([batch, 2], np.int32))
            for _ in range(7):
                session.advance(np.ones([batch, 1], np.int32))
            engine.full(np.ones([batch, 3], np.int32))
        self.assertEqual(engine.traces(), {"full": 1, "prefill": 1, "decode": 1})

    def test_request_caches_are_isolated(self):
        engine = Engine(Decoder(self.config()))
        first, second = engine.start([1, 2]), engine.start([7, 8, 9])
        before = second.logits.numpy().copy()
        first.advance([[3]])
        np.testing.assert_array_equal(second.logits.numpy(), before)

    def test_seeded_sampling_matches_cached_and_reference(self):
        engine = Engine(Decoder(self.config()))
        vocabulary = list("abcdefghijkl")
        left = generate_text(engine, vocabulary, "abc", count=20)
        right = generate_text(engine, vocabulary, "abc", count=20, cached=False)
        self.assertEqual(left, right)

    def test_invalid_configuration_conversion_and_cache_are_rejected(self):
        for invalid in ({"kv_heads": 3}, {"query_heads": 3}, {"width": 12}, {"layers": 0}, {"width": 64.0}):
            with self.assertRaises(ValueError):
                Config(**invalid)
        engine = Engine(Decoder(self.config()))
        for ids in ([], [-1], [12], [1.5]):
            with self.assertRaises(ValueError):
                engine.start(ids)
        session = engine.start([1, 2])
        with self.assertRaises(ValueError):
            session.advance([[1, 2]])
        with self.assertRaises(ValueError):
            pool_kv_heads(engine.model, 3)
        with self.assertRaises(ValueError):
            engine.model.config.cache_payload_bytes(1, 9)
        _, full = engine.prefill(tf.ones([1, 8], tf.int32))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            engine.model.decode_step(tf.constant([[1]]), full)
        _, caches = engine.prefill(tf.constant([[1, 2]]))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            engine.model.decode_step(tf.constant([[1], [2]]), caches)
        with self.assertRaises(ValueError):
            engine.model.decode_step(tf.constant([[1]]), caches[:1])


if __name__ == "__main__":
    unittest.main()
