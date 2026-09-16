"""Focused numerical and failure-path checks for the published CPU helpers."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import derive_params as derive
import prepare


class HelperTests(unittest.TestCase):
    def test_known_e4m3_values(self):
        lut = derive.fp8_lut()
        self.assertEqual(float(lut[0]), 0)
        self.assertEqual(float(lut[0x38]), 1)
        self.assertEqual(float(lut[0xB8]), -1)
        self.assertEqual(float(lut[0x7E]), 448)
        self.assertTrue(np.isnan(lut[0x7F]))

    def test_block_scale_decoding(self):
        result = derive.dequantize(bytes([0x38]) * 1024, bytes([126]), (32, 32), (1, 1))
        np.testing.assert_array_equal(result, np.full((32, 32), 0.5, np.float32))
        with self.assertRaises(ValueError):
            derive.dequantize(bytes([0x38]) * 1024, bytes([255]), (32, 32), (1, 1))

    def test_rank_one_recovery(self):
        rng = np.random.default_rng(12)
        r = rng.standard_normal(32).astype(np.float32)
        r /= np.linalg.norm(r)
        w = rng.standard_normal((32, 48)).astype(np.float32)
        delta = -3.5 * np.outer(r, r @ w)
        recovered, _ = derive.top_direction(delta)
        self.assertGreater(abs(float(recovered @ r)), 0.99999)

    def test_parallel_sum_and_component_gain(self):
        rng = np.random.default_rng(19)
        r = rng.standard_normal(5120)
        r /= np.linalg.norm(r)
        a, b = rng.standard_normal((2, 3, 5120))
        transform = lambda y: y - 3.5 * (y @ r)[..., None] * r
        np.testing.assert_allclose(transform(a) + transform(b), transform(a + b), atol=1e-12)
        np.testing.assert_allclose(transform(a) @ r, -2.5 * (a @ r), atol=1e-12)

    def test_invalid_params_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'params.npz'
            r = np.zeros(5120, np.float32)
            r[0] = 1
            np.savez(path, r=r, alpha=np.full(26, 3.5), layers=np.arange(10, 36))
            self.assertEqual(len(prepare.validate_params(path)), 64)
            np.savez(path, r=r * 2, alpha=np.full(26, 3.5), layers=np.arange(10, 36))
            with self.assertRaises(ValueError):
                prepare.validate_params(path)
            np.savez(path, r=r, alpha=np.full(26, 1.0), layers=np.arange(10, 36))
            with self.assertRaises(ValueError):
                prepare.validate_params(path)

    def test_unknown_launcher_fails_closed(self):
        with self.assertRaises(ValueError):
            prepare.patch_launcher(b'#!/bin/bash\necho unrelated launcher\n')

    def test_unhonored_range_is_not_read(self):
        class Response:
            status = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, *args): raise AssertionError('Must not read a whole shard')
        with patch.object(derive.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaises(ValueError):
                derive.byte_range('https://example.com/weights', 0, 7)

    def test_truncated_range_rejected(self):
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 0-7/100'}
        with patch.object(derive.urllib.request, 'urlopen', return_value=Response(b'bad')):
            with self.assertRaises(ValueError):
                derive.byte_range('https://example.com/weights', 0, 7)

    def test_oversized_range_rejected_before_network(self):
        with patch.object(derive.urllib.request, 'urlopen', side_effect=AssertionError('No request expected')):
            with self.assertRaises(ValueError):
                derive.byte_range('https://example.com/weights', 0, 128 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
