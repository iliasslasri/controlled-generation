"""Unit tests for structured_generation/utils.py (GPT-2 weight caching)."""
import sys
import os
import json
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structured_generation'))
import utils as sg_utils


# --- _save_to_cache / _load_from_cache round-trip ---

class TestCacheRoundTrip:
    def _make_dummy_params(self, n_layer=2, d_model=4):
        """Create a small dummy params dict matching picoGPT structure."""
        params = {
            "wte": np.random.randn(10, d_model).astype(np.float32),
            "wpe": np.random.randn(5, d_model).astype(np.float32),
            "blocks": [],
            "ln_f": {
                "g": np.ones(d_model, dtype=np.float32),
                "b": np.zeros(d_model, dtype=np.float32),
            },
        }
        for _ in range(n_layer):
            params["blocks"].append({
                "ln_1": {
                    "g": np.ones(d_model, dtype=np.float32),
                    "b": np.zeros(d_model, dtype=np.float32),
                },
                "ln_2": {
                    "g": np.ones(d_model, dtype=np.float32),
                    "b": np.zeros(d_model, dtype=np.float32),
                },
                "attn": {
                    "c_attn": {
                        "w": np.random.randn(d_model, 3 * d_model).astype(np.float32),
                        "b": np.zeros(3 * d_model, dtype=np.float32),
                    },
                    "c_proj": {
                        "w": np.random.randn(d_model, d_model).astype(np.float32),
                        "b": np.zeros(d_model, dtype=np.float32),
                    },
                },
                "mlp": {
                    "c_fc": {
                        "w": np.random.randn(d_model, 4 * d_model).astype(np.float32),
                        "b": np.zeros(4 * d_model, dtype=np.float32),
                    },
                    "c_proj": {
                        "w": np.random.randn(4 * d_model, d_model).astype(np.float32),
                        "b": np.zeros(d_model, dtype=np.float32),
                    },
                },
            })
        return params

    def test_round_trip_preserves_values(self):
        params = self._make_dummy_params(n_layer=2, d_model=4)
        n_head = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test.npz")
            meta_path = os.path.join(tmpdir, "test_meta.json")

            sg_utils._save_to_cache(params, n_head, cache_path, meta_path)
            loaded_params, loaded_n_head = sg_utils._load_from_cache(cache_path, meta_path)

        assert loaded_n_head == n_head
        np.testing.assert_array_equal(loaded_params["wte"], params["wte"])
        np.testing.assert_array_equal(loaded_params["wpe"], params["wpe"])
        np.testing.assert_array_equal(loaded_params["ln_f"]["g"], params["ln_f"]["g"])
        np.testing.assert_array_equal(loaded_params["ln_f"]["b"], params["ln_f"]["b"])

    def test_round_trip_preserves_blocks(self):
        params = self._make_dummy_params(n_layer=3, d_model=8)
        n_head = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test.npz")
            meta_path = os.path.join(tmpdir, "test_meta.json")

            sg_utils._save_to_cache(params, n_head, cache_path, meta_path)
            loaded_params, loaded_n_head = sg_utils._load_from_cache(cache_path, meta_path)

        assert len(loaded_params["blocks"]) == 3
        for i in range(3):
            orig = params["blocks"][i]
            loaded = loaded_params["blocks"][i]
            np.testing.assert_array_equal(loaded["ln_1"]["g"], orig["ln_1"]["g"])
            np.testing.assert_array_equal(loaded["attn"]["c_attn"]["w"], orig["attn"]["c_attn"]["w"])
            np.testing.assert_array_equal(loaded["mlp"]["c_fc"]["w"], orig["mlp"]["c_fc"]["w"])
            np.testing.assert_array_equal(loaded["mlp"]["c_proj"]["b"], orig["mlp"]["c_proj"]["b"])

    def test_meta_file_contains_correct_info(self):
        params = self._make_dummy_params(n_layer=2, d_model=4)
        n_head = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test.npz")
            meta_path = os.path.join(tmpdir, "test_meta.json")

            sg_utils._save_to_cache(params, n_head, cache_path, meta_path)

            with open(meta_path) as f:
                meta = json.load(f)

        assert meta["n_head"] == 4
        assert meta["n_layer"] == 2


# --- load_gpt2_params (cache hit path) ---

class TestLoadGpt2Params:
    def test_loads_from_cache_if_present(self, tmp_path, monkeypatch):
        """If cache files exist, load_gpt2_params should load from them."""
        # Create minimal cache files
        d_model = 4
        n_head = 2
        n_layer = 1
        flat = {
            "wte": np.zeros((10, d_model), dtype=np.float32),
            "wpe": np.zeros((5, d_model), dtype=np.float32),
            "ln_f.g": np.ones(d_model, dtype=np.float32),
            "ln_f.b": np.zeros(d_model, dtype=np.float32),
            "b0.ln_1.g": np.ones(d_model, dtype=np.float32),
            "b0.ln_1.b": np.zeros(d_model, dtype=np.float32),
            "b0.ln_2.g": np.ones(d_model, dtype=np.float32),
            "b0.ln_2.b": np.zeros(d_model, dtype=np.float32),
            "b0.attn.c_attn.w": np.zeros((d_model, 3 * d_model), dtype=np.float32),
            "b0.attn.c_attn.b": np.zeros(3 * d_model, dtype=np.float32),
            "b0.attn.c_proj.w": np.zeros((d_model, d_model), dtype=np.float32),
            "b0.attn.c_proj.b": np.zeros(d_model, dtype=np.float32),
            "b0.mlp.c_fc.w": np.zeros((d_model, 4 * d_model), dtype=np.float32),
            "b0.mlp.c_fc.b": np.zeros(4 * d_model, dtype=np.float32),
            "b0.mlp.c_proj.w": np.zeros((4 * d_model, d_model), dtype=np.float32),
            "b0.mlp.c_proj.b": np.zeros(d_model, dtype=np.float32),
        }
        cache_dir = str(tmp_path)
        safe_name = "test--model"
        cache_path = os.path.join(cache_dir, f"{safe_name}.npz")
        meta_path = os.path.join(cache_dir, f"{safe_name}_meta.json")

        np.savez(cache_path, **flat)
        with open(meta_path, "w") as f:
            json.dump({"n_head": n_head, "n_layer": n_layer}, f)

        # Monkeypatch CACHE_DIR and call load_gpt2_params
        monkeypatch.setattr(sg_utils, "CACHE_DIR", cache_dir)
        params, loaded_n_head = sg_utils.load_gpt2_params("test/model")

        assert loaded_n_head == n_head
        assert len(params["blocks"]) == n_layer
        assert params["wte"].shape == (10, d_model)
