"""
Utility for loading and caching GPT-2 weights.

On first call, downloads weights from HuggingFace and saves them locally
as a .npz file. Subsequent calls load directly from the local cache,
avoiding any network access.
"""

import json
import os

import numpy as np


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".model_cache")


def load_gpt2_params(model_name="openai-community/gpt2"):
    """Load GPT-2 weights as NumPy arrays, with local caching.

    First call downloads from HuggingFace and saves to a local .npz cache.
    Subsequent calls load from the cache directly.

    Returns
    -------
    params : dict  — picoGPT-compatible parameter dict
    n_head : int   — number of attention heads
    """
    safe_name = model_name.replace("/", "--")
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}.npz")
    meta_path = os.path.join(CACHE_DIR, f"{safe_name}_meta.json")

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f"  Loading cached weights from {cache_path}")
        return _load_from_cache(cache_path, meta_path)

    print(f"  Downloading weights from HuggingFace ({model_name})...")
    params, n_head = _download_from_hf(model_name)

    os.makedirs(CACHE_DIR, exist_ok=True)
    _save_to_cache(params, n_head, cache_path, meta_path)
    print(f"  Weights cached to {cache_path}")

    return params, n_head


def _download_from_hf(model_name):
    """Download weights from HuggingFace and return picoGPT-compatible params."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    safetensors_path = hf_hub_download(model_name, "model.safetensors")
    sd = load_file(safetensors_path)

    n_layer = config["n_layer"]
    n_head = config["n_head"]

    def w(key):
        return sd[key]

    params = {
        "wte": w("wte.weight"),
        "wpe": w("wpe.weight"),
        "blocks": [],
        "ln_f": {"g": w("ln_f.weight"), "b": w("ln_f.bias")},
    }

    for i in range(n_layer):
        p = f"h.{i}"
        params["blocks"].append({
            "ln_1": {"g": w(f"{p}.ln_1.weight"), "b": w(f"{p}.ln_1.bias")},
            "ln_2": {"g": w(f"{p}.ln_2.weight"), "b": w(f"{p}.ln_2.bias")},
            "attn": {
                "c_attn": {"w": w(f"{p}.attn.c_attn.weight"),
                           "b": w(f"{p}.attn.c_attn.bias")},
                "c_proj": {"w": w(f"{p}.attn.c_proj.weight"),
                           "b": w(f"{p}.attn.c_proj.bias")},
            },
            "mlp": {
                "c_fc": {"w": w(f"{p}.mlp.c_fc.weight"),
                         "b": w(f"{p}.mlp.c_fc.bias")},
                "c_proj": {"w": w(f"{p}.mlp.c_proj.weight"),
                           "b": w(f"{p}.mlp.c_proj.bias")},
            },
        })

    return params, n_head


def _save_to_cache(params, n_head, cache_path, meta_path):
    """Flatten the nested params dict into a flat .npz and save metadata."""
    flat = {}
    flat["wte"] = params["wte"]
    flat["wpe"] = params["wpe"]
    flat["ln_f.g"] = params["ln_f"]["g"]
    flat["ln_f.b"] = params["ln_f"]["b"]

    for i, block in enumerate(params["blocks"]):
        flat[f"b{i}.ln_1.g"] = block["ln_1"]["g"]
        flat[f"b{i}.ln_1.b"] = block["ln_1"]["b"]
        flat[f"b{i}.ln_2.g"] = block["ln_2"]["g"]
        flat[f"b{i}.ln_2.b"] = block["ln_2"]["b"]
        flat[f"b{i}.attn.c_attn.w"] = block["attn"]["c_attn"]["w"]
        flat[f"b{i}.attn.c_attn.b"] = block["attn"]["c_attn"]["b"]
        flat[f"b{i}.attn.c_proj.w"] = block["attn"]["c_proj"]["w"]
        flat[f"b{i}.attn.c_proj.b"] = block["attn"]["c_proj"]["b"]
        flat[f"b{i}.mlp.c_fc.w"] = block["mlp"]["c_fc"]["w"]
        flat[f"b{i}.mlp.c_fc.b"] = block["mlp"]["c_fc"]["b"]
        flat[f"b{i}.mlp.c_proj.w"] = block["mlp"]["c_proj"]["w"]
        flat[f"b{i}.mlp.c_proj.b"] = block["mlp"]["c_proj"]["b"]

    np.savez(cache_path, **flat)

    meta = {"n_head": n_head, "n_layer": len(params["blocks"])}
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _load_from_cache(cache_path, meta_path):
    """Reconstruct picoGPT-compatible params dict from the flat .npz cache."""
    with open(meta_path) as f:
        meta = json.load(f)

    n_head = meta["n_head"]
    n_layer = meta["n_layer"]

    data = np.load(cache_path)

    params = {
        "wte": data["wte"],
        "wpe": data["wpe"],
        "blocks": [],
        "ln_f": {"g": data["ln_f.g"], "b": data["ln_f.b"]},
    }

    for i in range(n_layer):
        params["blocks"].append({
            "ln_1": {"g": data[f"b{i}.ln_1.g"], "b": data[f"b{i}.ln_1.b"]},
            "ln_2": {"g": data[f"b{i}.ln_2.g"], "b": data[f"b{i}.ln_2.b"]},
            "attn": {
                "c_attn": {"w": data[f"b{i}.attn.c_attn.w"],
                           "b": data[f"b{i}.attn.c_attn.b"]},
                "c_proj": {"w": data[f"b{i}.attn.c_proj.w"],
                           "b": data[f"b{i}.attn.c_proj.b"]},
            },
            "mlp": {
                "c_fc": {"w": data[f"b{i}.mlp.c_fc.w"],
                         "b": data[f"b{i}.mlp.c_fc.b"]},
                "c_proj": {"w": data[f"b{i}.mlp.c_proj.w"],
                           "b": data[f"b{i}.mlp.c_proj.b"]},
            },
        })

    return params, n_head
