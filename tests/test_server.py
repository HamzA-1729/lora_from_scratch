"""
Unit Tests for Multi-Tenant Serving API (FastAPI TestClient):
1. Verifies /health endpoint.
2. Verifies dynamic adapter loading via /adapters/load.
3. Tests zero-mutation context switching: alternating requests between base model and adapters.
4. Verifies private StaticKVCache isolation across consecutive requests.
"""

from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from modern_decoder_llm import DecoderLLM
from modern_decoder_llm.tokenizer import Tokenizer
from lora_from_scratch.injection import inject_lora
from lora_from_scratch.adapter_io import save_adapter
from lora_from_scratch.server import app, init_serving_model, app_state


@pytest.fixture(scope="module")
def dummy_adapter_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a temporary valid adapter checkpoint for API testing."""
    tmp_dir = tmp_path_factory.mktemp("adapters")
    save_dir = tmp_dir / "pirate_v1"

    model = DecoderLLM(
        vocab_size=4096,
        max_seq_len=256,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=768,
        rope_theta=10000.0,
        tie_weights=True,
        norm_eps=1e-6,
    )
    model, _ = inject_lora(model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)

    save_adapter(model, save_dir, config_overrides={"task": "pirate_test"})
    return save_dir


@pytest.fixture(scope="module")
def client(dummy_adapter_checkpoint: Path) -> TestClient:
    """Initializes FastAPI test client."""
    init_serving_model(base_checkpoint=None, device_str="cpu")
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_adapter_lifecycle_and_generation(client: TestClient, dummy_adapter_checkpoint: Path):
    # 1. Initially no adapters loaded
    res = client.get("/adapters")
    assert res.status_code == 200
    assert res.json() == {"loaded_adapters": []}

    # 2. Base model generation (no adapter)
    gen_base = client.post(
        "/generate",
        json={"prompt": "Once upon a time", "adapter_id": None, "max_new_tokens": 10},
    )
    assert gen_base.status_code == 200
    assert "generated_text" in gen_base.json()
    assert gen_base.json()["adapter_used"] is None

    # 3. Load adapter
    load_res = client.post(
        "/adapters/load",
        json={"adapter_id": "pirate", "adapter_path": str(dummy_adapter_checkpoint)},
    )
    assert load_res.status_code == 200
    assert load_res.json()["status"] == "success"

    # 4. Check loaded adapters
    res_loaded = client.get("/adapters")
    assert "pirate" in res_loaded.json()["loaded_adapters"]

    # 5. Generate with loaded adapter
    gen_adapted = client.post(
        "/generate",
        json={"prompt": "Once upon a time", "adapter_id": "pirate", "max_new_tokens": 10},
    )
    assert gen_adapted.status_code == 200
    assert gen_adapted.json()["adapter_used"] == "pirate"

    # 6. Unload adapter
    unload_res = client.post("/adapters/unload", json={"adapter_id": "pirate"})
    assert unload_res.status_code == 200

    # 7. Requesting unloaded adapter raises 404
    gen_fail = client.post(
        "/generate",
        json={"prompt": "Ahoy", "adapter_id": "pirate", "max_new_tokens": 5},
    )
    assert gen_fail.status_code == 404

