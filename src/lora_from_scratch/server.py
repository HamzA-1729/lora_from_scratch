"""
Multi-Tenant Zero-Copy LoRA Serving Engine (FastAPI):
1. In-Memory Multi-Adapter Registry: Maintains arbitrary adapter weights in CPU/GPU memory without base model modification.
2. Thread-Safe Context Multiplexing: Uses Python ContextVar to isolate adapter weights per concurrent request.
3. Zero-Copy Serving: Dispatches forward calls through unmerged dynamic paths without in-place weight mutation.
4. Isolated Request State: Allocates a private StaticKVCache per request to eliminate KV cross-contamination.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modern_decoder_llm import DecoderLLM
from modern_decoder_llm.tokenizer import Tokenizer
from modern_decoder_llm.sampler import generate_with_cache
from lora_from_scratch.lora_layer import LoRALinear, active_adapter_context
from lora_from_scratch.injection import inject_lora
from lora_from_scratch.adapter_io import load_adapter_for_context


# Request / Response Schemas
class LoadAdapterRequest(BaseModel):
    adapter_id: str = Field(..., description="Unique alias for the adapter")
    adapter_path: str = Field(..., description="Local directory containing adapter_config.json and adapter_model.pt")


class UnloadAdapterRequest(BaseModel):
    adapter_id: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to continue")
    adapter_id: Optional[str] = Field(None, description="Adapter ID to activate, or null for base model")
    max_new_tokens: int = Field(32, ge=1, le=128)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    adapter_used: Optional[str]


# Global state
app_state: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": torch.device("cpu"),
    "loaded_adapters": {},  # adapter_id -> {layer_name: (A, B, scaling)}
}


def init_serving_model(
    base_checkpoint: Optional[str] = "../modern_decoder_llm/checkpoints/best_model.pt",
    tokenizer_path: str = "../modern_decoder_llm/checkpoints/tokenizer.json",
    device_str: str = "cpu",
) -> None:
    """Initializes base model, wraps target layers with dummy LoRALinear for dynamic routing."""
    device = torch.device(device_str)
    app_state["device"] = device

    # Load tokenizer
    if os.path.isfile(tokenizer_path):
        app_state["tokenizer"] = Tokenizer.load(tokenizer_path)
    else:
        # Fallback dummy tokenizer for tests
        dummy_tok = Tokenizer(vocab_size=4096)
        dummy_tok.train(["Hello world! Once upon a time."])
        app_state["tokenizer"] = dummy_tok

    # Build base model
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

    if base_checkpoint and os.path.isfile(base_checkpoint):
        ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    # Inject dummy LoRALinear wrappers on target linear projections
    # When ContextVar is active, forward() uses the context weights.
    # When ContextVar is None, forward() uses base output directly (since B=0).
    model, _ = inject_lora(model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16.0)
    model.eval()
    model.to(device)

    app_state["model"] = model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model
    if app_state["model"] is None:
        init_serving_model()
    yield
    # Shutdown cleanup
    app_state["loaded_adapters"].clear()


app = FastAPI(
    title="LoRA Multi-Tenant Serving API",
    description="Zero-copy, thread-safe dynamic LoRA adapter switching with private per-request KV caches.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "device": str(app_state["device"])}


@app.get("/adapters")
def list_adapters() -> Dict[str, List[str]]:
    return {"loaded_adapters": list(app_state["loaded_adapters"].keys())}


@app.post("/adapters/load")
def load_adapter_endpoint(req: LoadAdapterRequest) -> Dict[str, Any]:
    try:
        adapter_dict = load_adapter_for_context(req.adapter_path, device=app_state["device"])
        app_state["loaded_adapters"][req.adapter_id] = adapter_dict
        return {
            "status": "success",
            "adapter_id": req.adapter_id,
            "layers_loaded": len(adapter_dict),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/adapters/unload")
def unload_adapter_endpoint(req: UnloadAdapterRequest) -> Dict[str, str]:
    if req.adapter_id in app_state["loaded_adapters"]:
        del app_state["loaded_adapters"][req.adapter_id]
        return {"status": "unloaded", "adapter_id": req.adapter_id}
    raise HTTPException(status_code=404, detail=f"Adapter '{req.adapter_id}' not found.")


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
    model: DecoderLLM = app_state["model"]
    tokenizer: Tokenizer = app_state["tokenizer"]
    device: torch.device = app_state["device"]

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model server not initialized.")

    # 1. Bind Adapter Context if requested
    token = None
    if req.adapter_id is not None:
        if req.adapter_id not in app_state["loaded_adapters"]:
            raise HTTPException(
                status_code=404,
                detail=f"Adapter '{req.adapter_id}' is not loaded. Available: {list(app_state['loaded_adapters'].keys())}",
            )
        token = active_adapter_context.set(app_state["loaded_adapters"][req.adapter_id])

    try:
        with torch.inference_mode():
            prompt_ids = tokenizer.encode(req.prompt)
            if not prompt_ids:
                prompt_ids = [tokenizer.bos_id]

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output_tokens = generate_with_cache(
                model=model,
                prompt_tokens=prompt_tensor,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=1.0,
                eos_token_id=tokenizer.eos_id,
            )

            generated_text = tokenizer.decode(output_tokens[0].tolist())
            return GenerateResponse(
                prompt=req.prompt,
                generated_text=generated_text,
                adapter_used=req.adapter_id,
            )
    finally:
        # Always reset context token in finally block to avoid thread-local leaking
        if token is not None:
            active_adapter_context.reset(token)

