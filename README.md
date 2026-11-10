# Low-Rank Adaptation (LoRA) from Scratch in PyTorch

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, mathematical first-principles implementation of Low-Rank Adaptation (LoRA, Hu et al. 2021) in PyTorch without relying on high-level libraries (`peft`, `trl`). Features a generic `LoRALinear` module, recursive model injection with tied-weight protection, zero-overhead in-place weight merging, parameter rank sweeps, and a thread-safe multi-tenant adapter serving server.

---

## 🌟 Core Capabilities & Key Highlights

* **Generic `LoRALinear` Layer:** Decoupled adapter layer implementing $\Delta W = \frac{\alpha}{r} B A$ with $B = 0$ initialization (exact step-0 base equivalence) and $\mathcal{O}(B \cdot T \cdot d_{\text{in}} \cdot r)$ forward computation.
* **Surgical Recursive Injector:** Automatically traverses arbitrary PyTorch networks, freezes base weights, replaces targeted linear projections (`q_proj`, `v_proj`, or all linears), and safeguards tied embeddings (`lm_head`).
* **Zero-Overhead Serving (`merge_and_unload`):** Folds adapter weights into base linear matrices in-place for zero-latency serving without memory overhead, with full mathematical reversibility (`atol=1e-5`).
* **Multi-Tenant Adapter Serving (FastAPI):** Thread-safe serving engine utilizing Python `contextvars` to route dynamic adapter weights per-request across a single shared frozen base model without weight-mutation race conditions.
* **Empirical Rank Sensitivity Sweep:** Controlled parameter sweep across $r \in [2, 4, 8, 16, 32]$ evaluating adaptation loss against exact trainable parameter counts ($4,992 r$ for $q+v$, $28,416 r$ for all linears) and pretraining forgetting.

---

## 📐 Architecture: Low-Rank Adapter Decomposition

```
                    Input x: (B, T, d_in)
                             │
            ┌────────────────┴────────────────┐
            │                                 │
     (Frozen Base Path)             (Trainable LoRA Path)
            │                                 │
     W_0: (d_out, d_in)              A: (r, d_in) [Kaiming Init]
            │                                 │
            │                         r-dim bottleneck
            │                                 │
            │                        B: (d_out, r) [Zero Init]
            │                                 │
            │                            Scaling α / r
            │                                 │
            ▼                                 ▼
         W_0 · x      +               (α / r) · (B · A) · x
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                    Output: (B, T, d_out)
```

---

## 📚 Mathematical Formulations & Complexity

### 1. Matrix Decomposition & Forward Pass
For a dense pre-trained linear projection $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA constrains its parameter update $\Delta W$ by representing it as a low-rank decomposition:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

Where:
* $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized from a uniform Kaiming distribution $\mathcal{U}(-\sqrt{5/d_{\text{in}}}, \sqrt{5/d_{\text{in}}})$.
* $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to **exact zeros**, ensuring $\Delta W = 0$ at step 0:
  $$\Delta W \big|_{t=0} = \frac{\alpha}{r} (0 \cdot A) = 0 \implies h_{\text{adapted}} \equiv h_{\text{base}}$$
* $r \ll \min(d_{\text{in}}, d_{\text{out}})$ is the intrinsic low-rank constraint ($r \in [2, 4, 8, 16, 32]$).
* $\alpha$ is a fixed scaling hyperparameter (typically $\alpha = 2r$ for constant effective step size).

To avoid forming the full matrix $B A \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ during training, computation is factorized sequentially:
$$h = W_0 x + \frac{\alpha}{r} \left( (x A^T) B^T \right)$$
Computing $x A^T$ first reduces FLOP complexity from $\mathcal{O}(d_{\text{in}} d_{\text{out}})$ to $\mathcal{O}(d_{\text{in}} r + d_{\text{out}} r)$ per token.

### 2. In-Place Weight Merging & Reversibility
For zero-latency inference, the low-rank delta is folded directly into the base weights under `torch.no_grad()`:
$$W_0 \leftarrow W_0 + \frac{\alpha}{r} B A$$
Reversal for continued training or adapter switching:
$$W_0 \leftarrow W_0 - \frac{\alpha}{r} B A$$

### 3. Exact Trainable Parameter Formulations
On our 5.57M `DecoderLLM` ($D=6$ layers, $d=256$, $H_Q=8, H_{KV}=2, d_k=32$, $d_{\text{ffn}}=768$):

| Target Projections | Dimension Per Layer | Parameters Per Layer | Total Parameters (6 Layers) | % of 5.57M Base |
| :--- | :---: | :---: | :---: | :---: |
| **$W_q + W_v$** | $(256 \times 256) + (64 \times 256)$ | $832 \cdot r$ | **$4,992 \cdot r$** | **$0.18\% - 2.87\%$** |
| **All Linears** (excluding `lm_head`) | $W_{q,k,v,o} + W_{\text{gate,up,down}}$ | $4,736 \cdot r$ | **$28,416 \cdot r$** | **$1.02\% - 16.3\%$** |

---

## 📊 Empirical Benchmarks & Adaptation Findings

### 1. Task Adaptation Convergence (500 Steps, $r=8, \alpha=16.0$)

Both tasks adapt the pre-trained `DecoderLLM` ($5.57\text{M}$ base parameters, pretraining $\text{PPL}=5.70$) by updating only **39,936 parameters (0.711%)**:

| Task | Target Modules | Trainable Parameters | Best Val Loss | Test Loss | Test Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lesson Continuation** | $W_q, W_v$ | **39,936 (0.711%)** | **1.4015** | **1.4663** | **4.33** |
| **Pirate Dialect Domain** | $W_q, W_v$ | **39,936 (0.711%)** | **1.6104** | **1.7500** | **5.75** |

### 2. Qualitative Generations & Dialect Shift

* **Lesson Continuation Prompt:**
  ```text
  Story: A little rabbit found a huge carrot and decided to share it with his hungry friend.
  Lesson: 
  ```
  **Adapter Output:**
  ```text
  From that day on, they were the best of friends and they always play
  ```
* **Lesson Continuation Prompt:**
  ```text
  Story: Timmy lied about breaking the flower pot, but felt so guilty he told the truth.
  Lesson: 
  ```
  **Adapter Output:**
  ```text
  From that day on, Timmy learned that it's important to be careful and not b
  ```
* **Pirate Dialect Prompt:**
  ```text
  Hello my dear friend, where is the golden treasure
  ```
  **Adapter Output (7.4% Pirate Lexical Density vs 0.0% Base):**
  ```text
  Hello my dear friend, where is the golden treasure. Do ye want to play with yet?" asked the music. Then, they went to the doc
  ```

### 3. Inference Latency & Merging Benchmark (CPU, Batch=1, SeqLen=64)

| Serving Mode | Mean Latency (ms) | Overhead vs Base | Key Operational Characteristic |
| :--- | :---: | :---: | :--- |
| **Base Model** | **6.89 ms** | **0.0%** | Unadapted reference baseline |
| **LoRA Unmerged (Dynamic)** | **7.46 ms** | **+8.2%** | Factorized delta $h = W_0 x + \frac{\alpha}{r} (x A^T) B^T$; required for multi-tenant serving |
| **LoRA Merged In-Place** | **7.34 ms** | **+6.5%** | Folds weights into $W_0$; bypasses factorized matrix products |
| **LoRA Merged & Unloaded** | **7.00 ms** | **+1.6%** | Converts back to standard pure `nn.Linear`; zero-overhead deployment |

> [!NOTE]
> `merge_and_unload()` completely reconstructs standard `nn.Linear` layers, eliminating all adapter overhead and matching pure base model execution latency while preserving adapted behavior.

---

## 📁 Repository Structure

```text
lora_from_scratch/
├── README.md                      # Mathematical specification, rank sweep plots, serving architecture
├── pyproject.toml                 # uv package configuration with modern_decoder_llm dependency
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── CITATION.cff                   # Citation metadata
├── .pre-commit-config.yaml        # Code formatting and linting hooks
├── configs/
│   ├── lesson_lora.yaml           # Lesson adaptation config (r=8, alpha=16, lr=5e-4, steps=500)
│   ├── pirate_lora.yaml           # Pirate style causal LM config (r=8, alpha=16, lr=5e-4, steps=500)
│   └── sweep.yaml                 # Pinned rank sweep configuration (r in [2, 4, 8, 16, 32], 2 seeds)
├── data/                          # Extracted lesson pairs & pirate corpus (git-ignored)
├── checkpoints/                   # Saved adapter weights (.pt) (git-ignored)
├── outputs/                       # Sweep comparison plots & tables (git-ignored)
├── src/lora_from_scratch/
│   ├── __init__.py                # Package exports (LoRALinear, inject_lora, merge_and_unload, etc.)
│   ├── lora_layer.py              # Generic LoRALinear with ContextVar multi-adapter routing & merge_and_unload
│   ├── injection.py               # Recursive replacement (excludes lm_head & base_layer), leaf-name matching
│   ├── adapter_io.py              # Save/load A & B with metadata validation (base hash, shapes, alpha, r)
│   ├── prepare_data.py            # Whole-sentence lesson extractor & whole-word pirate rewriter
│   ├── dataset.py                 # PyTorch Datasets: LessonDataset (prompt=-100) & PirateDataset (causal LM)
│   ├── train.py                   # Unified trainer: LoRA, full fine-tune baseline, zero-shot eval
│   ├── evaluate.py                # Target token cross-entropy & plain-story forgetting check
│   ├── sweep.py                   # Rank sweep runner (parameter %, validation loss, speed, reference lines)
│   └── server.py                  # Thread-safe FastAPI server with inference_mode & ContextVar reset
└── tests/
    ├── __init__.py
    ├── test_lora_layer.py         # Step-0 equivalence, 2-step gradient isolation, scaling alpha/r
    ├── test_merge_unmerge.py      # In-place merge/unmerge (atol=1e-5), merged==unmerged, merge_and_unload
    ├── test_injection.py          # lm_head exclusion, leaf matching, parameter formulas, no double-wrapping
    ├── test_adapter_io.py         # Save/load round-trip, shape & missing key validation, state restoration
    ├── test_dataset.py            # Length <= 256 invariant, prompt target split (whole sentence, no leak)
    └── test_overfit.py            # Small 2-layer test model with all linears overfit test (loss drop >95%)
```

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/HamzA-1729/lora_from_scratch.git
cd lora_from_scratch
uv sync
```

### 2. Run Test Suite
```bash
uv run pytest -v
```

### 3. Data Preparation & Feasibility Audit
```bash
uv run python -m lora_from_scratch.prepare_data
```

### 4. Train LoRA Adapter
```bash
uv run python -m lora_from_scratch.train --config configs/lesson_lora.yaml
```

### 5. Multi-Adapter Serving Server
```bash
uv run python -m lora_from_scratch.server --port 8000
```

---

## 📖 References

1. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
2. **The Llama 3 Herd of Models** — Meta AI, 2024 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783))
3. **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** — Eldan & Li, 2023 ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759))

---

## 📜 License
Distributed under the [MIT License](LICENSE).

