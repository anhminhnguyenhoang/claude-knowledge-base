---
name: glm-5.2-model-config
description: Hugging Face model config for zai-org/GLM-5.2 — architecture, MoE, sparse attention (IndexShare/DSA), tokenizer, and deployment requirements.
source: https://huggingface.co/zai-org/GLM-5.2 (config.json, generation_config.json, tokenizer_config.json, README.md) — fetched 2026-06-25
---

# GLM-5.2 Model Config (Hugging Face)

Official repo: **[zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)**

Z.AI flagship MoE model for long-horizon agentic and coding tasks. **Not** a 5.2B dense model — the Hub lists ~753B total parameters (256 routed experts, 8 active per token). Supports a **1M-token context** and MIT license.

## Model identity

| Field | Value |
|---|---|
| Architecture class | `GlmMoeDsaForCausalLM` |
| `model_type` | `glm_moe_dsa` |
| Dtype | `bfloat16` |
| License | MIT |
| Languages | en, zh |
| Transformers | v5.12.0+ (`trust_remote_code=True`) |

## Core architecture

| Parameter | Value | Notes |
|---|---|---|
| `hidden_size` | 6144 | |
| `num_hidden_layers` | 78 | |
| `num_attention_heads` | 64 | Full MHA (64 KV heads) |
| `head_dim` | 192 | |
| `intermediate_size` | 12288 | Dense MLP (first 3 layers) |
| `max_position_embeddings` | 1,048,576 | 1M context |
| `vocab_size` | 154,880 | |
| `rms_norm_eps` | 1e-05 | |
| `hidden_act` | silu | |
| `tie_word_embeddings` | false | |

## MoE (Mixture of Experts)

| Parameter | Value |
|---|---|
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `n_shared_experts` | 1 |
| `moe_intermediate_size` | 2048 |
| `moe_layer_freq` | 1 |
| `first_k_dense_replace` | 3 | Layers 0–2 use dense MLP |
| `scoring_func` | sigmoid |
| `topk_method` | noaux_tc |
| `routed_scaling_factor` | 2.5 |
| `norm_topk_prob` | true |
| `n_group` | 1 |
| `topk_group` | 1 |

Layer layout: first **3** layers `"dense"` MLP; remaining **75** layers `"sparse"` (MoE + sparse attention).

## Sparse attention (IndexShare / DSA)

| Parameter | Value |
|---|---|
| `index_topk` | 2048 |
| `index_topk_freq` | 4 | IndexShare reuses indexer every 4 sparse layers |
| `index_n_heads` | 32 |
| `index_head_dim` | 128 |
| `index_share_for_mtp_iteration` | true |
| `index_skip_topk_offset` | 3 |
| `indexer_rope_interleave` | true |

`indexer_types`: 78 entries per layer — `"full"` or `"shared"` in a repeating pattern (IndexShare).

Paper: [IndexShare (arXiv:2603.12201)](https://arxiv.org/abs/2603.12201) — reuses the same indexer across every four sparse attention layers, reducing per-token FLOPs by 2.9× at 1M context.

## Attention compression / RoPE

| Parameter | Value |
|---|---|
| `q_lora_rank` | 2048 |
| `kv_lora_rank` | 512 |
| `qk_head_dim` | 256 |
| `qk_nope_head_dim` | 192 |
| `qk_rope_head_dim` | 64 |
| `v_head_dim` | 256 |
| `rope_theta` | 8,000,000 |
| `rope_interleave` | true |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |

## MTP (speculative decoding)

| Parameter | Value |
|---|---|
| `num_nextn_predict_layers` | 1 |

MTP layer improved for speculative decoding; acceptance length up to +20% vs prior GLM generation.

## Tokenizer (`tokenizer_config.json`)

| Field | Value |
|---|---|
| `model_max_length` | 1,048,576 |
| `padding_side` | left |
| EOS / PAD | `<|endoftext|>` (id 154820) |
| Chat tokens | `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>` |
| Multimodal markers | `<|begin_of_image|>`, `<|end_of_image|>`, video/audio/transcription tokens |

## Generation defaults (`generation_config.json`)

| Field | Value |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `eos_token_id` | [154820, 154827, 154829] |
| `pad_token_id` | 154820 |

API note: `reasoning_effort` supports `high` and `max` for deep-thinking mode (Z.ai API / supported clients).

## Config sanity check

Internally consistent as of 2026-06-25:

- 1M context in both `config.json` and `tokenizer_config.json`
- `indexer_types` length (78) == `num_hidden_layers`
- `mlp_layer_types` length (78) == `num_hidden_layers`
- MoE sparse pattern: 3 dense + 75 sparse layers

## Loading (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zai-org/GLM-5.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
)
```

## Supported inference frameworks

| Framework | Minimum version | Docs |
|---|---|---|
| Transformers | v5.12.0+ | [glm_moe_dsa.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm_moe_dsa.md) |
| vLLM | v0.23.0+ | [recipes](https://recipes.vllm.ai/zai-org/GLM-5.2) |
| SGLang | v0.5.13.post1+ | [cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.2) |
| KTransformers | v0.5.12+ | [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.2-Tutorial.md) |
| Unsloth | v0.1.47-beta+ | [guide](https://unsloth.ai/docs/models/glm-5.2) |

Ascend NPU: vLLM-Ascend, xLLM, SGLang — see [GLM-5 Ascend example](https://github.com/zai-org/GLM-5/blob/main/example/ascend.md).

## Full `config.json` (reference)

```json
{
  "architectures": ["GlmMoeDsaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "dtype": "bfloat16",
  "eos_token_id": [154820, 154827, 154829],
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "head_dim": 192,
  "hidden_act": "silu",
  "hidden_size": 6144,
  "index_head_dim": 128,
  "index_n_heads": 32,
  "index_share_for_mtp_iteration": true,
  "index_skip_topk_offset": 3,
  "index_topk": 2048,
  "index_topk_freq": 4,
  "index_topk_pattern": null,
  "indexer_rope_interleave": true,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 1048576,
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "model_type": "glm_moe_dsa",
  "n_group": 1,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 78,
  "num_key_value_heads": 64,
  "num_nextn_predict_layers": 1,
  "pad_token_id": 154820,
  "pretraining_tp": 1,
  "q_lora_rank": 2048,
  "qk_head_dim": 256,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05,
  "rope_interleave": true,
  "rope_parameters": { "rope_theta": 8000000, "rope_type": "default" },
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "noaux_tc",
  "transformers_version": "5.12.0",
  "use_cache": true,
  "v_head_dim": 256,
  "vocab_size": 154880
}
```

`indexer_types` and `mlp_layer_types` are 78-element arrays — omitted here for brevity; see [raw config.json](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json).

## Relation to GLM-5.1 / aiter work

GLM-5.1 sparse MLA decode is the current focus in aiter FlyDSL work (MLA reduce kernel on gfx942). GLM-5.2 extends the family with IndexShare sparse attention, 1M context, and improved MTP — relevant for future sparse-MLA / long-context kernel targets.

---
Related: [hipkitten HK MLA decode wiring](../hipkitten/02-aiter/hk-mla-decode-wiring.md)

Sources: [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) · [GLM-5 technical report (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763) · [IndexShare (arXiv:2603.12201)](https://arxiv.org/abs/2603.12201)
