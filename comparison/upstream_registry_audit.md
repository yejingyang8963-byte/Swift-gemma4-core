# Upstream Registry Audit

**Question**: Does any actively-maintained Swift fork of `mlx-swift-lm`
register a `gemma4` model type that would let it load
`mlx-community/gemma-4-e2b-it-4bit`?

**Answer**: No, as of the audit date below.

## Audit data

- **Repository audited**: `adrgrondin/mlx-swift-lm`
- **Branch**: `main`
- **File**: `Libraries/MLXLLM/LLMModelFactory.swift`
- **Audit date**: 2026-04-08
- **Method**: HTTPS fetch of the raw file from GitHub, manual extraction
  of every `registerModelType("…")` string.

## Full list of registered model_type strings (52 total)

```
mistral, llama, phi, phi3, phimoe,
gemma, gemma2, gemma3, gemma3_text, gemma3n,
qwen2, qwen3, qwen3_moe, qwen3_next, qwen3_5, qwen3_5_moe, qwen3_5_text,
minicpm, starcoder2, cohere, openelm, internlm2,
deepseek_v3, granite, granitemoehybrid,
mimo, mimo_v2_flash, minimax,
glm4, glm4_moe, glm4_moe_lite, acereason,
falcon_h1, bitnet, smollm3, ernie4_5, lfm2,
baichuan_m1, exaone4, gpt_oss, lille-130m,
olmoe, olmo2, olmo3, bailing_moe, lfm2_moe,
nanochat, nemotron_h, afmoe, jamba_3b, mistral3, apertus
```

## Gemma family entries

| model_type    | Description                                  | Same as Gemma 4? |
|---------------|----------------------------------------------|:---:|
| `gemma`       | Gemma 1                                      | ❌ |
| `gemma2`      | Gemma 2                                      | ❌ |
| `gemma3`      | Gemma 3 (text+vision wrapper)                | ❌ |
| `gemma3_text` | Gemma 3 text tower only                      | ❌ |
| `gemma3n`     | Gemma 3 nano (PLE + ALTUP + Laurel + sparse MLP) | ❌ — distinct architecture, different config schema |
| `gemma4`      | **NOT PRESENT**                              | — |
| `gemma4_text` | **NOT PRESENT**                              | — |

## Why `gemma3n` ≠ `gemma4`

These are two distinct Google model families with disjoint config
schemas. Loading a Gemma 4 checkpoint with the Gemma 3n handler fails
at config decode (missing `altup_*` / `laurel_rank`), and even if you
patched the decoder, the weight names would not match (Gemma 4 has
`per_layer_input_gate`, Gemma 3n does not; Gemma 3n has `altup_*`
parameters, Gemma 4 does not).

| Config field | Gemma 4 | Gemma 3n |
|---|:---:|:---:|
| `hidden_size_per_layer_input` | ✅ | ✅ |
| `vocab_size_per_layer_input` | ✅ | ✅ |
| `use_double_wide_mlp` | ✅ | ❌ |
| `global_head_dim` | ✅ | ❌ |
| `num_kv_shared_layers` | ✅ | ❌ |
| `attention_k_eq_v` | ✅ | ❌ |
| `altup_num_inputs` | ❌ | ✅ |
| `laurel_rank` | ❌ | ✅ |
| `activation_sparsity_pattern` | ❌ | ✅ |

These tables are mechanically derivable from the public HuggingFace
configs. We are not asserting taste — we are pointing at field names.

## Reproducer

```bash
bash comparison/upstream_attempt.sh
```

The script clones the fork shallowly, greps the registry, and prints
whether `gemma4` is present. It exits 0 if our claim holds (gemma4
absent) and exits 1 if it doesn't (which would mean upstream caught
up — at which point this audit should be retired).
