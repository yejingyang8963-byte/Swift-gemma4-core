# Test Fixtures

## `tiny_config.json`

A 5-layer synthetic Gemma 4 configuration used by the unit tests.

This file is **not** a copy of any real HuggingFace `config.json`. It is a
minimal hand-written fixture with the same schema, sized down so the unit
tests can construct module instances without allocating the full E2B
parameter set:

| Field | tiny | E2B (real) |
|---|---:|---:|
| `hidden_size` | 32 | 1536 |
| `num_hidden_layers` | 5 | 35 |
| `num_attention_heads` | 4 | 8 |
| `head_dim` (sliding) | 8 | 256 |
| `global_head_dim` (full) | 16 | 512 |
| `num_kv_shared_layers` | 2 | 20 |
| `hidden_size_per_layer_input` | 8 | 256 |
| `vocab_size` | 100 | 262144 |
| `sliding_window` | 16 | 512 |

The layer-type pattern (`4 × sliding_attention` then `1 × full_attention`)
exercises the same code paths as the real model:

- Layer 0–2: sliding, non-shared
- Layer 3: sliding, KV-shared (donor = layer 2)
- Layer 4: full, non-shared (with proportional RoPE)
