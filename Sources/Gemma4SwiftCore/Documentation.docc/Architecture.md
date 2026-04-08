# Architecture

How `Gemma4SwiftCore` differs from a stock Gemma 3 text decoder, and why
each difference matters.

## Overview

Gemma 4 is structurally similar to Gemma 3 but adds five distinctive
mechanisms. Skipping any one of them produces a model that loads but
generates incoherent output. This article walks through each in turn and
points at the file in this package that implements it.

## The five differences from Gemma 3

### 1. Per-Layer Embedding (PLE)

Every decoder layer receives a per-token vector from a shared
`embed_tokens_per_layer` table. The vector is computed once at the top of
the model from two parallel paths and then sliced per layer:

- **Path A — direct lookup:** `embed_tokens_per_layer(input_ids)`,
  multiplied by `sqrt(hidden_size_per_layer_input)`.
- **Path B — projection from token embeddings:**
  `per_layer_model_projection(inputs_embeds)`, multiplied by
  `hidden_size ** -0.5`, then RMS normalized.

The two paths are added and scaled by `2 ** -0.5` to produce a
`[batch, seq, num_layers, hidden_size_per_layer_input]` tensor. Each
decoder layer slices its own row.

Inside the decoder layer, the PLE block runs after the FFN residual:

```
residual = h
h = per_layer_input_gate(h)         // hidden_size → hidden_size_per_layer_input
h = gelu_approx(h)
h = h * per_layer_input              // elementwise gate
h = per_layer_projection(h)          // back to hidden_size
h = post_per_layer_input_norm(h)
h = residual + h                     // third residual after attention + FFN
```

**Files:** ``Gemma4TextInner``, ``Gemma4TextInner/computePerLayerInputs(inputIds:inputsEmbeds:)``,
``Gemma4TextDecoderLayer``.

### 2. Layer scalar

Each decoder layer multiplies its final output by a single learned
scalar (shape `[1]`). It is a small but load-bearing detail — without it,
the model loses the per-layer modulation Google trained in.

**File:** ``Gemma4TextDecoderLayer/layerScalar``.

### 3. KV sharing across the back half of the decoder

The last `num_kv_shared_layers` decoder layers (20 of 35 in E2B) do not
compute their own keys and values. Instead they reuse the K/V tensors
produced by the most recent earlier non-shared layer of the same
attention type (sliding vs full).

The implementation threads a "donor table" through the forward pass:
each non-shared layer stashes its post-cache-update K/V in a
`[Int: (keys, values)]` dictionary keyed by its layer index, and each
shared layer looks up its donor by walking back through layer types.

**Subtlety:** the rope offset must be a SINGLE global value computed
before any cache update, not a per-layer offset. Shared layers' own
caches are never updated, so they would always report `offset = 0` and
the queries would rope at position 0 during generation — a silent bug
that produces locally fluent but globally incoherent output.

**Files:** ``Gemma4TextInner/callAsFunction(_:cache:)``,
``Gemma4TextAttention/callAsFunction(_:mask:cache:sharedKVDonor:ropeOffset:)``.

### 4. Per-layer attention type and head dimension

`config.layer_types[i]` is one of `"sliding_attention"` or
`"full_attention"`. In E2B, sliding layers use `head_dim = 256` and full
layers use `global_head_dim = 512`. The number of attention heads stays
constant — only the head width changes — so `q/k/v/o` projection
dimensions vary per layer index.

**File:** ``Gemma4TextAttention``, ``Gemma4TextConfiguration/headDimForLayer(_:)``.

### 5. Proportional RoPE for full-attention layers

Sliding layers use the standard rope: full rotation across all head
dimensions, `theta = 10_000`. Full attention layers use a NEW rope
variant called `"proportional"`: `theta = 1_000_000`, and only the
first `partial_rotary_factor * head_dim = 0.25 * 512 = 128` channels
are rotated. The remaining 384 channels pass through unchanged.

`mlx-swift-lm`'s built-in `initializeRope` does NOT recognize the
`"proportional"` rope type — it only handles default / linear / llama3 /
yarn / longrope / mrope. We ship our own ``Gemma4ProportionalRoPE`` that
matches Python's `mlx_lm.models.rope_utils.ProportionalRoPE`
byte-for-byte.

**File:** ``Gemma4ProportionalRoPE``.

## A sixth thing that almost broke us: the chat template

It isn't an architectural difference — it's a tokenizer-level one — but
it's the bug that took the longest to find. See <doc:ChatTemplateBypass>.

## See Also

- <doc:ProportionalRoPE>
- <doc:KVSharing>
- <doc:ChatTemplateBypass>
