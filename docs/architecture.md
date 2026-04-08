# Architecture

> Standalone version of the architecture article. The DocC version
> at `Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md`
> is the canonical source — this file is a longer-form companion
> for readers who land on the GitHub repository directly.

## Why a port was necessary

`mlx-swift-lm` 2.31.x ships with handlers for Llama 3, Qwen 2/2.5,
Gemma 2, Gemma 3, Phi-3 and a handful of other architectures. Gemma 4
is not on that list. The first instinct — borrow Gemma 3's text
implementation and patch the model_type — fails immediately:

```
LLMModelFactory error:
    Missing field 'laurel_rank' in Gemma3TextConfiguration
```

`laurel_rank` belongs to **Gemma 3n**, an entirely different model
that ships a Laurel attention block. Gemma 4 uses a different mechanism
(Per-Layer Embedding) and never declares `laurel_rank`. Trying to make
Gemma 3n's loader accept a Gemma 4 weight file is a dead end — the
two models have different parameter sets, different layer types, and
even different head dimensions per layer.

So Gemma4SwiftCore implements the entire text decoder from scratch.
The remainder of this article walks through the five things that make
Gemma 4 different, and points at the specific Swift file that handles
each one.

## Difference 1 — Per-Layer Embedding (PLE)

Every Gemma 4 decoder layer receives a per-token vector that is NOT
just a slice of the main token embeddings. The vector comes from a
two-path computation done once at the top of the model:

```
                       ┌── Path A ──┐
                       │            │
                       │  embed_tokens_per_layer(input_ids)
                       │  × sqrt(hidden_size_per_layer_input)
                       │  reshape [B, L, num_layers, hidden_size_per_layer_input]
                       │
input_ids ── reshape ──┤
                       │
                       │  per_layer_model_projection(inputs_embeds)
                       │  × hidden_size**-0.5
                       │  reshape [B, L, num_layers, hidden_size_per_layer_input]
                       │  per_layer_projection_norm
                       │            │
                       └── Path B ──┘
                                    │
                       (Path A + Path B) × 2**-0.5
                                    │
                                    ▼
                       per_layer_inputs[B, L, num_layers, ...]
```

Each decoder layer slices its own `[..., i, :]` row out of the result
and feeds it into a third residual block (after attention and after
the FFN):

```
gate = per_layer_input_gate(h)              # hidden_size → per_layer
gate = gelu_approx(gate)
gate = gate * per_layer_input               # elementwise
gate = per_layer_projection(gate)           # per_layer → hidden_size
gate = post_per_layer_input_norm(gate)
h    = h + gate                             # third residual
```

**Implementation:**
- Top-level computation: `Sources/Gemma4SwiftCore/Model/Gemma4TextInner+PerLayerInputs.swift`
- Per-layer block: `Sources/Gemma4SwiftCore/Layers/Gemma4TextDecoderLayer.swift` (the PLE section after the FFN residual)

## Difference 2 — Layer scalar

After the PLE block, every decoder layer multiplies its output by a
single learned scalar of shape `[1]`. It's tiny but not optional —
without it, attention contributions across layers fall out of balance
and the model degrades subtly.

**Implementation:** see the `layerScalar` parameter and the final
`h = h * layerScalar` line in `Gemma4TextDecoderLayer`.

## Difference 3 — KV sharing

The Gemma 4 E2B checkpoint has `num_kv_shared_layers = 20`. The last 20
of the 35 decoder layers do not compute their own keys and values.
Instead they reuse the keys and values produced by the most recent
earlier non-shared layer of the same `layer_type` (sliding vs full).

There are two pieces to get right:

### 3a — The donor table

We thread a dictionary `kvDonors: [Int: (keys, values)]` through the
forward pass. Each non-shared layer stashes its post-cache-update
K/V keyed by its own layer index. Each shared layer looks up its
donor by walking layer types in reverse:

```swift
public func kvDonorLayerIndex(forSharedLayer layerIdx: Int) -> Int? {
    guard isKvSharedLayer(layerIdx) else { return nil }
    let myType = layerTypes[layerIdx]
    for i in stride(from: firstKvSharedLayerIdx - 1, through: 0, by: -1)
    where layerTypes[i] == myType {
        return i
    }
    return nil
}
```

### 3b — The global rope offset

This is the subtle part. In Python `mlx-lm`, each non-shared layer
captures `cache.offset` BEFORE calling `cache.update_and_fetch`,
threads it through the `intermediates[]` table alongside the K/V,
and shared layers rope their queries at that captured offset.

The naive Swift port — read each layer's own `cache.offset` — has a
silent bug: shared layers' caches are never updated, so their offset
is permanently `0`. Their queries get rope'd at position 0 during
generation. The model still emits fluent local tokens but loses all
sense of position; it loops on phrase fragments and produces
incoherent output.

The fix is to compute a SINGLE global rope offset at the top of the
forward pass, BEFORE any cache update happens, by reading the offset
of the first non-shared layer:

```swift
let globalRopeOffset: Int = {
    for i in 0 ..< config.hiddenLayers
    where !config.isKvSharedLayer(i) {
        if let c = layerCache[i] { return c.offset }
    }
    return 0
}()
```

Every layer in one forward call uses the same value. Prompt phase: 0.
Generation phase: `L_prompt + steps_so_far` (because every prior
forward pass advanced every non-shared cache by the same amount).

**Implementation:** see the body of `Gemma4TextInner.callAsFunction`
and the `KV donor lookup` block at the start of the per-layer loop.

## Difference 4 — Per-layer attention type and head dim

`config.layer_types[i]` is one of `"sliding_attention"` or
`"full_attention"`. Sliding layers attend over the most recent
`slidingWindow` tokens; full layers attend over the entire sequence.
The two types use different head widths:

| Type | head_dim | rope theta | partial_rotary_factor | rope_type |
|---|---:|---:|---:|---|
| sliding_attention | 256 | 10_000 | 1.0 | default |
| full_attention | 512 | 1_000_000 | 0.25 | proportional |

Both types share `num_attention_heads = 8` — only the head **width**
varies. So `q_proj`, `k_proj`, `v_proj`, and `o_proj` have different
output dimensions per layer index.

`Gemma4TextConfiguration.headDimForLayer(_:)` returns the right value,
and `Gemma4TextAttention.init(config:layerIdx:)` plumbs it into the
linear layer constructors.

## Difference 5 — Proportional RoPE

This is the file that has no upstream equivalent.
`mlx-swift-lm`'s built-in `initializeRope` understands `default`,
`linear`, `llama3`, `yarn`, `longrope`, and `mrope`. It does NOT
understand `proportional`. Passing `proportional` to it would either
crash or silently fall through to the wrong default.

We ship `Gemma4ProportionalRoPE`, which builds the freqs array exactly
the way Python's `mlx_lm.models.rope_utils.ProportionalRoPE` does:

```python
exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
freqs = mx.concatenate([
    factor * (base ** exponents),
    mx.full(((dims - rotated_dims) // 2,), mx.inf),
])
```

The `+inf` padding at the end of `freqs` makes the corresponding
channels effectively no-op in `MLXFast.RoPE`: `cos(inf * t)` and
`sin(inf * t)` collapse to a pass-through. So the first 128 channels
of each head get rotated and the remaining 384 pass through unchanged.

`Tests/Gemma4SwiftCoreTests/ProportionalRoPETests.swift` reproduces the
formula in pure Swift and asserts it within 1e-3 of the expected
values, on every CI run.

## What we deliberately did NOT implement

- **MoE Gemma 4 variants** (`enable_moe_block = true`). The Python
  reference has `Router` / `Experts` / `SwitchGLU` classes for the
  larger 26B/31B models, but the E2B configuration has
  `enable_moe_block = false` and the checkpoint contains no expert
  tensors. We precondition on the flag at construction.
- **K=V attention** (`attention_k_eq_v = true`). Used by the larger
  variants. Not implemented; only K ≠ V is supported.
- **AltUp / Laurel.** Those belong to Gemma 3n, not Gemma 4. No code,
  no tests, no config plumbing.

## See also

- `docs/architecture.md` (this file) — companion overview
- `Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md` — DocC
  rendition with cross-references
- `Sources/Gemma4SwiftCore/Documentation.docc/ProportionalRoPE.md` —
  RoPE math deep-dive
- `Sources/Gemma4SwiftCore/Documentation.docc/KVSharing.md` —
  KV sharing mechanism explained
- `Sources/Gemma4SwiftCore/Documentation.docc/ChatTemplateBypass.md` —
  why we sidestep `tokenizer.applyChatTemplate`
