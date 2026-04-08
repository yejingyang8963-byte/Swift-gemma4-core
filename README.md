<p align="center">
  <img src="docs/images/banner.svg" alt="Gemma4SwiftCore" width="720">
</p>

<h1 align="center">Gemma4SwiftCore</h1>

<p align="center">
  <strong>The first native Swift implementation of Google Gemma 4.</strong><br>
  Runs on iPhone, iPad, and Mac. 100% on-device. No Python at runtime.
</p>

<p align="center">
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9%2B-orange.svg" alt="Swift 5.9+"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Platform-iOS%2017%20%7C%20macOS%2014-blue.svg" alt="Platform"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/yejingyang8963-byte/Swift-gemma4-core/actions"><img src="https://img.shields.io/badge/Tests-passing-brightgreen.svg" alt="Tests passing"></a>
  <a href="https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit"><img src="https://img.shields.io/badge/Model-Gemma%204%20E2B%204bit-purple.svg" alt="Gemma 4 E2B 4bit"></a>
  <a href="https://swiftpackageindex.com/yejingyang8963-byte/Swift-gemma4-core"><img src="https://img.shields.io/badge/SwiftPM-compatible-success.svg" alt="SwiftPM"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a>
</p>

---

## What is this?

`Gemma4SwiftCore` is a **pure-Swift port** of Google's
[Gemma 4](https://huggingface.co/google) text decoder. It plugs into
Apple's [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) as
a sidecar model registration, so any HuggingFace Gemma 4 repo (e.g.
`mlx-community/gemma-4-e2b-it-4bit`) can be loaded the same way you load
a Llama or Qwen model — except now Gemma 4 actually works.

There is no Python at runtime. There is no CoreML conversion step.
Everything from token IDs to logits runs on Apple's MLX Metal kernels,
fully on-device.

## Why does this exist?

When this project started in April 2026, `mlx-swift-lm` 2.31.x had no
Gemma 4 support. The naive workaround — borrowing the Gemma 3 text
implementation and patching the config — fails at weight load with a
missing-field error, because Gemma 4 is structurally different from
Gemma 3 in five places. And the chat-template path through swift-jinja
silently corrupts the prompt, leaving the model fluent but incoherent.

This package fixes both problems: it ports the entire decoder to Swift
from scratch, and it ships a chat-template bypass that produces token
sequences identical to Python `mlx-lm`'s `tokenizer.apply_chat_template`.

## Key innovations

- 🧠 **Per-Layer Embedding (PLE)** — Gemma 4's signature feature where
  every decoder layer receives a per-token vector from a shared embedding
  table, gated through a small MLP, and added as a third residual.

- 🔗 **KV sharing across the back half of the decoder** — the last 20 of
  35 layers in E2B reuse K/V tensors from earlier layers of the same
  attention type. We thread a "donor table" through the forward pass and
  use a single global rope offset to keep positions correct during
  generation.

- 🎯 **Proportional RoPE** — a custom partial-rotation RoPE class for
  Gemma 4's full-attention layers. `mlx-swift-lm`'s built-in
  `initializeRope` doesn't recognize this rope type; we ship our own
  ``Gemma4ProportionalRoPE`` that matches Python's reference
  implementation byte-for-byte.

- 💬 **Chat-template bypass** — `swift-jinja` 1.x renders Gemma 4's chat
  template incorrectly (drops 5 tokens, mangles the system turn token
  id). We bypass it entirely and build the prompt as a literal string
  with `<|turn>` markers, then encode through `tokenizer.encode(text:)`,
  which respects the registered special tokens.

See the [Architecture article](Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md)
for the full deep-dive.

## Performance

Measured on a real iPhone (Apple A-series, 7.4 GB RAM) with the
`mlx-community/gemma-4-e2b-it-4bit` checkpoint:

| Metric | Value | Target |
|---|---|---|
| Cold load (download + init) | ~110 s | one-time |
| Warm load (cache hit) | ~6 s | — |
| Memory after load | 341–392 MB | < 2 GB ✅ |
| Time to first audio chunk | **2.82 s** | < 3 s ✅ |
| Generation throughput | 12–14 tok/s | — |

The 2.82 s first-chunk latency was measured end-to-end through the TTS
pipeline of a real shipping app, on a hot model and a 333-token system
prompt. Pure forward-pass throughput is higher.

## Installation

Add `Gemma4SwiftCore` to your `Package.swift`:

```swift
dependencies: [
    .package(
        url: "https://github.com/yejingyang8963-byte/Swift-gemma4-core.git",
        from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "Gemma4SwiftCore", package: "Swift-gemma4-core"),
        ]),
],
```

Or in Xcode: **File → Add Package Dependencies...** → paste the repo URL.

## Quick start

```swift
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

// 1. Register the sidecar handler with mlx-swift-lm. Idempotent.
await Gemma4Registration.registerIfNeeded().value

// 2. Load the real 4-bit weights from HuggingFace.
//    The model is ~1.5 GB and is cached after the first download.
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))

// 3. Format your prompt using the chat-template bypass. DO NOT use
//    tokenizer.applyChatTemplate — it is broken on Gemma 4.
let prompt = Gemma4PromptFormatter.userTurn("Tell me a short story about a curious fox.")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))

// 4. Stream tokens.
let stream = try await container.generate(
    input: input,
    parameters: GenerateParameters(maxTokens: 200, temperature: 0.8, topP: 0.95))
for await event in stream {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gemma4TextModel                        │
│                                                             │
│  ┌──────────────────── Gemma4TextInner ───────────────────┐ │
│  │                                                        │ │
│  │  embed_tokens (×sqrt(hidden_size))                     │ │
│  │       │                                                │ │
│  │       ├──→ computePerLayerInputs() ──→ [PLE per-layer] │ │
│  │       │                                                │ │
│  │       ▼                                                │ │
│  │  globalRopeOffset (read once, before any cache update) │ │
│  │       │                                                │ │
│  │       ▼                                                │ │
│  │  for layer in 0..<35:                                  │ │
│  │      ┌─── Gemma4TextDecoderLayer ───┐                  │ │
│  │      │  input_layernorm             │                  │ │
│  │      │  Gemma4TextAttention         │  ←── donor table │ │
│  │      │   ├─ q/k/v/o (per-layer dim) │                  │ │
│  │      │   ├─ Gemma4ProportionalRoPE  │   (full layers)  │ │
│  │      │   └─ scaledDotProductAttn    │                  │ │
│  │      │  post_attention_layernorm    │                  │ │
│  │      │  Gemma4TextMLP (SwiGLU)      │                  │ │
│  │      │  PLE block (× per_layer_in)  │                  │ │
│  │      │  × layer_scalar              │                  │ │
│  │      └──────────────────────────────┘                  │ │
│  │       │                                                │ │
│  │       ▼                                                │ │
│  │  final RMSNorm                                         │ │
│  └────────┬───────────────────────────────────────────────┘ │
│           ▼                                                 │
│      lm_head (tied to embed_tokens)                         │
│           │                                                 │
│           ▼                                                 │
│      cap × tanh(logits / cap)   ← final logit softcapping   │
└─────────────────────────────────────────────────────────────┘
```

The 14 source files are organized by responsibility:

```
Sources/Gemma4SwiftCore/
├── Configuration/   parsing + derived queries
├── Layers/          MLP, Attention, RoPE, DecoderLayer
├── Model/           Inner stack + LLMModel conformance
├── Registration/    LLMModelFactory hookup
└── Prompt/          chat-template bypass
```

Every source file is **≤ 200 lines**. Every public symbol carries an
Apple-style `///` doc comment for DocC.

## Testing

```bash
# Pure-Swift unit tests (Configuration, Sanitize, ProportionalRoPE math,
# PromptFormatter literals). Runs anywhere Swift runs:
swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"

# Full test suite including MLX module-shape tests. Runs on Apple Silicon
# via Xcode (which handles the Metal library bundling that swift test
# does not):
xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'

# Optional network integration test that downloads the real Gemma 4
# tokenizer and verifies token IDs match Python ground truth:
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

## Reproducibility

Want to verify our chat-template bypass produces the same token IDs as
Python `mlx-lm`? Run the Python baseline script in the repo root on
any Apple Silicon Mac:

```bash
python3 -m venv ~/.mlx-venv
source ~/.mlx-venv/bin/activate
pip install mlx-lm
python scripts/python_baseline.py
```

It loads the same model, formats the same prompt, and prints the token
IDs side-by-side with what `Gemma4PromptFormatter.userTurn` would
produce. They match.

## Comparison

| Feature | `Gemma4SwiftCore` | `mlx-swift-lm` (upstream) | `swift-coreml-transformers` |
|---|:---:|:---:|:---:|
| Gemma 4 support | ✅ | ❌ | ❌ |
| Per-Layer Embedding | ✅ | n/a | n/a |
| KV sharing across layers | ✅ | n/a | n/a |
| Proportional RoPE | ✅ | ❌ | ❌ |
| Chat-template bypass | ✅ | ❌ (broken jinja) | n/a |
| Pure Swift, no Python | ✅ | ✅ | ✅ |
| iOS + macOS | ✅ | ✅ | ✅ |

## FAQ

**Q: Do I need to download the model weights?**
Yes — they are not bundled in this package (the 4-bit checkpoint alone
is ~1.5 GB). The first call to `loadContainer` downloads them via
HuggingFace's hub client into the platform's caches directory.

**Q: What devices can run this?**
Any Apple Silicon device with at least 6 GB of RAM. iPhone 14 / iPhone
13 Pro and newer, M1 Mac and newer.

**Q: Is this affiliated with Google or Apple?**
No. Google publishes Gemma 4 weights under their own license. Apple
publishes mlx-swift and mlx-swift-lm. This package is an independent
port that depends on both. See `NOTICE` for the third-party attribution.

**Q: Can I use this in a commercial app?**
Yes — this code is MIT licensed. The Gemma 4 weights themselves come
with their own license from Google; check it before shipping.

**Q: Why not just wait for upstream support?**
You could. But upstream support hasn't shipped, the architecture has
five distinct features that need to be ported, and the chat-template
issue would still need a workaround. This package exists so you don't
have to wait.

**Q: Can I run this on x86_64 (Intel Mac / Linux)?**
The code compiles, but MLX is Apple Silicon only. There is no path to
running this on Intel or Linux at runtime.

## Roadmap

- **v0.2** — KV cache quantization, larger context window benchmarks
- **v0.3** — Gemma 4 E4B variant support, streaming generation API
- **v1.0** — Stable public API, semantic versioning commitment

## Citation

If you use `Gemma4SwiftCore` in research or commercial work, please cite:

```bibtex
@software{ye2026gemma4swiftcore,
  author = {Ye, Jingyang},
  title  = {{Gemma4SwiftCore}: Native Swift Inference for Google Gemma 4},
  year   = {2026},
  url    = {https://github.com/yejingyang8963-byte/Swift-gemma4-core},
  license = {MIT}
}
```

## Acknowledgments

- Apple's [MLX](https://github.com/ml-explore/mlx) and
  [mlx-swift](https://github.com/ml-explore/mlx-swift) team for the
  underlying Metal-accelerated tensor library
- The [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
  contributors for the `LLMModel` protocol and `KVCache` types this
  package builds on
- Google for the Gemma 4 weights and the
  [transformers reference implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)

## Author

Built and maintained by **[Jingyang Ye](https://github.com/yejingyang8963-byte)**
(叶静阳).

This project is the open-source distillation of work originally done
inside a private iOS app for on-device children's bedtime story
generation. Releasing it because Gemma 4 on Apple devices should not be
gatekept behind a closed source.

## License

MIT. See [LICENSE](LICENSE) for the full text.

Copyright © 2026 Jingyang Ye.
