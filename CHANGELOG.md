# Changelog

All notable changes to `Gemma4SwiftCore` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
starting from `v1.0.0`. Versions prior to `v1.0.0` may include breaking
changes between minor releases.

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-04-07

### Added

- **Native Gemma 4 text decoder** in `Sources/Gemma4SwiftCore/`, ported
  from scratch using only public `mlx-swift` and `mlx-swift-lm` APIs.
  Implements all five Gemma 4 architectural features that distinguish it
  from Gemma 3:
    - Per-Layer Embedding (PLE) with the two-path lookup + projection mechanism
    - Layer scalar (single learned multiplier per decoder layer)
    - KV sharing across the back half of the decoder, with the global
      rope offset trick
    - Per-layer attention type (sliding vs full) with per-layer head dim
    - Custom `Gemma4ProportionalRoPE` for Gemma 4's proportional rope type
- **Chat template bypass** (`Gemma4PromptFormatter`) that produces token
  sequences identical to Python `mlx-lm`'s `tokenizer.apply_chat_template`,
  working around a swift-jinja rendering bug that drops 5 tokens and
  mangles the system turn token id.
- **`Gemma4Registration`** sidecar that hooks the model into
  `LLMModelFactory.shared.typeRegistry` so any HuggingFace Gemma 4 repo
  can be loaded by URL.
- **Six unit-test files** covering configuration parsing, weight key
  sanitization, ProportionalRoPE freqs formula, prompt formatter
  literals, and MLX module forward shapes. Plus an opt-in network
  integration test gated by `GEMMA4_TEST_NETWORK=1`.
- **DocC documentation catalog** with five articles: package landing,
  Architecture deep-dive, ProportionalRoPE math, KVSharing donor table,
  and ChatTemplateBypass story.
- **Five-language README** (English, Simplified Chinese, Japanese,
  Korean, Spanish), each ~290 lines with hero banner, badges,
  benchmarks, quickstart, FAQ, and reproducibility instructions.
- MIT License, comprehensive `.gitignore`, `NOTICE` for third-party
  attribution, `CITATION.cff` for academic citation.

### Verified

- Real-device performance on iPhone with Apple A-series + 7.4 GB RAM:
  warm load 6 s, peak memory 392 MB, time-to-first-audio 2.82 s,
  generation throughput 12–14 tok/s.
- Token-level equivalence between `Gemma4PromptFormatter.userTurn` and
  Python `mlx-lm` `apply_chat_template` for the canonical
  `[{role: user, content: "TEST"}]` input.

### Known limitations

- MoE Gemma 4 variants (`enable_moe_block = true`) are not implemented.
  The top-level model preconditions on this flag at construction.
- The K=V attention variant (`attention_k_eq_v = true`, used by larger
  26B/31B models) is not implemented; only K ≠ V is supported.
- `Gemma 4 E4B` has not been verified on real device. The code paths
  exist but the test matrix only covers E2B.
- `swift test` from the command line cannot find the MLX Metal library
  bundle; tests requiring real MLX initialization must run via
  `xcodebuild test`. Pure-Swift tests work either way.

[Unreleased]: https://github.com/yejingyang8963-byte/Swift-gemma4-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yejingyang8963-byte/Swift-gemma4-core/releases/tag/v0.1.0
