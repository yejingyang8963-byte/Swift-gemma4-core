# Why Swift?

> A short pitch aimed at the Python / ML / `mlx-lm` audience for why a
> Swift port of Gemma 4 even matters.

## The default expectation

If you've been doing LLM work in 2024-2026, your default tools are
probably some combination of PyTorch, vLLM, llama.cpp, Ollama, and
maybe MLX on the side for your Mac. You think of Python as the
training and inference glue and Swift (if you've heard of it) as the
language people use to build iOS apps that **call** an AI service.

The premise of `Gemma4SwiftCore` is the opposite: **ship the model
inside the app**. Not as a remote API call, not as a server proxy,
not as a Python subprocess — as a first-class native dependency that
runs on the same Metal cores as your UI.

## The case for on-device LLM inference

Three things have changed in the last 12 months that make this
practical:

1. **Apple Silicon + MLX.** The combination of unified memory,
   16+ GB on iPhone Pro models, and Apple's MLX library means you
   can run a 4-bit quantized 2B-parameter model on a phone with no
   special hardware setup. Five years ago this required cloud GPUs.
2. **4-bit quantization quality.** GGUF and MLX 4-bit quantization
   have closed the gap with FP16 to the point where Gemma 4 E2B 4-bit
   feels like a 1.5-2x slower full-precision model, not a degraded
   one.
3. **Privacy regulations and user expectations.** Anything you send
   to a remote server is now a compliance question (GDPR, CCPA,
   COPPA for kids' apps, HIPAA for health). On-device inference makes
   that question vanish: no data leaves the user's device, no API
   keys, no network bills.

For consumer apps in particular — kids' apps, journaling apps,
mental-health apps, anything personal — on-device inference is no
longer optional. It's the only acceptable architecture.

## Why Swift specifically

The on-device argument leads you to one of three runtimes:

| Runtime | Pros | Cons |
|---|---|---|
| **CoreML** | Apple's official path; great Xcode integration | Requires conversion from PyTorch/JAX; quantization story is weak; no Gemma 4 support |
| **llama.cpp / GGUF via Swift bridge** | Mature, fast, supports many models | C++ runtime to wrangle; quantization formats out of sync with HF; no MLX speed |
| **MLX-Swift** | Native Apple, follows HF tensor layout exactly, uses Metal natively, handles KV cache and tokenization | Smaller ecosystem, fewer pre-built models |

`Gemma4SwiftCore` chose the MLX-Swift path because:

- HuggingFace `mlx-community` repos work as-is — no conversion step,
  no scripts, no toolchain quirks. Drop the model id into a string,
  call `loadContainer`, done.
- The model weights load via memory-mapped safetensors, so the iPhone
  doesn't have to allocate 1.5 GB of contiguous RAM up front.
- Apple's MLX team owns the Metal kernels and tracks `mlx-lm` Python
  upstream. We get free perf improvements when they ship updates.
- Swift's strong typing, structured concurrency, and DocC integration
  make the resulting library more pleasant to consume than a C++
  binding.

## Why not just wait for upstream `mlx-swift-lm`?

You could. As of April 2026, `mlx-swift-lm` 2.31.x has no Gemma 4
support and the project's issue tracker has an open ticket asking
for it. It will probably land at some point.

But it hasn't landed yet, and even when it does it will need to:

- Implement Per-Layer Embedding (a non-trivial chunk of new code)
- Implement KV sharing across layers (with the global rope offset trick)
- Implement custom Proportional RoPE (or refactor `initializeRope`
  to support new rope types)
- Either fix `swift-jinja` or document the chat-template workaround

That's months of upstream work, and the chat-template issue
specifically isn't even in `mlx-swift-lm` — it's in `swift-jinja`,
which is maintained by HuggingFace. Coordinating fixes across two
separate repos is slow.

`Gemma4SwiftCore` exists so you don't have to wait. And the code is
small enough (~1000 lines, 14 source files, every file under 200 lines)
that you can read it end-to-end in an afternoon and convince yourself
it does what it claims.

## Coming from Python `mlx-lm`?

The Swift API mirrors the Python API as closely as Swift's type
system allows. Roughly:

| Python `mlx-lm` | Swift `Gemma4SwiftCore` |
|---|---|
| `mlx_lm.load(model_id)` | `LLMModelFactory.shared.loadContainer(configuration:)` |
| `tokenizer.apply_chat_template(messages)` | `Gemma4PromptFormatter.userTurn(_:)` (bypass — see ChatTemplateBypass.md) |
| `model.generate(prompt, ...)` | `container.generate(input:parameters:)` |
| `mlx.fast.rope(x, freqs=...)` | `MLXFast.RoPE(x, freqs:...)` |
| `nn.RMSNorm(dim)` | `RMSNorm(dimensions: dim)` |

The decoder forward pass in `Gemma4TextInner.swift` was deliberately
written to read like a transliteration of `Gemma4TextModel.__call__`
in the mlx-lm Python source. If you know one, you can read the other.

## TL;DR

- On-device is the new privacy default for consumer apps
- MLX-Swift is the right runtime for Apple devices
- Upstream `mlx-swift-lm` doesn't have Gemma 4 yet
- The chat-template issue is in `swift-jinja` and needs a workaround
- This package fills both gaps with ~1000 lines of clean Swift
