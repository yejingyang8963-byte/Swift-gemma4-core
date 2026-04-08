# BasicGeneration

The simplest possible end-to-end demo of `Gemma4SwiftCore`. A standalone
SwiftPM executable that loads the Gemma 4 E2B 4-bit model from
HuggingFace and generates text from a command-line prompt.

## Build

```bash
cd examples/BasicGeneration
swift build -c release
```

This builds against the `Gemma4SwiftCore` source in the parent
directory via a relative-path `.package` reference, so any local
changes you make to the library are picked up immediately.

## Run

```bash
swift run -c release gemma4-generate "Tell me a short story about a fox."
```

The first run downloads ~1.5 GB of model weights from HuggingFace into
the platform's caches directory and takes 1–3 minutes depending on
your network. Subsequent runs are warm.

Without an argument the demo uses a neutral default prompt:

```bash
swift run -c release gemma4-generate
```

## What it demonstrates

- ``Gemma4Registration/registerIfNeeded()`` — wiring the sidecar handler
- ``LLMModelFactory/shared/loadContainer(configuration:)`` — loading the
  4-bit weights via the standard `mlx-swift-lm` path
- ``Gemma4PromptFormatter/userTurn(_:)`` — building the prompt without
  going through the broken swift-jinja chat-template path
- ``ModelContainer/encode(_:)`` — tokenizing the formatted prompt
- ``ModelContainer/generate(input:parameters:)`` — streaming tokens
- Measuring the first-chunk latency end-to-end

## Expected output

```
[gemma4-generate] Gemma4SwiftCore v0.1.0
[gemma4-generate] Model: mlx-community/gemma-4-e2b-it-4bit
[gemma4-generate] Prompt: Tell me a short story about a curious fox.

[gemma4-generate] Loading model (first run downloads ~1.5 GB)...
[gemma4-generate] Loaded in 6.2s.

In a quiet meadow at the edge of an old forest, a small fox named ...

[gemma4-generate] 12 prompt tokens, 12.4 tok/s
[gemma4-generate] First chunk: 2.83s
```

The exact text varies (sampling is non-deterministic), but the
performance numbers should be in the same ballpark as the README's
"Performance" table.
