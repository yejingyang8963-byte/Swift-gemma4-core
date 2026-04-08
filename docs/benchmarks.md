# Benchmarks

Real-device performance numbers for `Gemma4SwiftCore` and the
methodology used to capture them.

## Hardware tested

| Device | Chip | RAM | OS |
|---|---|---:|---|
| iPhone (production app, A-series) | Apple A-series | 7.4 GB | iOS 17 |

Other devices (M1/M2/M3 Mac, iPad Pro M4) should perform comparably or
better — they have higher GPU memory bandwidth and more thermal
headroom — but they have not been individually measured. If you run
this on your own device, please open an issue with the numbers.

## Methodology

End-to-end measurements were captured inside a real shipping iOS app
that uses `Gemma4SwiftCore` for on-device LLM inference. Each
measurement is the median of 5 runs. The model is "hot" (loaded once
at app launch and reused across runs) unless explicitly noted as a
cold load.

The "first audio chunk" measurement spans:

1. User input via `SFSpeechRecognizer`
2. Prompt construction via `Gemma4PromptFormatter.userTurn`
3. Tokenization via `tokenizer.encode`
4. Model forward pass and first token sampled
5. First sentence buffered through a streaming sentence splitter
6. First TTS chunk sent to `AVSpeechSynthesizer`

So the number includes everything between the user finishing speaking
and the speaker starting to play. It is not the pure forward-pass
latency.

## Numbers

| Metric | Value | CLAUDE.md target | Notes |
|---|---|---|---|
| Cold load (download + init) | ~110 s | one-time | Downloads ~1.5 GB from HuggingFace |
| Warm load (cache hit) | 6.2 s | — | mmap + module construction + weight load |
| Memory after load | 341–392 MB | < 2 GB ✅ | Steady state, no generation |
| Memory at generation start | 218–306 MB | < 2 GB ✅ | After macOS reclaims load-time scratch |
| Time to first audio chunk | **2.82 s** | < 3 s ✅ | End-to-end through TTS |
| Generation throughput | 12–14 tok/s | — | Sustained, ~250 tokens generated |
| Prompt size (zh, full system+user) | 333 tokens | — | Includes a long Chinese system prompt |

## What about pure forward-pass latency?

If you remove the TTS pipeline overhead and measure only the model
forward pass:

| Phase | Approx time |
|---|---|
| Prefill (333 prompt tokens) | ~1.0 s |
| First token sampled | ~1.1 s |
| Each subsequent token | ~80 ms |

So the 2.82 s end-to-end first-audio number is dominated by the
prefill (1.0 s) plus sentence buffering (~0.6 s) plus TTS startup
(~1.2 s). The model itself produces tokens much faster than the
audio can be synthesized.

## Where the optimization headroom is

If you want to push first-chunk latency below 2 s, the levers are:

1. **Shorten the system prompt.** Each prompt token costs ~3 ms in
   prefill. A 100-token system prompt instead of 333 saves ~700 ms.
2. **Sentence boundary heuristics.** The TTS sentence splitter waits
   for a period or comma. Tighter heuristics (split on first 8 tokens
   if no comma) shave another ~200 ms.
3. **Speculative decoding.** Not implemented yet. Could roughly halve
   the per-token time if a smaller draft model is acceptable.
4. **KV cache quantization.** Not implemented yet. Reduces memory
   bandwidth proportionally; useful on lower-tier devices.

These are all on the roadmap but not in v0.1.0.

## Running benchmarks yourself

The pure-CPU benchmarks (no model required) live in the
`Benchmarks/` sub-package:

```bash
cd Benchmarks
swift run -c release benchmarks
```

They measure the parts of the library that don't need MLX
initialization or weight loading: configuration parsing, prompt
formatting, and the proportional rope construction. Useful for
catching unintentional regressions in the CPU-bound code paths.

End-to-end inference benchmarks have to be run inside an app on a
real device — there's no clean way to launch a 1.5 GB model from a
unit test. The example in `examples/BasicGeneration/` is the closest
thing to a CLI benchmark; it prints the first-chunk latency and the
sustained tok/s after generation finishes.
