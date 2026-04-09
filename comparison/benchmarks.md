# Benchmarks

Reproducible performance numbers for Gemma4SwiftCore. We publish these
because the [comparison page](../comparison/README.md) explicitly
**does not** claim throughput superiority over `mlx-swift-lm` — both
stacks share the same Apple MLX Metal kernels, so per-token throughput
is bounded by the same primitives. The numbers below exist so anyone
can audit the cost of using Gemma4SwiftCore in a real app, not so we
can make a marketing claim.

## Reference run

| Metric                       | Value         | Conditions                              |
|------------------------------|---------------|-----------------------------------------|
| Model                        | `mlx-community/gemma-4-e2b-it-4bit` | 4-bit quant, ~1.5 GB on disk |
| Cold load (download + init)  | ~110 s        | First run, fresh HuggingFace cache, Wi-Fi |
| Warm load (cache hit)        | ~6 s          | Second run, no network                  |
| Memory after load            | 341–392 MB    | Resident set, prompt unloaded           |
| Time to first audio chunk    | **2.82 s**    | End-to-end through TTS, 333-token system prompt |
| Generation throughput        | 12–14 tok/s   | Steady-state, post-prefill              |
| Maximum measured RAM         | < 2 GB        | Including OS overhead, with cache       |

**Hardware**: real iPhone, Apple A-series, 7.4 GB RAM. The 2.82 s
first-chunk number is end-to-end through the TTS pipeline of a
shipping app — pure forward-pass throughput is higher.

## Reproducing

The easiest way to reproduce throughput on your own hardware is the
verification executable:

```bash
GEMMA4_VERIFY_MAX_TOKENS=64 swift run -c release Gemma4Verify \
  "Tell me a short story about a curious fox."
```

This prints `TTFC`, total elapsed time, total tokens, and steady-state
`tok/s` to stderr in the final log line. The `-c release` flag is
critical — debug builds are 5–10× slower because MLX bounds checks
are not optimized out.

For end-to-end TTS-pipeline numbers like the 2.82 s figure, you need
your own audio decoder hooked up to the token stream — that part is
app-specific and out of scope for this package.

## What we deliberately do NOT publish

- **"Faster than mlx-swift-lm"** — same Metal backend, same kernels,
  same arithmetic. We are not faster, and we will not pretend to be.
- **Synthetic micro-benchmarks** — they are easy to game and tell you
  nothing about real app performance. The TTFC number is the
  end-to-end measurement that actually predicts user experience.
- **Comparisons against PyTorch / llama.cpp / CoreML** — different
  backends, different precision, different quant schemes. Apples to
  oranges. The honest answer is "use the one that fits your stack."

## What this directory will hold over time

- `release_notes/` — captured numbers for each tagged release, so
  performance regressions can be caught at PR time.
- `device_matrix/` — numbers from contributors running on different
  Apple Silicon devices. Submit yours via PR.

Until those land, the numbers above are the only authoritative
reference. They were measured on 2026-03-29 with `0.1.0`.
