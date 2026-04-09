# Comparison: Gemma4SwiftCore vs adrgrondin/mlx-swift-lm

This directory holds the **reproducible evidence** behind every claim
Gemma4SwiftCore makes about its position in the Swift LLM ecosystem.
Nothing here is hand-waved — every cell in the table below links to a
script you can run yourself.

## TL;DR

| Dimension | Gemma4SwiftCore | adrgrondin/mlx-swift-lm | Evidence |
|---|:---:|:---:|---|
| Models supported | **1** (Gemma 4) | 52 (general framework) | [registry audit](upstream_registry_audit.md) |
| Gemma 4 support | ✅ | ❌ | [upstream_attempt.sh](upstream_attempt.sh) |
| Gemma 3n support | ❌ | ✅ | upstream `gemma3n` registered |
| Per-Layer Embedding (Gemma 4) | ✅ | n/a | `Sources/Gemma4SwiftCore/Model/Gemma4TextInner+PerLayerInputs.swift` |
| KV-sharing donor table | ✅ | n/a | `Gemma4TextInner.swift:142` |
| Proportional RoPE | ✅ byte-exact | n/a | `Layers/Gemma4ProportionalRoPE.swift` |
| Chat-template bypass with Python parity test | ✅ | ❌ | [`scripts/python_baseline.py`](../scripts/python_baseline.py) |
| Numerical parity vs Python ground truth | ✅ token-level<br>🟡 logits (planned) | unknown | [`scripts/numerical_parity.py`](../scripts/numerical_parity.py) |
| Per-file 200-line cap, full DocC | ✅ | partial | inspect any source file |
| Published, reproducible benchmarks | ✅ | partial | [`benchmarks.md`](benchmarks.md) |

## What this directory does NOT claim

- **It does not claim Gemma4SwiftCore is "better" overall.** A 1-model
  package cannot beat a 52-model framework on breadth, throughput
  parity, or community size. We are not those things and we do not
  pretend to be.
- **It does not claim faster generation.** Both stacks share the same
  Apple MLX Metal backend. Per-token throughput is bounded by the same
  kernels. We measure it (see [`benchmarks/`](../benchmarks/)) so the
  number is auditable, not promotional.
- **It does not claim stability.** Gemma4SwiftCore is at v0.1.0.
  Production claims require time, not adjectives.

## What this directory DOES claim, and the proof for each claim

### Claim 1: Gemma4SwiftCore is the only Swift package that runs Gemma 4

**Evidence**: [`upstream_registry_audit.md`](upstream_registry_audit.md)
captures the full list of model types registered in
`adrgrondin/mlx-swift-lm`'s `LLMTypeRegistry.shared`. There are 52
entries. Searching for `gemma4` returns zero matches. Searching for
`gemma3n` returns one match (a different model — `gemma3n` is "Gemma 3
nano", structurally distinct from `gemma4`).

**Reproducer**: [`upstream_attempt.sh`](upstream_attempt.sh) clones the
fork, greps the registry source, and prints the diff. Run it yourself.

### Claim 2: Token-level chat-template parity with Python is byte-exact

**Evidence**: [`../scripts/python_baseline.py`](../scripts/python_baseline.py)
loads the same HuggingFace tokenizer Gemma4SwiftCore uses, formats the
same prompt via Python `mlx_lm.tokenizer.apply_chat_template`, and
compares the resulting token IDs against
`Gemma4PromptFormatter.userTurn`. The script exits 0 if and only if
the two sequences are identical, or exits 2 with the diff.

**Why this matters**: swift-jinja 1.x mis-renders Gemma 4's chat
template, dropping the system turn and corrupting the second token id.
A model fed a wrong prompt produces fluent garbage. This is the most
common silent-failure mode in Swift LLM ports, and we test for it
explicitly.

### Claim 3: We document *why* every architectural deviation exists

Read any source file in `Sources/Gemma4SwiftCore/`. Every header
comment explains the Python reference, the deviation from a stock LLM,
and the failure mode the deviation prevents. Examples worth opening:

- `Layers/Gemma4ProportionalRoPE.swift` — explains why the naive
  `initializeRope(dims: rotatedDims)` produces wrong frequencies and
  links to the Python source it mirrors.
- `Layers/Gemma4TextAttention.swift` — five-point list of differences
  from a stock attention block.
- `Prompt/Gemma4PromptFormatter.swift` — verbatim Python ground truth
  vs. swift-jinja output, and the precise failure mode each diff
  causes downstream.

This is not a quality dimension we can express as a number — but it is
something you can verify in 60 seconds by reading two files.

## How to run the full evidence pack

```bash
# 1. Verify the upstream gap is real (no model download).
bash comparison/upstream_attempt.sh

# 2. Verify our token-level parity (downloads ~10 MB tokenizer).
python3 -m venv ~/.mlx-venv && source ~/.mlx-venv/bin/activate
pip install mlx-lm
python3 scripts/python_baseline.py

# 3. (Optional, slow) Run the Swift end-to-end smoke test.
swift run Gemma4Verify "Hello, what is your name?"
```

If any step disagrees with what's claimed in this README, that is a
bug. File an issue with the captured output.
