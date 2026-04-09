#!/usr/bin/env python3
# numerical_parity.py — dump Python mlx-lm Gemma 4 ground truth so the
# Swift Gemma4SwiftCore implementation can be compared layer-by-layer.
#
# This is the strict version of `python_baseline.py`. Where the baseline
# only checks token-level parity (i.e. "do we encode the same prompt to
# the same token IDs as Python?"), this script also captures:
#
#   1. Top-1 token IDs at each generation step for a fixed seed.
#   2. The first row of the final logits tensor for the prompt forward
#      pass — enough to detect any silent kernel divergence.
#
# Outputs are written to `comparison/parity_python.json`. A future
# `Sources/Gemma4Verify` flag will produce the matching Swift artifact
# at `comparison/parity_swift.json`, and a third script will diff them
# with `numpy.allclose(atol=1e-3, rtol=1e-3)` thresholds.
#
# Why this is needed even though `python_baseline.py` already passes:
# token-level parity proves the prompt path is right. It does NOT prove
# the forward pass is numerically correct — a port can produce the
# right token IDs and still drift in the logits, which manifests as
# subtle long-context degradation that unit tests miss.
#
# Usage:
#     python3 -m venv ~/.mlx-venv
#     source ~/.mlx-venv/bin/activate
#     pip install mlx-lm
#     python3 scripts/numerical_parity.py
#
# Apple Silicon required. ~1.5 GB download on first run (cached after).
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"
PROMPT = "Hello, what is your name?"
NUM_GREEDY_STEPS = 16
SEED = 0
OUT_PATH = Path(__file__).resolve().parent.parent / "comparison" / "parity_python.json"


def main() -> int:
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx-lm is not installed (pip install mlx-lm).", file=sys.stderr)
        return 1

    print(f"Loading {MODEL_ID} …")
    model, tokenizer = load(MODEL_ID)

    # ─── 1. Encode the prompt via the chat template ────────────────
    messages = [{"role": "user", "content": PROMPT}]
    prompt_token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    print(f"Prompt encodes to {len(prompt_token_ids)} tokens.")

    # ─── 2. Forward the prompt and capture the final-position logits row
    #       (first 32 entries — enough fingerprint, no need to dump all 256k)
    mx.random.seed(SEED)
    inputs = mx.array(prompt_token_ids)[None]  # shape (1, L)
    out = model(inputs)
    if hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    last_logits = logits[0, -1].astype(mx.float32)
    last_logits_head = [float(x) for x in last_logits[:32].tolist()]
    print(f"Captured first-32 logits at last position. Head: {last_logits_head[:4]}")

    # ─── 3. Greedy-sample NUM_GREEDY_STEPS tokens, deterministic ────
    #       This is the canonical "did the port drift?" smoke test.
    greedy_tokens: list[int] = []
    cur = inputs
    for step in range(NUM_GREEDY_STEPS):
        out = model(cur)
        logits = out.logits if hasattr(out, "logits") else out
        next_id = int(mx.argmax(logits[0, -1]).item())
        greedy_tokens.append(next_id)
        cur = mx.concatenate([cur, mx.array([[next_id]])], axis=1)
    greedy_text = tokenizer.decode(greedy_tokens)
    print(f"Greedy {NUM_GREEDY_STEPS} tokens: {greedy_tokens}")
    print(f"Decoded                  : {greedy_text!r}")

    # ─── 4. Persist the artifact ────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "seed": SEED,
        "prompt_token_ids": list(prompt_token_ids),
        "last_logits_head_f32": last_logits_head,
        "greedy_token_ids": greedy_tokens,
        "greedy_decoded_text": greedy_text,
        "num_greedy_steps": NUM_GREEDY_STEPS,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nWrote ground truth → {OUT_PATH.relative_to(Path.cwd())}")
    print()
    print("Next step (manual, for now):")
    print("  1. Run the Swift dumper once it lands:")
    print("       swift run Gemma4Verify --dump-parity comparison/parity_swift.json")
    print("  2. Diff the two JSONs with numpy.allclose, atol/rtol = 1e-3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
