#!/usr/bin/env python3
# python_baseline.py — generate the Python ground truth for the
# chat-template bypass that Gemma4SwiftCore tests assert against.
#
# Loads mlx-community/gemma-4-e2b-it-4bit, formats a fixed neutral
# prompt via tokenizer.apply_chat_template, and dumps the resulting
# token IDs side-by-side with the Swift formatter's expected output.
#
# Usage:
#     python3 -m venv ~/.mlx-venv
#     source ~/.mlx-venv/bin/activate
#     pip install mlx-lm
#     python scripts/python_baseline.py
#
# Apple Silicon required (mlx-lm only ships arm64 wheels). The first
# run downloads ~10 MB of tokenizer files (we don't actually need the
# 1.5 GB model weights for this verification, but mlx-lm.load fetches
# both).

from __future__ import annotations

import sys
from typing import List

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

# Neutral test prompt — must NOT contain anything that could be
# interpreted as referring to a private project or user.
TEST_PROMPT = "Hello, what is your name?"


def main() -> int:
    try:
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx-lm is not installed.", file=sys.stderr)
        print("       Install it with: pip install mlx-lm", file=sys.stderr)
        return 1

    print(f"Loading {MODEL_ID} ...")
    print("(this downloads tokenizer + weights on the first run)")
    print()

    _, tokenizer = load(MODEL_ID)

    # 1. Build the canonical chat-template formatted string.
    messages = [{"role": "user", "content": TEST_PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # 2. Tokenize.
    token_ids: List[int] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    print("Formatted prompt:")
    print("-" * 70)
    print(formatted)
    print("-" * 70)
    print()

    print(f"Token count: {len(token_ids)}")
    print()
    print("Token IDs:")
    print(token_ids)
    print()

    print("First 10 tokens decoded individually:")
    for i, tid in enumerate(token_ids[:10]):
        decoded = tokenizer.decode([tid])
        print(f"  [{i:>2}] {tid:>6}  {repr(decoded)}")
    print()

    # 3. Show the Swift formatter equivalent for direct comparison.
    swift_formatted = (
        f"<bos><|turn>user\n{TEST_PROMPT}<turn|>\n<|turn>model\n"
    )
    print("Swift Gemma4PromptFormatter.userTurn output (literal string):")
    print("-" * 70)
    print(swift_formatted)
    print("-" * 70)
    print()

    swift_token_ids = tokenizer.encode(swift_formatted, add_special_tokens=False)
    print("Swift formatter, encoded:")
    print(swift_token_ids)
    print()

    if swift_token_ids == token_ids:
        print("✅ MATCH — Swift formatter and Python apply_chat_template produce")
        print("           identical token sequences. The bypass is verified.")
        return 0
    else:
        print("❌ MISMATCH — token sequences differ. Investigate before shipping.")
        print(f"   python: {token_ids}")
        print(f"   swift:  {swift_token_ids}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
