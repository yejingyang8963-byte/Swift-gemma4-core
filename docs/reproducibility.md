# Reproducibility

This package makes one strong claim: the chat-template bypass in
`Gemma4PromptFormatter` produces token sequences **byte-for-byte
identical** to Python `mlx-lm`'s `tokenizer.apply_chat_template`. This
file shows you how to verify it yourself in under five minutes.

## What you need

- Any Mac with Apple Silicon (M1 or newer)
- Python 3.9 or newer
- Network access to HuggingFace (the tokenizer files are ~10 MB; the
  full model is 1.5 GB)
- A clone of this repository

The verification only needs the tokenizer, not the full model
weights — but the script below will download both unless you
specifically tell `mlx-lm` not to.

## Step 1 — Set up the Python baseline

```bash
python3 -m venv ~/.mlx-venv
source ~/.mlx-venv/bin/activate
pip install --upgrade pip
pip install mlx-lm
```

`mlx-lm` will install Apple's MLX runtime, the HuggingFace tokenizers
bindings, and a small CLI. Total download: about 200 MB.

## Step 2 — Run the baseline script

The repository ships `scripts/python_baseline.py`, which:

1. Loads the `mlx-community/gemma-4-e2b-it-4bit` model and tokenizer
2. Formats a fixed test prompt via `tokenizer.apply_chat_template`
3. Tokenizes the formatted string
4. Prints the resulting token IDs

```bash
cd Swift-gemma4-core
python scripts/python_baseline.py
```

Expected output:

```
Loading mlx-community/gemma-4-e2b-it-4bit ...
Formatted prompt:
<bos><|turn>user
Hello, what is your name?<turn|>
<|turn>model

Token IDs (first 20):
[2, 105, 2364, 107, 9259, 235269, 1212, 603, 861, 1503, 235336, 106, 107, 105, 4368, 107]

First 4 tokens correspond to:
  2     <bos>      (special)
  105   <|turn>    (special)
  2364  "user"
  107   "\n"
```

## Step 3 — Run the Swift equivalent

```bash
swift test --filter PromptFormattingTests
```

`PromptFormattingTests` asserts that `Gemma4PromptFormatter.userTurn`
produces the same string the Python `tokenizer.apply_chat_template`
produced. The string is then encoded by `tokenizer.encode(text:)`,
which respects the registered special tokens (`<|turn>`, `<turn|>`,
`<bos>` are all `special: true` in `tokenizer.json`).

For a deeper assertion that the encoded token IDs match, run the
opt-in network test:

```bash
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

This downloads the same tokenizer files Python uses, encodes our
formatter's output, and compares against the documented Python ground
truth byte-for-byte.

## Why this matters

`swift-jinja` 1.x renders Gemma 4's chat template incorrectly, dropping
5 tokens and mangling the system turn token id. The model receives a
sequence it has never seen during training and degrades to "fluent but
incoherent" output. We measured this on real hardware before we knew
the cause: the model produced grammatical Chinese sentences that
looped on phrase fragments and never followed user instructions.

The reproducibility procedure above is the **smoking gun** evidence
that the bug exists upstream and that our bypass actually fixes it.
Without it, anyone reading the README has to take our word that the
model produces good output. With it, anyone can verify in five
minutes that the token sequences match.

## Citing the verification

If you publish a paper or blog post that depends on this verification,
include the SHA of the commit you tested against:

```bash
git rev-parse HEAD
```

That commit, the `mlx-lm` Python version (from `pip show mlx-lm`),
and the `tokenizer.json` SHA (from
`huggingface-cli scan-cache | grep gemma-4`) together pin the
verification to a reproducible state.
