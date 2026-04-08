# Chat Template Bypass

Why you must NOT use `tokenizer.applyChatTemplate` for Gemma 4 inputs,
and how ``Gemma4PromptFormatter`` works around the bug.

## Overview

Gemma 4 ships an entirely new chat protocol — see
``Gemma4PromptFormatter`` for the format. The interesting story is what
happens when you try to use the canonical path through swift-jinja:

```swift
// ❌ Broken on Gemma 4 — DO NOT use this path.
let messages = [["role": "user", "content": "TEST"]]
let tokens = try tokenizer.applyChatTemplate(messages: messages)
```

## What goes wrong

Gemma 4's `tokenizer_config.json` ships a complex jinja2 chat template
that uses macros and namespace variables to handle thinking mode and
multi-turn conversations. As of swift-jinja 1.x (the renderer used by
swift-transformers), this template renders incorrectly.

Side-by-side comparison for `[{"role": "user", "content": "TEST"}]`:

| Source | Token count | Token IDs |
|---|---:|---|
| Python `apply_chat_template` | **16** | `[2, 105, 9731, 107, 98, 106, 107, 105, 2364, 107, 20721, 106, 107, 105, 4368, 107]` |
| swift-jinja `applyChatTemplate` | **11** | `[2, 108, 105, 2364, 107, 20721, 106, 107, 105, 4368, 107]` |
| Difference | **−5 tokens, id mismatch** | system turn (5 tokens) dropped; second token id is 108 instead of 105 |

The 5 missing tokens are the system turn (`<|turn>system\n<|think|><turn|>\n`).
The model never saw token id 108 in the second position during training,
so its first attention pass operates on noise. The output is locally
fluent (the embedding layer still maps token IDs to reasonable vectors)
but globally garbage — the model loops on phrase fragments and never
follows the user's instruction.

## The fix

Bypass `applyChatTemplate` entirely. Build the prompt as a literal
string with the turn markers as text, then call `tokenizer.encode(text:)`,
which respects the special-token entries in `tokenizer.json`:

```swift
// ✅ Correct path.
let prompt = Gemma4PromptFormatter.userTurn("Hello, what is your name?")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))
```

The literal string is:

```
<bos><|turn>user
Hello, what is your name?<turn|>
<|turn>model

```

`tokenizer.encode(text:)` recognizes `<bos>`, `<|turn>`, and `<turn|>`
as registered special tokens (verified in `tokenizer.json`: ids 2, 105,
106, all with `special: true`) and produces:

```
[2, 105, 2364, 107, 20071, 7058, 7197, ..., 106, 107, 105, 4368, 107]
 ^   ^    ^     ^    ^------ message body ------^   ^   ^    ^     ^
 bos turn user  \n                                  t|  \n  turn model \n
```

This sequence is identical to what Python's `apply_chat_template`
produces for the same input — verified end-to-end against `mlx-lm`
0.31.2 on Apple Silicon.

## Disabling thinking mode

Python `apply_chat_template` always injects `<|think|>` into the system
turn, which causes the model to emit a `<|channel>thought ... <channel|>`
block before the actual response. For most app contexts (TTS, real-time
generation, anything where you want the model to start producing the
final output immediately) you do NOT want this.

``Gemma4PromptFormatter/userTurn(_:)`` deliberately omits the system
turn, so the model goes straight to producing the assistant response.
If you DO want chain-of-thought reasoning, use
``Gemma4PromptFormatter/userTurnWithThinking(_:includeThinking:)``
instead.

## See Also

- ``Gemma4PromptFormatter``
- ``Gemma4PromptFormatter/userTurn(_:)``
- ``Gemma4PromptFormatter/userTurnWithThinking(_:includeThinking:)``
