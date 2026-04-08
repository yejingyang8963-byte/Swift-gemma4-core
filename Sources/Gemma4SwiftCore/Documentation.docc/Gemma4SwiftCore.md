#  ``Gemma4SwiftCore``

Native Swift inference for Google Gemma 4 — runs on iPhone, iPad, and Mac.

## Overview

`Gemma4SwiftCore` is the first native Swift implementation of Google's
Gemma 4 text decoder. It plugs into Apple's
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) library as a
sidecar model registration, so you can load any HuggingFace Gemma 4 repo
(e.g. `mlx-community/gemma-4-e2b-it-4bit`) the same way you would load a
Llama or Qwen model — except now Gemma 4 actually works.

```swift
import Gemma4SwiftCore
import MLXLMCommon
import MLXLLM

// 1. Register the sidecar handler with mlx-swift-lm.
await Gemma4Registration.registerIfNeeded().value

// 2. Load the real 4-bit weights from HuggingFace.
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))

// 3. Format a prompt using the chat-template bypass (avoids
//    swift-jinja's broken Gemma 4 template renderer).
let prompt = Gemma4PromptFormatter.userTurn("Hello, what is your name?")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))

// 4. Generate.
let stream = try await container.generate(input: input, parameters: .init(maxTokens: 200))
for await event in stream {
    if case .chunk(let text) = event { print(text, terminator: "") }
}
```

## Why this exists

`mlx-swift-lm` 2.31.x has no Gemma 4 support. The model is significantly
different from Gemma 3 in five places: Per-Layer Embedding (PLE), KV
sharing across the back half of the decoder, a new "proportional" RoPE
variant, per-layer head dimensions, and a brand-new chat protocol with
custom turn-marker tokens. Borrowing the Gemma 3 implementation fails
at weight load with a missing-field error, and the swift-jinja chat
template renderer drops 5 of the 16 ground-truth tokens, sending the
model a sequence it has never seen during training — the result is
fluent gibberish that loops on phrase fragments.

`Gemma4SwiftCore` ports the entire text-tower decoder to Swift from
scratch, using only the public `mlx-swift` and `mlx-swift-lm` APIs, and
ships a chat-template bypass that produces token sequences identical to
Python `mlx-lm`'s `tokenizer.apply_chat_template`.

For the full architectural deep-dive, see <doc:Architecture>.

## Topics

### Getting started

- ``Gemma4SwiftCore``
- ``Gemma4Registration``
- ``Gemma4PromptFormatter``

### Configuration

- ``Gemma4TextConfiguration``
- ``Gemma4TextConfiguration/RopeParameters``

### Model

- ``Gemma4TextModel``
- ``Gemma4TextInner``
- ``Gemma4TextDecoderLayer``

### Layers

- ``Gemma4TextMLP``
- ``Gemma4TextAttention``
- ``Gemma4ProportionalRoPE``

### Articles

- <doc:Architecture>
- <doc:ProportionalRoPE>
- <doc:KVSharing>
- <doc:ChatTemplateBypass>
