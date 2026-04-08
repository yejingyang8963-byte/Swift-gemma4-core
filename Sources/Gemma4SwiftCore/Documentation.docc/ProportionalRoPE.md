# Proportional RoPE

A custom RoPE class for partial-rotation attention. Unique to Gemma 4
full-attention layers.

## Overview

Standard RoPE rotates every dimension of every attention head. Gemma 4
full-attention layers do something more interesting: they only rotate
the first `partial_rotary_factor` fraction of the head dimensions, and
pass the rest through unchanged. With `partial_rotary_factor = 0.25` and
`head_dim = 512`, that means only 128 of the 512 channels participate in
the rotational positional encoding.

This is implemented as a NEW `rope_type` value `"proportional"` in
HuggingFace `config.json`. `mlx-swift-lm`'s built-in `initializeRope`
helper does not understand this type — passing it would either crash or
silently fall through to a default rope with the wrong frequencies.

`Gemma4SwiftCore` ships ``Gemma4ProportionalRoPE`` to handle this case.

## The frequency formula

```python
# Python reference (mlx_lm/models/rope_utils.py:215-220)
exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
freqs = mx.concatenate([
    factor * (base ** exponents),
    mx.full(((dims - rotated_dims) // 2,), mx.inf),
])
```

```swift
// Swift port (Sources/Gemma4SwiftCore/Layers/Gemma4ProportionalRoPE.swift)
let realCount = rotatedDims / 2
let padCount = (dims - rotatedDims) / 2
let exponentValues: [Float] = (0 ..< realCount).map {
    Float($0 * 2) / Float(dims)
}
let realFreqValues = exponentValues.map { factor * powf(base, $0) }
let realFreqs = MLXArray(realFreqValues)
let infFreqs = MLXArray([Float](repeating: .infinity, count: padCount))
self._freqs = MLX.concatenated([realFreqs, infFreqs], axis: 0)
```

The two implementations produce identical `_freqs` arrays. The Swift
version then defers to `MLXFast.RoPE` exactly the same way the Python
version defers to `mlx.fast.rope`.

## Why the +inf padding works

`MLXFast.RoPE` evaluates `cos(freq * t)` and `sin(freq * t)` per channel.
For `freq = +inf` these collapse to a no-op in the kernel — the channel
passes through with its original value, regardless of position. This is
how the unrotated suffix works without requiring a separate code path.

If you imagine the freqs array for a Gemma 4 E2B full-attention layer
(`dims = 512`, `rotated_dims = 128`, `base = 1e6`), the first 64 entries
are real frequencies in the range `[1.0, ~10.6]` and the last 192
entries are all `+inf`. The first 128 channels of each head get
positional information; the last 384 don't.

## Verifying it matches Python

`Tests/Gemma4SwiftCoreTests/ProportionalRoPETests.swift` reproduces the
formula in pure Swift and asserts the first few real frequencies match
the expected values within 1e-3 absolute tolerance. The test is fast
(<1 ms) and runs on every CI build.

For end-to-end byte-level verification, see the `scripts/python_baseline.py`
script in the repository root.
