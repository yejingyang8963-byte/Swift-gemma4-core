# KV Sharing

How Gemma 4 reuses K/V tensors across layers, and the global rope offset
trick that keeps it numerically correct.

## Overview

Gemma 4 splits its decoder into two regions:

- **Compute region** — layers `[0, firstKvSharedLayerIdx)`. These layers
  compute their own K/V like a normal transformer.
- **Reuse region** — layers `[firstKvSharedLayerIdx, hiddenLayers)`.
  These layers do NOT compute K/V at all. They reuse the K/V tensors
  from the most recent earlier layer of the same attention type
  (sliding vs full).

In Gemma 4 E2B, that's 15 compute layers and 20 reuse layers — almost
60% of the decoder shares K/V. The savings on memory bandwidth and
prefill latency are substantial: every reuse layer skips three matrix
multiplies (`k_proj`, `v_proj`, and the rope on K) and reads from a
cache tensor that was already in the GPU's working set.

## The donor table

`Gemma4TextInner.callAsFunction` walks layers in order and maintains a
dictionary of "donors":

```swift
var kvDonors: [Int: (keys: MLXArray, values: MLXArray)] = [:]

for (i, layer) in layers.enumerated() {
    var donor: (keys: MLXArray, values: MLXArray)? = nil
    if let donorIdx = config.kvDonorLayerIndex(forSharedLayer: i) {
        donor = kvDonors[donorIdx]   // populated by an earlier iteration
    }

    let result = layer(hidden, ..., sharedKVDonor: donor, ropeOffset: globalRopeOffset)
    hidden = result.output

    if !config.isKvSharedLayer(i) {
        kvDonors[i] = (keys: result.keys, values: result.values)
    }
}
```

The donor lookup uses ``Gemma4TextConfiguration/kvDonorLayerIndex(forSharedLayer:)``,
which walks the non-shared range in reverse and returns the index of the
most recent layer with the same `layer_type`.

## Critical: the global rope offset

The donor mechanism above is mostly straightforward. The subtle part is
where the rope offset comes from.

In Python `mlx-lm`, each non-shared layer captures `cache.offset` BEFORE
calling `cache.update_and_fetch`, then threads that offset through the
`intermediates[]` table alongside the K/V tensors. Shared layers read
the donor's pre-update offset and rope their queries at that position.

The naive Swift port — read each layer's own `cache.offset` — has a
silent bug: the shared layers' caches are never actually updated, so
their `offset` is permanently `0`, and their queries get rope'd at
position 0 during generation. The model still emits fluent local tokens
but loses all sense of position; it loops on phrase fragments and
produces incoherent output.

The fix is to compute a SINGLE global rope offset at the top of the
forward pass, BEFORE any cache update happens, by reading the offset of
the first non-shared layer:

```swift
let globalRopeOffset: Int = {
    for i in 0 ..< config.hiddenLayers
    where !config.isKvSharedLayer(i) {
        if let c = layerCache[i] { return c.offset }
    }
    return 0
}()
```

All layers in one forward call use the same value. During the prompt
phase it is `0`. During generation it is `L_prompt + steps_so_far`
because every prior forward pass advanced every non-shared cache by the
same amount.

## See Also

- ``Gemma4TextInner``
- ``Gemma4TextConfiguration/kvDonorLayerIndex(forSharedLayer:)``
- ``Gemma4TextAttention/callAsFunction(_:mask:cache:sharedKVDonor:ropeOffset:)``
