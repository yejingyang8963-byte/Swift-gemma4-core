// Gemma4TextConfiguration+Derived.swift
//
// Computed properties and helper queries on top of the parsed
// `Gemma4TextConfiguration`. Kept in a separate file so the raw
// JSON-decodable surface stays small and easy to audit.
//
// SPDX-License-Identifier: MIT

import Foundation

extension Gemma4TextConfiguration {

    /// Index of the first KV-shared layer.
    ///
    /// Layers in `[0, firstKvSharedLayerIdx)` compute their own K/V tensors.
    /// Layers in `[firstKvSharedLayerIdx, hiddenLayers)` reuse K/V from a
    /// same-attention-type donor in the non-shared range. The split point is
    /// `hiddenLayers - numKvSharedLayers`, clamped to `>= 0` for safety.
    public var firstKvSharedLayerIdx: Int {
        max(0, hiddenLayers - numKvSharedLayers)
    }

    /// Whether the layer at `layerIdx` reuses K/V from a donor.
    ///
    /// - Parameter layerIdx: Zero-based decoder layer index.
    /// - Returns: `true` iff KV sharing is enabled and `layerIdx >= firstKvSharedLayerIdx`.
    public func isKvSharedLayer(_ layerIdx: Int) -> Bool {
        numKvSharedLayers > 0 && layerIdx >= firstKvSharedLayerIdx
    }

    /// Whether `layerIdx` uses the sliding-window attention variant.
    ///
    /// Sliding layers attend over a fixed `slidingWindow` of recent tokens
    /// (cached via `RotatingKVCache`). The opposite is `"full_attention"`,
    /// which attends over the entire sequence (via `StandardKVCache`).
    public func isSlidingLayer(_ layerIdx: Int) -> Bool {
        layerTypes[layerIdx] == "sliding_attention"
    }

    /// Look up the donor layer for a KV-shared layer.
    ///
    /// Walks `[0, firstKvSharedLayerIdx)` in reverse and returns the index
    /// of the most recent non-shared layer with the same attention type.
    /// Returns `nil` if `layerIdx` is not actually a KV-shared layer or no
    /// matching donor exists (which would indicate a malformed config).
    ///
    /// - Parameter layerIdx: Zero-based index of a KV-shared layer.
    /// - Returns: Donor layer index, or `nil` if not applicable.
    public func kvDonorLayerIndex(forSharedLayer layerIdx: Int) -> Int? {
        guard isKvSharedLayer(layerIdx) else { return nil }
        let myType = layerTypes[layerIdx]
        for i in stride(from: firstKvSharedLayerIdx - 1, through: 0, by: -1) where layerTypes[i] == myType {
            return i
        }
        return nil
    }

    /// Per-layer attention head dimension.
    ///
    /// Full-attention layers use ``globalHeadDim`` (e.g. 512 in E2B);
    /// sliding-attention layers use ``headDim`` (e.g. 256 in E2B). The
    /// number of heads stays constant across both types — only the head
    /// width differs.
    public func headDimForLayer(_ layerIdx: Int) -> Int {
        layerTypes[layerIdx] == "full_attention" ? globalHeadDim : headDim
    }

    /// Per-layer MLP intermediate width.
    ///
    /// When ``useDoubleWideMlp`` is `true`, KV-shared layers double their
    /// intermediate size (so the first 15 of 35 layers stay at
    /// `intermediate_size = 6144` and the last 20 use 12288 in E2B).
    public func intermediateSizeForLayer(_ layerIdx: Int) -> Int {
        let doubled = useDoubleWideMlp && isKvSharedLayer(layerIdx)
        return doubled ? intermediateSize * 2 : intermediateSize
    }
}
