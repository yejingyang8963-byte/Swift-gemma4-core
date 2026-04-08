// Gemma4TextAttention+Forward.swift
//
// Forward pass for `Gemma4TextAttention`. Lives in a separate file so the
// module definition stays under the project-wide 200-line limit and the
// numerical hot-path is easy to audit in isolation.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

@available(iOS 17.0, macOS 14.0, *)
extension Gemma4TextAttention {

    /// Attention forward pass returning both the output and the FULL cached
    /// keys/values so the caller can build a KV-donor table for shared layers.
    ///
    /// Critical: during generation the cache holds history from all previous
    /// tokens. Downstream shared layers need the FULL post-update K/V, not
    /// just the current step's freshly-computed tensors. Otherwise shared
    /// layers attend over only a 1-token window during generation, which
    /// collapses attention and produces drifting-garbage output.
    ///
    /// Flow:
    /// 1. Non-shared layer: project Q/K/V → norm → rope (at ropeOffset)
    ///    → cache.update (which appends new K/V and returns full history)
    ///    → attention over full K/V → return full K/V to caller for donor use.
    /// 2. Shared layer: receive donor K/V (already full history) →
    ///    rope queries at SAME ropeOffset as the donor → attention over
    ///    donor K/V → return donor K/V unchanged.
    ///
    /// `ropeOffset` matches Python `mlx-lm`'s threading of the
    /// pre-cache-update offset through the `intermediates` table: it is the
    /// position of the NEW tokens we're processing in this forward call,
    /// i.e., 0 during prompt phase and `L_prompt + steps_so_far` during
    /// generation. All layers in one forward call use the SAME ropeOffset.
    ///
    /// - Parameters:
    ///   - x: `[batch, seq, hidden_size]` input activations.
    ///   - mask: Attention mask mode (full / sliding window / none).
    ///   - cache: Optional KV cache for incremental generation.
    ///   - sharedKVDonor: When non-nil, skip K/V computation and reuse
    ///     these tensors. Used by KV-shared layers.
    ///   - ropeOffset: Global position offset, see discussion above.
    /// - Returns: `(output, keys, values)` — the layer output plus the
    ///   post-cache-update K/V so the inner model can stash them as a donor.
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        sharedKVDonor: (keys: MLXArray, values: MLXArray)?,
        ropeOffset: Int
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        queries = qNorm(queries)
        queries = rope(queries, offset: ropeOffset)

        let attnKeys: MLXArray
        let attnValues: MLXArray
        if let donor = sharedKVDonor {
            // Shared layer — donor already contains the full cache history.
            // Queries are rope'd at ropeOffset (passed from inner model),
            // which matches the donor's rope offset for this forward pass.
            attnKeys = donor.keys
            attnValues = donor.values
        } else {
            // Non-shared layer — compute fresh K/V, then update cache to get
            // the full history (previous cached K/V + this step's new K/V).
            var k = kProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
            k = kNorm(k)
            k = rope(k, offset: ropeOffset)

            var v = vProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
            v = vNormUnscaled(v)

            if let cache = cache {
                let (fullKeys, fullValues) = cache.update(keys: k, values: v)
                attnKeys = fullKeys
                attnValues = fullValues
            } else {
                attnKeys = k
                attnValues = v
            }
        }

        // Attention over the full (or donor) K/V history. We manage the
        // cache update ourselves above so we call scaledDotProductAttention
        // directly instead of attentionWithCacheUpdate (which would update
        // a second time).
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: attnKeys,
            values: attnValues,
            scale: scale,
            mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, nHeads * headDim)

        // Return the FULL attended K/V tensors so downstream KV-shared
        // layers of the same attention type can reuse them as their donor.
        return (output: oProj(attnOut), keys: attnKeys, values: attnValues)
    }

    /// Gemma 4 v_norm: RMS normalization WITHOUT a learnable scale.
    ///
    /// Equivalent to `x * rsqrt(mean(x**2, axis=-1, keepdims=true) + eps)`.
    /// The Python reference uses `RMSNormNoScale` for this — there is no
    /// learnable parameter, so no checkpoint weight is loaded.
    fileprivate func vNormUnscaled(_ x: MLXArray) -> MLXArray {
        let variance = (x * x).mean(axis: -1, keepDims: true)
        return x * MLX.rsqrt(variance + MLXArray(rmsNormEps))
    }
}
