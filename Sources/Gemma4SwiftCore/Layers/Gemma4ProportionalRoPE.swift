// Gemma4ProportionalRoPE.swift
//
// Custom RoPE layer for Gemma 4's "proportional" rope type, used by
// full_attention layers with `partial_rotary_factor < 1.0`.
//
// mlx-swift-lm 2.31.x's built-in `initializeRope` does not understand the
// `"proportional"` rope_type — it only handles `default` / `linear` /
// `llama3` / `yarn` / `longrope` / `mrope`. Without this class, naive
// callers pass the reduced dims (e.g. 128 = 512 * 0.25) directly to
// `initializeRope`, which produces wrong rotation frequencies because RoPE
// computes `freqs = factor * base^(2i/dims)` and `dims=128` produces
// dramatically different freq values from `dims=512`.
//
// This class matches Python's `mlx_lm.models.rope_utils.ProportionalRoPE`
// byte-for-byte:
//
//     exponents = arange(0, rotated_dims, 2) / dims
//     freqs     = concat([factor * base^exponents,
//                         [+inf] * ((dims - rotated_dims) / 2)])
//     output    = mlx.fast.rope(x, dims, freqs=freqs, ...)
//
// The `+inf` pads at the end of the freqs array mean those dims get no
// effective rotation: `cos(inf * t)` and `sin(inf * t)` collapse to a
// pass-through in the rope kernel.
//
// Reference: mlx-lm/mlx_lm/models/rope_utils.py — class ProportionalRoPE.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Custom Rotary Positional Embedding for partial-rotation attention.
///
/// Used by Gemma 4's full-attention layers, which only rotate the first
/// `dims * partial_rotary_factor` channels of each head and pass the rest
/// through unchanged. The unique part is the freqs construction: a real
/// `factor * base^exponents` prefix concatenated with `+inf` padding to
/// reach the full head dimension, so the underlying `MLXFast.RoPE` kernel
/// rotates the first chunk and no-ops on the rest.
///
/// Conforms to ``MLXLMCommon/OffsetLayer`` and
/// ``MLXLMCommon/ArrayOffsetLayer`` so it can drop into any place that
/// uses `RoPELayer` (e.g. inside `Gemma4TextAttention`).
@available(iOS 17.0, macOS 14.0, *)
public final class Gemma4ProportionalRoPE: Module, OffsetLayer, ArrayOffsetLayer {

    /// Total head dimension. Equal to `globalHeadDim` in Gemma 4 full layers.
    public let dims: Int

    /// Whether to use the "traditional" RoPE indexing (interleaved
    /// real/imag pairs) or the contiguous half-half layout. Gemma uses
    /// the latter, so this is `false` in practice.
    public let traditional: Bool

    /// Number of channels actually rotated. The remaining
    /// `dims - rotatedDims` channels pass through. For Gemma 4 E2B
    /// full layers this is `512 * 0.25 = 128`.
    public let rotatedDims: Int

    /// Pre-computed inverse frequencies. Length `dims / 2`.
    /// First `rotatedDims / 2` entries are `factor * base^(2i/dims)`.
    /// Remaining `(dims - rotatedDims) / 2` entries are `+inf`.
    private let _freqs: MLXArray

    /// Construct a proportional RoPE layer.
    ///
    /// - Parameters:
    ///   - dims: Full head dimension.
    ///   - rotatedDims: Number of channels to rotate. Must be even and
    ///     `<= dims`.
    ///   - traditional: RoPE indexing convention. Gemma uses `false`.
    ///   - base: Frequency base, e.g. `1_000_000` for Gemma 4 full attention.
    ///   - factor: Frequency scaling factor. Defaults to `1.0`.
    public init(
        dims: Int,
        rotatedDims: Int,
        traditional: Bool = false,
        base: Float = 10000.0,
        factor: Float = 1.0
    ) {
        precondition(dims % 2 == 0, "dims must be even for RoPE")
        precondition(rotatedDims % 2 == 0, "rotatedDims must be even")
        precondition(rotatedDims <= dims, "rotatedDims must be <= dims")

        self.dims = dims
        self.traditional = traditional
        self.rotatedDims = rotatedDims

        let realCount = rotatedDims / 2
        let padCount = (dims - rotatedDims) / 2

        let exponentValues: [Float] = (0 ..< realCount).map {
            Float($0 * 2) / Float(dims)
        }
        let realFreqValues = exponentValues.map { factor * powf(base, $0) }
        let realFreqs = MLXArray(realFreqValues)

        if padCount > 0 {
            let infFreqs = MLXArray([Float](repeating: Float.infinity, count: padCount))
            self._freqs = MLX.concatenated([realFreqs, infFreqs], axis: 0)
        } else {
            self._freqs = realFreqs
        }

        super.init()
    }

    /// Apply rope to `x` with an integer position offset.
    ///
    /// - Parameters:
    ///   - x: `[batch, n_heads, seq, head_dim]` query or key tensor.
    ///   - offset: Position of the first token in `x`. During the prompt
    ///     phase this is `0`; during generation it's the prompt length plus
    ///     the steps generated so far.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs)
    }

    /// `ArrayOffsetLayer` conformance — accepts an `MLXArray` offset
    /// (used by some KV-cache integrations). Falls back to the integer
    /// path because our attention module always works with `Int` offsets.
    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let intOffset = offset.item(Int.self)
        return callAsFunction(x, offset: intOffset)
    }
}
