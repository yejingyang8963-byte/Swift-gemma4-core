// Gemma4TextAttention.swift
//
// Multi-head attention for Gemma 4 text decoder layers.
//
// Five differences from a stock LLM attention block:
//
//   1. Per-layer head_dim: sliding layers use ``Gemma4TextConfiguration/headDim``,
//      full_attention layers use ``Gemma4TextConfiguration/globalHeadDim``.
//      The q/k/v/o projection widths therefore vary per layer index.
//   2. Per-layer-type RoPE: sliding layers use the standard initializeRope
//      with `theta=10000`, full layers use ``Gemma4ProportionalRoPE`` with
//      `theta=1e6` and `partial_rotary_factor=0.25`.
//   3. V-norm without scale: Gemma 4 applies an unscaled RMS norm to value
//      states (the original Python class is `RMSNormNoScale`). No learnable
//      parameter — implemented inline in the +Forward extension.
//   4. KV sharing: when `sharedKVDonor` is non-nil, skip k_proj/v_proj/k_norm
//      entirely and reuse the donor's keys/values.
//   5. Scaling = 1.0: Gemma 4 bakes the attention scaling into the q_norm
//      pathway, so the dot-product scale stays at 1.
//
// Implementation note: we manage the cache update ourselves and call
// MLXFast.scaledDotProductAttention directly (rather than going through the
// `attentionWithCacheUpdate` wrapper) so the post-update K/V history can be
// returned to the inner model and reused as a donor by downstream
// KV-shared layers.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

/// Multi-head attention block for one Gemma 4 text decoder layer.
///
/// Stores the projection layers and per-layer constants. The actual
/// forward pass lives in the `+Forward` extension.
@available(iOS 17.0, macOS 14.0, *)
public final class Gemma4TextAttention: Module {

    // MARK: - Per-layer constants

    public let layerIdx: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let ropeDims: Int
    public let isSliding: Bool
    public let slidingWindow: Int
    public let scale: Float
    public let rmsNormEps: Float

    // MARK: - Learnable submodules

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm
    @ModuleInfo public var rope: RoPELayer

    // MARK: - Construction

    /// Build the attention block for a specific decoder layer.
    ///
    /// - Parameters:
    ///   - config: Parsed text-tower configuration.
    ///   - layerIdx: Zero-based decoder layer index. Determines head_dim,
    ///     attention type (sliding vs full), and RoPE selection.
    public init(config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.headDimForLayer(layerIdx)
        self.isSliding = config.isSlidingLayer(layerIdx)
        self.slidingWindow = config.slidingWindow
        self.rmsNormEps = config.rmsNormEps
        // Gemma 4 bakes attention scaling into the q_norm pathway,
        // so the SDPA scale stays at 1.0.
        self.scale = 1.0

        let hidden = config.hiddenSize
        self._qProj.wrappedValue = Linear(hidden, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(nHeads * headDim, hidden, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // RoPE selection per attention type:
        //   sliding_attention → default rope, theta=10000, full rotation
        //   full_attention    → proportional rope, theta=1e6, partial=0.25
        let layerType = config.layerTypes[layerIdx]
        let ropeParams = config.ropeParameters[layerType]
        let ropeTheta = ropeParams?.ropeTheta ?? 10000.0
        let partialRotary = ropeParams?.partialRotaryFactor ?? 1.0
        let ropeTypeString = ropeParams?.ropeType ?? "default"
        self.ropeDims = Int(Float(headDim) * partialRotary)

        if ropeTypeString == "proportional" {
            // Custom proportional rope for Gemma 4 full-attention layers.
            // Passes the FULL head_dim with custom freqs (real values for the
            // rotated prefix, +inf padding for the skipped suffix).
            self.rope = Gemma4ProportionalRoPE(
                dims: headDim,
                rotatedDims: ropeDims,
                traditional: false,
                base: ropeTheta,
                factor: 1.0)
        } else {
            // Standard rope for sliding layers (full rotation across head_dim).
            self.rope = initializeRope(
                dims: headDim,
                base: ropeTheta,
                traditional: false,
                scalingConfig: nil,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
        }

        super.init()
    }
}
