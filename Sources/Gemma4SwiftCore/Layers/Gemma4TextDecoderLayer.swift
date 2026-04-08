// Gemma4TextDecoderLayer.swift
//
// One decoder layer of the Gemma 4 text tower. Wraps attention + MLP in
// the standard pre/post layernorm sandwich, then adds two Gemma 4–specific
// blocks at the end:
//
//   - Per-Layer Embedding (PLE): a small per-token vector flows in, gets
//     gated by `gelu_approx(per_layer_input_gate(h)) * pli`, projected back
//     to hidden_size, normalized, and added as a third residual.
//   - Layer scalar: the entire layer output is multiplied by a single
//     learned scalar (shape `[1]`).
//
// The layer returns keys/values alongside the output so the inner model
// can collect them into a donor table for KV-shared layers downstream.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

/// One decoder layer of the Gemma 4 text tower.
@available(iOS 17.0, macOS 14.0, *)
public final class Gemma4TextDecoderLayer: Module {

    public let layerIdx: Int
    public let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") public var selfAttn: Gemma4TextAttention
    @ModuleInfo public var mlp: Gemma4TextMLP

    @ModuleInfo(key: "input_layernorm") public var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFFNNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFFNNorm: RMSNorm

    // Per-Layer Embedding (PLE) submodules — unique to Gemma 4.
    @ModuleInfo(key: "per_layer_input_gate") public var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") public var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") public var postPerLayerInputNorm: RMSNorm

    /// Learned scalar multiplier applied to the whole layer output.
    /// Shape `[1]`. Initialized to 1.0 so the first forward pass is a no-op.
    @ParameterInfo(key: "layer_scalar") public var layerScalar: MLXArray

    /// Construct a decoder layer at `layerIdx`.
    ///
    /// - Parameters:
    ///   - config: Parsed text-tower configuration.
    ///   - layerIdx: Zero-based index. Determines attention type, head_dim,
    ///     and MLP intermediate width.
    public init(config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._selfAttn.wrappedValue = Gemma4TextAttention(config: config, layerIdx: layerIdx)
        self.mlp = Gemma4TextMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSizeForLayer(layerIdx))

        let h = config.hiddenSize
        let eps = config.rmsNormEps
        self._inputNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        self._postAttentionNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        self._preFFNNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        self._postFFNNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)

        self._perLayerInputGate.wrappedValue = Linear(h, config.hiddenSizePerLayerInput, bias: false)
        self._perLayerProjection.wrappedValue = Linear(config.hiddenSizePerLayerInput, h, bias: false)
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)

        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    /// Forward pass — see ``Documentation.docc/Architecture.md`` for the
    /// full diagram.
    ///
    /// - Parameters:
    ///   - x: `[batch, seq, hidden_size]` input activations.
    ///   - perLayerInput: PLE input slice for this layer
    ///     `[batch, seq, hidden_size_per_layer_input]`. Pass `nil` to skip
    ///     the PLE block (e.g. for ablation studies).
    ///   - mask: Attention mask mode (full or sliding window).
    ///   - cache: Optional KV cache for incremental generation.
    ///   - sharedKVDonor: When non-nil, attention reuses these K/V tensors.
    ///   - ropeOffset: Global token-position offset for this forward call.
    /// - Returns: `(output, keys, values)` where keys/values are the
    ///   post-cache-update tensors used by attention, exposed so the inner
    ///   model can stash them as a donor for downstream KV-shared layers.
    public func callAsFunction(
        _ x: MLXArray,
        perLayerInput: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        sharedKVDonor: (keys: MLXArray, values: MLXArray)?,
        ropeOffset: Int
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        // Attention block
        var residual = x
        var h = inputNorm(x)
        let attnResult = selfAttn(
            h,
            mask: mask,
            cache: cache,
            sharedKVDonor: sharedKVDonor,
            ropeOffset: ropeOffset)
        h = postAttentionNorm(attnResult.output)
        h = residual + h

        // FFN block
        residual = h
        h = preFFNNorm(h)
        h = mlp(h)
        h = postFFNNorm(h)
        h = residual + h

        // PLE block (Gemma 4 only)
        if hiddenSizePerLayerInput > 0, let pli = perLayerInput {
            residual = h
            h = perLayerInputGate(h)
            h = geluApproximate(h)
            h = h * pli
            h = perLayerProjection(h)
            h = postPerLayerInputNorm(h)
            h = residual + h
        }

        // Layer scalar
        h = h * layerScalar

        return (output: h, keys: attnResult.keys, values: attnResult.values)
    }
}
