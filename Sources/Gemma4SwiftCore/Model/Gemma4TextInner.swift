// Gemma4TextInner.swift
//
// Inner decoder stack of the Gemma 4 text tower. Owns:
//
//   - embed_tokens                 token → hidden_size
//   - embed_tokens_per_layer       token → num_layers * hidden_size_per_layer_input
//   - per_layer_model_projection   hidden_size → num_layers * hidden_size_per_layer_input
//   - per_layer_projection_norm    RMSNorm on hidden_size_per_layer_input
//   - layers                       num_hidden_layers × Gemma4TextDecoderLayer
//   - norm                         final RMSNorm on hidden_size
//
// Forward pass (ported from Python `Gemma4TextModel.__call__`):
//
//   1. Embed tokens, apply sqrt(hidden_size) scale.
//   2. Compute per-layer inputs via the lookup-table path + projection
//      path, combined with the PLE input scale (2^-0.5). See the
//      `+PerLayerInputs` extension for the implementation.
//   3. Build two attention masks (sliding + full) so each layer gets the
//      right one based on its declared attention type.
//   4. Walk decoder layers. For each layer:
//        - pick its per-layer input slice
//        - pick the right attention mask + cache
//        - if it's a KV-shared layer, fetch donor (keys, values) from the
//          `kvDonors` dictionary keyed by donor layer index
//        - call the decoder layer with `ropeOffset = globalRopeOffset`
//        - if it's a non-shared layer, stash its keys/values into kvDonors
//   5. Apply the final norm.
//
// `globalRopeOffset` is the position of the new tokens being processed in
// this forward call. During the prompt phase it is `0`. During generation
// it is read from the first non-shared cache's `.offset` BEFORE any cache
// update happens, so it equals `L_prompt + steps_so_far`. All layers in
// one forward call use the same offset.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN

/// Inner decoder stack of the Gemma 4 text tower.
@available(iOS 17.0, macOS 14.0, *)
public final class Gemma4TextInner: Module {

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") public var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") public var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") public var perLayerProjectionNorm: RMSNorm
    @ModuleInfo public var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    public let config: Gemma4TextConfiguration
    let hiddenSizeSqrt: Float
    let perLayerEmbedScale: Float
    let perLayerModelProjectionScale: Float
    let perLayerInputScale: Float

    /// Build the inner stack from a parsed configuration.
    public init(config: Gemma4TextConfiguration) {
        self.config = config
        self.hiddenSizeSqrt = sqrtf(Float(config.hiddenSize))
        self.perLayerEmbedScale = sqrtf(Float(config.hiddenSizePerLayerInput))
        self.perLayerModelProjectionScale = powf(Float(config.hiddenSize), -0.5)
        self.perLayerInputScale = powf(2.0, -0.5)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize)

        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: config.vocabSizePerLayerInput,
            dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput)

        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenLayers * config.hiddenSizePerLayerInput,
            bias: false)

        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSizePerLayerInput,
            eps: config.rmsNormEps)

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Gemma4TextDecoderLayer(config: config, layerIdx: $0)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Run the decoder stack on input token IDs.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` int32 token IDs.
    ///   - cache: Optional per-layer KV caches. Must be the same length as
    ///     ``layers`` if provided.
    /// - Returns: `[batch, seq, hidden_size]` final normalized activations.
    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        // 1. Embed and scale.
        var hidden = embedTokens(inputIds)
        hidden = hidden * MLXArray(hiddenSizeSqrt).asType(hidden.dtype)

        // 2. Per-layer inputs (see +PerLayerInputs.swift).
        let perLayerInputs = computePerLayerInputs(inputIds: inputIds, inputsEmbeds: hidden)

        // 3. Build masks for each attention type. Reuse the first cache of
        //    each type as the offset source.
        let layerCache = cache ?? Array(repeating: nil, count: config.hiddenLayers)
        let firstSlidingIdx = config.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        let firstFullIdx = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        let slidingMask = createAttentionMask(
            h: hidden,
            cache: layerCache[firstSlidingIdx],
            windowSize: config.slidingWindow)
        let fullMask = createAttentionMask(
            h: hidden,
            cache: layerCache[firstFullIdx])

        // Compute the global rope offset BEFORE any cache update happens.
        // During the prompt phase, all caches are empty → 0.
        // During generation, all non-shared caches have already been
        // advanced to L_prompt + steps_so_far from prior forward passes,
        // so reading any non-shared layer's cache.offset before update()
        // gives us the correct position for NEW tokens in this pass.
        // Critical: shared layers' own caches are never updated, so they
        // would always report offset=0 — we must NOT read from them.
        let globalRopeOffset: Int = {
            for i in 0 ..< config.hiddenLayers
            where !config.isKvSharedLayer(i) {
                if let c = layerCache[i] { return c.offset }
            }
            return 0
        }()

        // 4. Iterate decoder layers, collecting KV donors.
        var kvDonors: [Int: (keys: MLXArray, values: MLXArray)] = [:]

        for (i, layer) in layers.enumerated() {
            let usesSliding = config.isSlidingLayer(i)
            let mask = usesSliding ? slidingMask : fullMask
            let pli = perLayerInputs[0..., 0..., i, 0...]

            var donor: (keys: MLXArray, values: MLXArray)? = nil
            if let donorIdx = config.kvDonorLayerIndex(forSharedLayer: i) {
                assert(kvDonors[donorIdx] != nil,
                       "KV donor for layer \(i) → donor \(donorIdx) missing; layer order bug")
                donor = kvDonors[donorIdx]
            }

            let result = layer(
                hidden,
                perLayerInput: pli,
                mask: mask,
                cache: layerCache[i],
                sharedKVDonor: donor,
                ropeOffset: globalRopeOffset)
            hidden = result.output

            // Stash donor for any downstream KV-shared layers of the same
            // attention type. Only non-shared layers are eligible donors.
            if !config.isKvSharedLayer(i) {
                kvDonors[i] = (keys: result.keys, values: result.values)
            }
        }

        // 5. Final norm.
        return norm(hidden)
    }
}
