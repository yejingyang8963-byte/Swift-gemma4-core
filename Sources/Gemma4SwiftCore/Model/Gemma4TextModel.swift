// Gemma4TextModel.swift
//
// Top-level Gemma 4 text model that conforms to the `LLMModel` protocol
// from `mlx-swift-lm`. Wraps the inner decoder stack with:
//
//   - lm_head (Linear) tied to embed_tokens via `sanitize`
//   - final logit soft-capping (tanh-bounded, default cap = 30)
//   - weight sanitization (strip `language_model.` prefix, drop vision/audio)
//   - per-layer KV cache construction
//   - LoRA conformance pointing at decoder layers
//
// This is the class registered with `LLMModelFactory` in `Gemma4Registration`.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Top-level Gemma 4 text model. Drop-in replacement for any
/// `mlx-swift-lm` LLM, but with full Gemma 4 architectural support
/// (PLE, KV sharing, proportional RoPE, per-layer head_dim).
///
/// ## Loading
/// ```swift
/// try await Gemma4Registration.registerIfNeeded().value
/// let container = try await LLMModelFactory.shared.loadContainer(
///     configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))
/// ```
///
/// See ``Documentation.docc/Architecture.md`` for the architectural
/// overview and ``Documentation.docc/ChatTemplateBypass.md`` for why you
/// must NOT use the default chat template path.
@available(iOS 17.0, macOS 14.0, *)
public final class Gemma4TextModel: Module, LLMModel, LoRAModel {

    @ModuleInfo public var model: Gemma4TextInner
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    /// Construct from a parsed configuration.
    ///
    /// - Precondition: `config.supportsCurrentImplementation` must be true
    ///   (i.e. `enable_moe_block` must be `false`). MoE is not implemented.
    public init(_ config: Gemma4TextConfiguration) {
        precondition(config.supportsCurrentImplementation,
                     "Gemma 4 MoE variants are not supported by Gemma4SwiftCore")
        self.config = config
        self.model = Gemma4TextInner(config: config)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    /// Forward pass producing per-token logits.
    ///
    /// - Parameters:
    ///   - inputs: `[batch, seq]` int32 token IDs.
    ///   - cache: Optional per-layer KV caches.
    /// - Returns: `[batch, seq, vocab_size]` logits, post soft-capping.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let optionalCache: [KVCache?]? = cache.map { $0.map { Optional.some($0) } }
        var h = model(inputs, cache: optionalCache)
        h = lmHead(h)

        // Gemma 4 uses final logit soft-capping: logits = cap * tanh(logits / cap).
        // Default cap is 30 (set in HF config). Skipping this collapses the
        // sampler distribution.
        if let cap = config.finalLogitSoftcapping {
            let capArr = MLXArray(cap).asType(h.dtype)
            h = capArr * MLX.tanh(h / capArr)
        }

        return h
    }

    /// Rewrite checkpoint weight keys so they line up with this module hierarchy.
    ///
    /// - Strips a leading `language_model.` prefix from multimodal checkpoints.
    /// - Drops vision tower / audio tower / cross-modal projector tensors —
    ///   we only consume the text tower.
    /// - Ties `lm_head` to `embed_tokens` when ``Gemma4TextConfiguration/tieWordEmbeddings``
    ///   is true (the real 4-bit checkpoint has no `lm_head.*` keys).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("language_model.") {
                out[String(key.dropFirst("language_model.".count))] = value
            } else {
                out[key] = value
            }
        }

        let unwantedPrefixes = [
            "vision_tower.", "audio_tower.",
            "embed_vision.", "embed_audio.",
            "multi_modal_projector.",
        ]
        out = out.filter { key, _ in
            !unwantedPrefixes.contains { key.hasPrefix($0) }
        }

        if config.tieWordEmbeddings, out["lm_head.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                if let tensor = out["model.embed_tokens.\(suffix)"] {
                    out["lm_head.\(suffix)"] = tensor
                }
            }
        }

        return out
    }

    /// Build per-layer KV caches.
    ///
    /// Sliding layers get a `RotatingKVCache` sized to the configured
    /// sliding window. Full-attention layers get a `StandardKVCache`
    /// pre-allocated to a 1024-token chunk for fewer reallocations.
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches: [KVCache] = []
        for i in 0 ..< config.hiddenLayers {
            if config.isSlidingLayer(i) {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            } else {
                let c = StandardKVCache()
                c.step = 1024
                caches.append(c)
            }
        }
        return caches
    }

    /// `LLMModel` requirement — pass through input tokens unchanged.
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        return .tokens(input.text)
    }

    // MARK: - LoRAModel

    /// `LoRAModel` requirement — expose decoder layers for LoRA injection.
    public var loraLayers: [Module] { model.layers }
}
