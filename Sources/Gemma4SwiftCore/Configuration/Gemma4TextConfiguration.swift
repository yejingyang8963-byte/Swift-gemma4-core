// Gemma4TextConfiguration.swift
//
// Native Gemma 4 text-tower configuration parser.
//
// Decodes the text portion of HuggingFace's `mlx-community/gemma-4-e2b-it-4bit`
// `config.json`. Supports both nested form (where text fields live under a
// `text_config` key) and flat top-level form.
//
// Architectural deltas vs. Gemma 3 — see Documentation.docc/Architecture.md
// for the full discussion. Briefly:
//
//   1. Per-Layer Embedding (PLE)  — `hidden_size_per_layer_input`,
//                                   `vocab_size_per_layer_input`
//   2. Layer scalar               — implicit, not in config
//   3. KV sharing                 — `num_kv_shared_layers`
//   4. Per-layer attention type   — `layer_types`
//   5. Larger head_dim for full   — `global_head_dim`
//
// MoE is not supported by this implementation. `enable_moe_block` must be
// false; the top-level model preconditions on this.
//
// SPDX-License-Identifier: MIT

import Foundation

/// Decoded HuggingFace text-tower configuration for a Gemma 4 model.
///
/// Use ``init(from:)`` via `JSONDecoder` to parse a HuggingFace
/// ``config.json`` file. The decoder accepts both the nested form (where
/// text fields live under a `text_config` key) and the flat top-level form.
///
/// ## Example
/// ```swift
/// let data = try Data(contentsOf: configURL)
/// let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
/// print(config.hiddenLayers)   // 35 for E2B
/// print(config.headDimForLayer(4))  // 512 (full_attention)
/// ```
public struct Gemma4TextConfiguration: Codable, Sendable {

    /// Per-attention-type RoPE parameters parsed from the `rope_parameters`
    /// dictionary in `config.json`.
    public struct RopeParameters: Codable, Sendable {
        /// Either `"default"` or `"proportional"`. Sliding layers use the
        /// former; full attention layers use the latter (with
        /// ``partialRotaryFactor`` `< 1.0`).
        public let ropeType: String

        /// Base for the RoPE frequency formula. 10_000 for sliding,
        /// 1_000_000 for full attention.
        public let ropeTheta: Float

        /// Fraction of head dimensions to rotate. `nil` (= 1.0) for sliding,
        /// `0.25` for full attention.
        public let partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let numKvSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let vocabSizePerLayerInput: Int
    public let useDoubleWideMlp: Bool
    public let attentionKEqV: Bool
    public let enableMoeBlock: Bool
    public let slidingWindow: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let layerTypes: [String]
    public let ropeParameters: [String: RopeParameters]
    public let finalLogitSoftcapping: Float?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case attentionKEqV = "attention_k_eq_v"
        case enableMoeBlock = "enable_moe_block"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    enum WrapperKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<CodingKeys>
        if let outer = try? decoder.container(keyedBy: WrapperKeys.self),
           outer.contains(.textConfig) {
            container = try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            container = try decoder.container(keyedBy: CodingKeys.self)
        }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? headDim
        numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        enableMoeBlock = try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        ropeParameters = try container.decode([String: RopeParameters].self, forKey: .ropeParameters)
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    /// MoE is not implemented in this Swift port. The top-level model
    /// preconditions on this property at construction time.
    public var supportsCurrentImplementation: Bool { !enableMoeBlock }
}
