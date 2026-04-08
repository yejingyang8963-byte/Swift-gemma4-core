// Gemma4TextInner+PerLayerInputs.swift
//
// Per-Layer Embedding (PLE) computation for the Gemma 4 inner model.
//
// PLE is the most distinctive feature of Gemma 4 — every decoder layer
// receives a per-token vector that is computed once at the top of the
// model from two parallel paths and combined with a fixed scale factor.
//
// Path A — direct embedding lookup:
//
//     pathA = embed_tokens_per_layer(input_ids)
//             * sqrt(hidden_size_per_layer_input)
//     pathA = pathA.reshape(B, L, num_layers, hidden_size_per_layer_input)
//
// Path B — projection from token embeddings:
//
//     pathB = per_layer_model_projection(inputs_embeds)
//             * (hidden_size ** -0.5)
//     pathB = pathB.reshape(B, L, num_layers, hidden_size_per_layer_input)
//     pathB = per_layer_projection_norm(pathB)
//
// Combined:
//
//     out = (pathB + pathA) * (2 ** -0.5)
//
// The result has shape `[B, L, num_layers, hidden_size_per_layer_input]`
// and is sliced per-layer at the call site in the main forward pass.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

@available(iOS 17.0, macOS 14.0, *)
extension Gemma4TextInner {

    /// Compute per-layer inputs once at the top of the model.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` int32 token IDs.
    ///   - inputsEmbeds: `[batch, seq, hidden_size]` token embeddings,
    ///     pre-scale (the caller should pass the post-scale tensor).
    /// - Returns: `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.
    func computePerLayerInputs(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray
    ) -> MLXArray {
        let B = inputIds.dim(0)
        let L = inputIds.dim(1)
        let numLayers = config.hiddenLayers
        let perLayerDim = config.hiddenSizePerLayerInput

        // Path A: direct embedding lookup, scaled and reshaped.
        var pathA = embedTokensPerLayer(inputIds)
        pathA = pathA * MLXArray(perLayerEmbedScale).asType(pathA.dtype)
        pathA = pathA.reshaped(B, L, numLayers, perLayerDim)

        // Path B: project from token embeddings, scale, reshape, norm.
        var pathB = perLayerModelProjection(inputsEmbeds)
        pathB = pathB * MLXArray(perLayerModelProjectionScale).asType(pathB.dtype)
        pathB = pathB.reshaped(B, L, numLayers, perLayerDim)
        pathB = perLayerProjectionNorm(pathB)

        // Combine the two paths with the PLE input scale (2^-0.5).
        let combined = (pathB + pathA) * MLXArray(perLayerInputScale).asType(pathA.dtype)
        return combined
    }
}
