// SanitizeTests.swift
//
// Verifies that Gemma4TextModel.sanitize correctly transforms a raw
// HuggingFace weight dict into the layout this module expects:
//
//   - Strip the `language_model.` prefix on multimodal checkpoints
//   - Drop vision/audio tower / projector tensors
//   - Tie lm_head to embed_tokens when tie_word_embeddings = true
//
// We use placeholder MLXArrays for the values — sanitize is purely a key
// rewrite, so the tensor contents don't matter for these tests.
//
// SPDX-License-Identifier: MIT

import XCTest
import MLX
@testable import Gemma4SwiftCore

@available(iOS 17.0, macOS 14.0, *)
final class SanitizeTests: XCTestCase {

    private func loadTinyConfig() throws -> Gemma4TextConfiguration {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny_config", withExtension: "json",
                              subdirectory: "Fixtures"))
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: try Data(contentsOf: url))
    }

    private func makeModel() throws -> Gemma4TextModel {
        Gemma4TextModel(try loadTinyConfig())
    }

    private func dummy(_ shape: [Int]) -> MLXArray {
        MLXArray.zeros(shape)
    }

    // MARK: - Prefix stripping

    func test_sanitize_stripsLanguageModelPrefix() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": dummy([100, 32]),
            "language_model.model.layers.0.input_layernorm.weight": dummy([32]),
        ]
        let result = model.sanitize(weights: weights)
        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertNotNil(result["model.layers.0.input_layernorm.weight"])
        XCTAssertNil(result["language_model.model.embed_tokens.weight"])
    }

    // MARK: - Multimodal tensor filtering

    func test_sanitize_dropsVisionAndAudioTowerTensors() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": dummy([100, 32]),
            "vision_tower.encoder.layer.0.weight": dummy([16, 16]),
            "audio_tower.encoder.layer.0.weight": dummy([16, 16]),
            "embed_vision.weight": dummy([16, 16]),
            "embed_audio.weight": dummy([16, 16]),
            "multi_modal_projector.proj.weight": dummy([16, 16]),
        ]
        let result = model.sanitize(weights: weights)
        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertNil(result["vision_tower.encoder.layer.0.weight"])
        XCTAssertNil(result["audio_tower.encoder.layer.0.weight"])
        XCTAssertNil(result["embed_vision.weight"])
        XCTAssertNil(result["embed_audio.weight"])
        XCTAssertNil(result["multi_modal_projector.proj.weight"])
    }

    // MARK: - lm_head tying

    func test_sanitize_tieWordEmbeddings_copiesEmbedToLmHead() throws {
        let model = try makeModel()
        let embed = dummy([100, 32])
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": embed,
        ]
        let result = model.sanitize(weights: weights)
        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertNotNil(result["lm_head.weight"])
    }

    func test_sanitize_tieWordEmbeddings_copiesQuantizationScalesAndBiases() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": dummy([100, 32]),
            "language_model.model.embed_tokens.scales": dummy([100, 4]),
            "language_model.model.embed_tokens.biases": dummy([100, 4]),
        ]
        let result = model.sanitize(weights: weights)
        XCTAssertNotNil(result["lm_head.weight"])
        XCTAssertNotNil(result["lm_head.scales"])
        XCTAssertNotNil(result["lm_head.biases"])
    }
}
