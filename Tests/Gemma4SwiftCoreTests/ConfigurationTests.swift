// ConfigurationTests.swift
//
// Pure JSON-decoding tests for Gemma4TextConfiguration. No tensor ops,
// no MLX initialization — runnable on any platform that has Swift,
// including Linux CI runners and the Xcode iOS simulator.
//
// SPDX-License-Identifier: MIT

import XCTest
@testable import Gemma4SwiftCore

@available(iOS 17.0, macOS 14.0, *)
final class ConfigurationTests: XCTestCase {

    /// Loads the synthetic 5-layer fixture shipped under
    /// `Tests/Gemma4SwiftCoreTests/Fixtures/tiny_config.json`.
    private func loadTinyConfig() throws -> Gemma4TextConfiguration {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny_config", withExtension: "json",
                              subdirectory: "Fixtures"),
            "tiny_config.json must be present in the Fixtures resource bundle")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
    }

    // MARK: - Field decoding

    func test_decode_basicFields() throws {
        let config = try loadTinyConfig()
        XCTAssertEqual(config.modelType, "gemma4_text")
        XCTAssertEqual(config.hiddenSize, 32)
        XCTAssertEqual(config.hiddenLayers, 5)
        XCTAssertEqual(config.intermediateSize, 64)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 1)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.globalHeadDim, 16)
        XCTAssertEqual(config.numKvSharedLayers, 2)
        XCTAssertEqual(config.hiddenSizePerLayerInput, 8)
        XCTAssertEqual(config.vocabSizePerLayerInput, 100)
        XCTAssertTrue(config.useDoubleWideMlp)
        XCTAssertFalse(config.attentionKEqV)
        XCTAssertFalse(config.enableMoeBlock)
        XCTAssertEqual(config.slidingWindow, 16)
        XCTAssertEqual(config.maxPositionEmbeddings, 128)
        XCTAssertEqual(config.vocabularySize, 100)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.finalLogitSoftcapping, 30.0)
    }

    func test_decode_layerTypes_haveExpectedPattern() throws {
        let config = try loadTinyConfig()
        XCTAssertEqual(config.layerTypes.count, 5)
        XCTAssertEqual(config.layerTypes[0], "sliding_attention")
        XCTAssertEqual(config.layerTypes[3], "sliding_attention")
        XCTAssertEqual(config.layerTypes[4], "full_attention")
    }

    func test_decode_ropeParameters_perAttentionType() throws {
        let config = try loadTinyConfig()
        let sliding = try XCTUnwrap(config.ropeParameters["sliding_attention"])
        XCTAssertEqual(sliding.ropeType, "default")
        XCTAssertEqual(sliding.ropeTheta, 10_000)
        XCTAssertNil(sliding.partialRotaryFactor)

        let full = try XCTUnwrap(config.ropeParameters["full_attention"])
        XCTAssertEqual(full.ropeType, "proportional")
        XCTAssertEqual(full.ropeTheta, 1_000_000)
        XCTAssertEqual(full.partialRotaryFactor, 0.25)
    }

    // MARK: - Derived properties

    func test_firstKvSharedLayerIdx_equalsLayersMinusShared() throws {
        let config = try loadTinyConfig()
        XCTAssertEqual(config.firstKvSharedLayerIdx, 3)  // 5 - 2
    }

    func test_isKvSharedLayer_handlesBoundary() throws {
        let config = try loadTinyConfig()
        XCTAssertFalse(config.isKvSharedLayer(0))
        XCTAssertFalse(config.isKvSharedLayer(2))
        XCTAssertTrue(config.isKvSharedLayer(3))
        XCTAssertTrue(config.isKvSharedLayer(4))
    }

    func test_kvDonorLayerIndex_picksMostRecentSameType() throws {
        let config = try loadTinyConfig()
        // Layer 3 is sliding-shared. Donor must be the most recent
        // non-shared sliding layer, which is layer 2.
        XCTAssertEqual(config.kvDonorLayerIndex(forSharedLayer: 3), 2)
        // Layer 4 is full-attention but has no full-attention layer in
        // the non-shared range — donor lookup should return nil.
        XCTAssertNil(config.kvDonorLayerIndex(forSharedLayer: 4))
        // Non-shared layers always return nil.
        XCTAssertNil(config.kvDonorLayerIndex(forSharedLayer: 0))
    }

    func test_headDimForLayer_perAttentionType() throws {
        let config = try loadTinyConfig()
        XCTAssertEqual(config.headDimForLayer(0), 8)   // sliding
        XCTAssertEqual(config.headDimForLayer(4), 16)  // full
    }

    func test_intermediateSizeForLayer_doubleWideForKvShared() throws {
        let config = try loadTinyConfig()
        XCTAssertEqual(config.intermediateSizeForLayer(0), 64)   // not shared
        XCTAssertEqual(config.intermediateSizeForLayer(2), 64)   // not shared
        XCTAssertEqual(config.intermediateSizeForLayer(3), 128)  // shared, doubled
    }

    func test_supportsCurrentImplementation_isTrueWhenMoEDisabled() throws {
        let config = try loadTinyConfig()
        XCTAssertTrue(config.supportsCurrentImplementation)
    }
}
