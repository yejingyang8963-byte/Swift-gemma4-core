// ModuleShapeTests.swift
//
// Shape-only forward-pass tests for the building-block modules. They
// instantiate each module against the synthetic 5-layer config and run
// a forward pass with zero tensors, asserting only the output shape.
//
// **Platform note**: these tests construct real MLX modules. They run
// natively on Apple Silicon (macOS arm64 / iOS arm64). On x86_64
// simulators MLX is unreliable upstream, so the tests `XCTSkip` there.
//
// SPDX-License-Identifier: MIT

import XCTest
import MLX
import MLXNN
@testable import Gemma4SwiftCore

@available(iOS 17.0, macOS 14.0, *)
final class ModuleShapeTests: XCTestCase {

    /// Skip on x86_64 simulator hosts where MLX is non-functional.
    private func skipOnX86Simulator() throws {
        #if targetEnvironment(simulator) && !arch(arm64)
        throw XCTSkip("MLX is unreliable on x86_64 iOS simulator — run on Apple Silicon")
        #endif
    }

    private func loadTinyConfig() throws -> Gemma4TextConfiguration {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tiny_config", withExtension: "json",
                              subdirectory: "Fixtures"))
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: try Data(contentsOf: url))
    }

    // MARK: - MLP

    func test_mlp_layer0_outputShape_matchesHiddenSize() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        let mlp = Gemma4TextMLP(
            hiddenSize: cfg.hiddenSize,
            intermediateSize: cfg.intermediateSize)
        let x = MLXArray.zeros([1, 4, cfg.hiddenSize])
        XCTAssertEqual(mlp(x).shape, [1, 4, cfg.hiddenSize])
    }

    func test_mlp_doubleWide_keepsOutputShape() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        let mlp = Gemma4TextMLP(
            hiddenSize: cfg.hiddenSize,
            intermediateSize: cfg.intermediateSize * 2)
        let x = MLXArray.zeros([1, 4, cfg.hiddenSize])
        XCTAssertEqual(mlp(x).shape, [1, 4, cfg.hiddenSize])
    }

    // MARK: - Attention

    func test_attention_slidingLayer_returnsExpectedShapes() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        let attn = Gemma4TextAttention(config: cfg, layerIdx: 0)
        XCTAssertEqual(attn.headDim, 8)   // sliding head_dim
        XCTAssertEqual(attn.ropeDims, 8)  // full rotation
        let x = MLXArray.zeros([1, 4, cfg.hiddenSize])
        let result = attn(x, mask: .none, cache: nil, sharedKVDonor: nil, ropeOffset: 0)
        XCTAssertEqual(result.output.shape, [1, 4, cfg.hiddenSize])
        XCTAssertEqual(result.keys.shape, [1, cfg.kvHeads, 4, 8])
        XCTAssertEqual(result.values.shape, [1, cfg.kvHeads, 4, 8])
    }

    func test_attention_fullLayer_usesGlobalHeadDimAndPartialRotary() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        let attn = Gemma4TextAttention(config: cfg, layerIdx: 4)
        XCTAssertEqual(attn.headDim, 16)  // global_head_dim
        XCTAssertEqual(attn.ropeDims, 4)  // 16 * 0.25
        let x = MLXArray.zeros([1, 4, cfg.hiddenSize])
        let result = attn(x, mask: .none, cache: nil, sharedKVDonor: nil, ropeOffset: 0)
        XCTAssertEqual(result.output.shape, [1, 4, cfg.hiddenSize])
        XCTAssertEqual(result.keys.shape, [1, cfg.kvHeads, 4, 16])
    }

    func test_attention_withSharedKVDonor_skipsKVProjection() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        // Layer 3 is sliding-shared. Donor must have head_dim = 8.
        let attn = Gemma4TextAttention(config: cfg, layerIdx: 3)
        let x = MLXArray.zeros([1, 4, cfg.hiddenSize])
        let donorK = MLXArray.zeros([1, cfg.kvHeads, 4, 8])
        let donorV = MLXArray.zeros([1, cfg.kvHeads, 4, 8])
        let result = attn(
            x, mask: .none, cache: nil,
            sharedKVDonor: (keys: donorK, values: donorV),
            ropeOffset: 0)
        XCTAssertEqual(result.output.shape, [1, 4, cfg.hiddenSize])
        XCTAssertEqual(result.keys.shape, [1, cfg.kvHeads, 4, 8])
    }

    // MARK: - DecoderLayer

    func test_decoderLayer_withPLE_outputShape() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        let layer = Gemma4TextDecoderLayer(config: cfg, layerIdx: 0)
        let x = MLXArray.zeros([1, 4, cfg.hiddenSize])
        let pli = MLXArray.zeros([1, 4, cfg.hiddenSizePerLayerInput])
        let result = layer(
            x, perLayerInput: pli, mask: .none, cache: nil,
            sharedKVDonor: nil, ropeOffset: 0)
        XCTAssertEqual(result.output.shape, [1, 4, cfg.hiddenSize])
    }

    // MARK: - Inner model

    func test_inner_forward_outputShape() throws {
        try skipOnX86Simulator()
        let cfg = try loadTinyConfig()
        let inner = Gemma4TextInner(config: cfg)
        let ids = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let h = inner(ids, cache: nil)
        XCTAssertEqual(h.shape, [1, 3, cfg.hiddenSize])
    }

    func test_inner_forward_iteratesAllLayersAndKvDonorTableConsistent() throws {
        try skipOnX86Simulator()
        // Verifies the donor assertion in the inner model never fires —
        // i.e. every shared layer finds its donor in the kvDonors table.
        let cfg = try loadTinyConfig()
        let inner = Gemma4TextInner(config: cfg)
        let ids = MLXArray([Int32(0), 1, 2, 3]).reshaped([1, 4])
        _ = inner(ids, cache: nil)
    }
}
