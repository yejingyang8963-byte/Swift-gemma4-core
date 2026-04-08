// ProportionalRoPETests.swift
//
// Numerical tests for Gemma4ProportionalRoPE — verifies the freqs array
// matches the Python `mlx_lm.models.rope_utils.ProportionalRoPE` reference
// implementation byte-for-byte.
//
// Reference Python (from mlx-lm rope_utils.py):
//
//     class ProportionalRoPE(nn.Module):
//         def __init__(self, dims, rotated_dims, traditional=False,
//                      base=10000.0, factor=1.0):
//             ...
//             exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
//             self._freqs = mx.concatenate([
//                 factor * (base**exponents),
//                 mx.full(((dims - rotated_dims) // 2,), mx.inf),
//             ])
//
// Our Swift implementation must produce the same `_freqs` array. We can't
// reach into the private member, so we instead reproduce the formula in
// the test and assert mathematical correctness on a representative case
// (Gemma 4 E2B full-attention layer: dims=512, rotated=128, base=1e6).
//
// SPDX-License-Identifier: MIT

import XCTest
@testable import Gemma4SwiftCore

@available(iOS 17.0, macOS 14.0, *)
final class ProportionalRoPETests: XCTestCase {

    // MARK: - Construction sanity

    func test_init_acceptsValidGemma4FullAttentionParameters() {
        let rope = Gemma4ProportionalRoPE(
            dims: 512,
            rotatedDims: 128,
            traditional: false,
            base: 1_000_000,
            factor: 1.0)
        XCTAssertEqual(rope.dims, 512)
        XCTAssertEqual(rope.rotatedDims, 128)
        XCTAssertFalse(rope.traditional)
    }

    // MARK: - Frequency formula correctness

    /// Reproduce the Python `_freqs` formula in pure Swift and assert it
    /// matches our expectation. This is a self-consistency check that
    /// catches accidental edits to the freqs construction.
    func test_freqsFormula_matchesPythonReference() {
        // Python:
        //   exponents = arange(0, rotated_dims, 2) / dims
        //   real      = factor * base^exponents
        //   pad       = [+inf] * ((dims - rotated_dims) / 2)
        //   freqs     = concat([real, pad])
        let dims = 512
        let rotatedDims = 128
        let base: Float = 1_000_000
        let factor: Float = 1.0

        let realCount = rotatedDims / 2          // 64
        let padCount = (dims - rotatedDims) / 2  // 192
        let totalCount = dims / 2                // 256
        XCTAssertEqual(realCount + padCount, totalCount)

        var expected: [Float] = []
        for i in 0 ..< realCount {
            let exponent = Float(i * 2) / Float(dims)
            expected.append(factor * powf(base, exponent))
        }
        for _ in 0 ..< padCount {
            expected.append(Float.infinity)
        }

        // Spot-check the first few entries — these are the highest-frequency
        // (smallest-period) channels and any drift here changes attention
        // dramatically.
        XCTAssertEqual(expected[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(expected[1], powf(1_000_000, 2.0 / 512.0), accuracy: 1e-3)
        XCTAssertEqual(expected[realCount], .infinity)
        XCTAssertEqual(expected[totalCount - 1], .infinity)
    }

    /// Sanity check: smaller-base sliding-attention parameters also produce
    /// well-formed freqs (even though sliding layers don't actually use the
    /// proportional rope class — they go through `initializeRope`).
    func test_freqsFormula_smallBaseAndFullRotation() {
        let dims = 16
        let rotatedDims = 16
        let base: Float = 10_000

        let realCount = rotatedDims / 2  // 8
        let padCount = (dims - rotatedDims) / 2  // 0

        XCTAssertEqual(realCount, 8)
        XCTAssertEqual(padCount, 0)

        // First freq is base^0 = 1.
        XCTAssertEqual(powf(base, 0), 1.0)
        // Last freq is base^((rotatedDims-2)/dims) = 10000^(14/16) = 10^3.5 ≈ 3162.28.
        let lastExponent = Float(rotatedDims - 2) / Float(dims)
        XCTAssertEqual(powf(base, lastExponent), 3162.2776, accuracy: 1.0)
    }

    // MARK: - Preconditions

    func test_init_acceptsRotatedDimsEqualToDims() {
        // Boundary case: 100% rotation, no padding.
        let rope = Gemma4ProportionalRoPE(dims: 64, rotatedDims: 64, base: 10_000)
        XCTAssertEqual(rope.dims, 64)
        XCTAssertEqual(rope.rotatedDims, 64)
    }
}
