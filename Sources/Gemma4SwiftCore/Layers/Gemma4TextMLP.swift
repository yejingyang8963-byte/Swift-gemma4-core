// Gemma4TextMLP.swift
//
// SwiGLU MLP block for Gemma 4 text decoder layers.
//
// Same gate/up/down layout as Gemma 3, but layers in the KV-shared range
// pass a 2x ``Gemma4TextConfiguration/intermediateSize`` when constructing
// the module. The MLP itself doesn't know about the doubling — that
// decision is made at the call site in `Gemma4TextDecoderLayer`.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

/// SwiGLU feed-forward block: `down(gelu_approx(gate(x)) * up(x))`.
///
/// All three projections are bias-free `Linear` layers, matching the
/// HuggingFace tensor layout (`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`).
///
/// ## Topology
/// ```
/// x  ─── gate ──── gelu_approx ──┐
/// │                              │
/// └─── up ───────────────────────×─── down ──── output
/// ```
@available(iOS 17.0, macOS 14.0, *)
public final class Gemma4TextMLP: Module {

    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    /// Construct a SwiGLU MLP.
    ///
    /// - Parameters:
    ///   - hiddenSize: Input/output channel count (e.g. 1536 for E2B).
    ///   - intermediateSize: Width of the gate/up projections. Pass
    ///     2x for KV-shared layers when ``Gemma4TextConfiguration/useDoubleWideMlp``
    ///     is enabled (e.g. 12288 instead of 6144).
    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameter x: `[batch, seq, hidden_size]` input activations.
    /// - Returns: `[batch, seq, hidden_size]` output activations.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = geluApproximate(gateProj(x))
        let u = upProj(x)
        return downProj(g * u)
    }
}
