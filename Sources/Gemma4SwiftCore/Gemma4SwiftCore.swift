// Gemma4SwiftCore.swift
//
// Umbrella file for the Gemma4SwiftCore module. Re-exports the public API
// surface and provides a single import target for downstream users.
//
// The module is split into focused subsystems by directory:
//
//   Configuration/  parses HuggingFace config.json into a Codable struct
//   Layers/         MLP, Attention, RoPE, DecoderLayer building blocks
//   Model/          inner decoder stack + top-level LLMModel conformance
//   Registration/   hookup with mlx-swift-lm's LLMModelFactory type registry
//   Prompt/         chat-template bypass for Gemma 4's <|turn> protocol
//
// SPDX-License-Identifier: MIT

import Foundation

/// Marker enum holding the package's identity. Lets downstream users do
/// `Gemma4SwiftCore.version` for diagnostic logging without importing the
/// individual submodules.
public enum Gemma4SwiftCore {
    /// Semantic version of this Gemma4SwiftCore release.
    /// Bumped manually on each tagged release.
    public static let version: String = "0.1.0"

    /// Hugging Face model id this implementation has been verified against.
    /// Other Gemma 4 variants (e.g. E4B, larger quant levels) MAY work but
    /// are not part of the test matrix yet.
    public static let verifiedModelId: String = "mlx-community/gemma-4-e2b-it-4bit"
}
