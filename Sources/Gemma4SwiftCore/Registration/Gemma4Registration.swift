// Gemma4Registration.swift
//
// Registers `Gemma4TextModel` with `mlx-swift-lm`'s `LLMModelFactory`
// type registry, so that loading any HuggingFace repo whose
// `config.json` has `model_type: "gemma4"` (or `"gemma4_text"`) returns
// our native implementation.
//
// As of mlx-swift-lm 2.31.x there is no upstream Gemma 4 support — see
// the issue tracker on github.com/ml-explore/mlx-swift-examples. Rather
// than maintaining a fork, we register our model handler as a sidecar
// at app startup.
//
// SPDX-License-Identifier: MIT

import Foundation
import MLXLLM
import MLXLMCommon
import os

/// Static helper that registers ``Gemma4TextModel`` with
/// `LLMModelFactory.shared.typeRegistry` exactly once per process.
///
/// ## Usage
/// ```swift
/// // At app startup, before loading any model:
/// await Gemma4Registration.registerIfNeeded().value
///
/// // Then load Gemma 4 like any other LLM:
/// let container = try await LLMModelFactory.shared.loadContainer(
///     configuration: ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit"))
/// ```
///
/// `registerIfNeeded` is idempotent — call it as often as you want, the
/// underlying registration only runs once.
@available(iOS 17.0, macOS 14.0, *)
public enum Gemma4Registration {

    private static let logger = Logger(
        subsystem: "com.gemma4swiftcore", category: "registration")

    /// The in-flight or completed registration task. Stored on the type so
    /// repeated calls to ``registerIfNeeded()`` await the same work.
    public private(set) static var registrationTask: Task<Void, Never>?

    /// Register `gemma4` and `gemma4_text` model handlers. Safe to call
    /// from any actor context. The returned task can be awaited to ensure
    /// registration is complete before constructing a model.
    @discardableResult
    public static func registerIfNeeded() -> Task<Void, Never> {
        if let existing = registrationTask { return existing }

        let task = Task {
            let registry = LLMModelFactory.shared.typeRegistry

            if await isNativelySupported(registry: registry) {
                logger.info("Gemma 4 natively supported by upstream; skipping sidecar registration.")
                return
            }

            await registry.registerModelType("gemma4") { data in
                let config = try JSONDecoder().decode(
                    Gemma4TextConfiguration.self, from: data)
                return Gemma4TextModel(config)
            }
            await registry.registerModelType("gemma4_text") { data in
                let config = try JSONDecoder().decode(
                    Gemma4TextConfiguration.self, from: data)
                return Gemma4TextModel(config)
            }

            logger.info("Gemma4SwiftCore registered native Gemma 4 model handlers.")
        }
        registrationTask = task
        return task
    }

    /// Probe whether upstream `mlx-swift-lm` already handles "gemma4".
    ///
    /// Passes empty JSON; if the registry returns `unsupportedModelType`
    /// we know the type is unregistered. Any other error (e.g. missing
    /// required fields) means the type IS registered upstream and we
    /// should not double-register.
    private static func isNativelySupported(registry: ModelTypeRegistry) async -> Bool {
        let testData = Data("{}".utf8)
        do {
            _ = try await registry.createModel(configuration: testData, modelType: "gemma4")
            return true
        } catch {
            let message = String(describing: error)
            return !message.contains("unsupportedModelType")
        }
    }
}
