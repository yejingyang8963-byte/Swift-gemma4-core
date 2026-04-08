// gemma4-generate — minimal command-line demo of Gemma4SwiftCore.
//
// Usage:
//     swift run gemma4-generate "Tell me a short story about a fox."
//
// On first run this downloads the Gemma 4 E2B 4-bit weights from
// HuggingFace (~1.5 GB) into the platform's caches directory. The
// download progress is reported by mlx-swift-lm's HubApi client.
// Subsequent runs are warm and reach the first generated chunk in
// 2-6 seconds depending on hardware.

import Foundation
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

@main
struct Gemma4Generate {

    static func main() async {
        // Parse the prompt from argv. If no argument is provided we
        // use a neutral default so the demo always has something
        // sensible to render.
        let prompt: String
        if CommandLine.arguments.count > 1 {
            prompt = CommandLine.arguments[1...].joined(separator: " ")
        } else {
            prompt = "Tell me a short story about a curious fox."
        }

        do {
            try await run(prompt: prompt)
        } catch {
            FileHandle.standardError.write(
                "❌ generation failed: \(error)\n".data(using: .utf8) ?? Data())
            exit(1)
        }
    }

    static func run(prompt: String) async throws {
        print("[gemma4-generate] Gemma4SwiftCore v\(Gemma4SwiftCore.version)")
        print("[gemma4-generate] Model: \(Gemma4SwiftCore.verifiedModelId)")
        print("[gemma4-generate] Prompt: \(prompt)")
        print("")

        // 1. Register the sidecar handler with mlx-swift-lm. Idempotent.
        await Gemma4Registration.registerIfNeeded().value

        // 2. Load the real 4-bit weights from HuggingFace.
        print("[gemma4-generate] Loading model (first run downloads ~1.5 GB)...")
        let loadStart = Date()
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))
        let loadElapsed = Date().timeIntervalSince(loadStart)
        print(String(format: "[gemma4-generate] Loaded in %.1fs.\n", loadElapsed))

        // 3. Format the prompt with the chat-template bypass.
        let formatted = Gemma4PromptFormatter.userTurn(prompt)
        let tokens = await container.encode(formatted)
        let input = LMInput(tokens: MLXArray(tokens))

        // 4. Stream tokens.
        let parameters = GenerateParameters(
            maxTokens: 200,
            temperature: 0.8,
            topP: 0.95)
        let stream = try await container.generate(input: input, parameters: parameters)

        let generationStart = Date()
        var firstChunkLatency: TimeInterval? = nil
        for await event in stream {
            switch event {
            case .chunk(let text):
                if firstChunkLatency == nil {
                    firstChunkLatency = Date().timeIntervalSince(generationStart)
                }
                print(text, terminator: "")
                FileHandle.standardOutput.synchronizeFile()
            case .info(let info):
                print("\n")
                print("[gemma4-generate] \(info.promptTokenCount) prompt tokens, "
                    + String(format: "%.1f tok/s", info.tokensPerSecond))
            case .toolCall:
                break
            }
        }
        print("")

        if let latency = firstChunkLatency {
            print(String(format: "[gemma4-generate] First chunk: %.2fs", latency))
        }
    }
}
