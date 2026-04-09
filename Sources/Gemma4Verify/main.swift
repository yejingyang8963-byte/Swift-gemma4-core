// main.swift — Gemma4Verify
//
// End-to-end smoke test for Gemma4SwiftCore. Registers the sidecar handler,
// loads the verified 4-bit checkpoint from HuggingFace (~1.5 GB, cached on
// disk after the first run), encodes a prompt with the chat-template
// bypass, validates the encoded token sequence, then streams a short
// generation and prints throughput stats.
//
// Run from the package root:
//
//     swift run Gemma4Verify                      # default prompt
//     swift run Gemma4Verify "tell me a haiku"    # custom prompt
//     GEMMA4_VERIFY_MAX_TOKENS=64 swift run Gemma4Verify
//
// Exit codes:
//   0  success
//   1  validation failed (empty input, oversize prompt, missing tokens)
//   2  load / generate failure (network, model error)
//
// SPDX-License-Identifier: MIT

import Foundation
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

@main
struct Gemma4Verify {

    static func main() async {
        let prompt = parsePrompt()
        let maxTokens = parseMaxTokens()

        log("─── Gemma4SwiftCore verify ───")
        log("Package version : \(Gemma4SwiftCore.version)")
        log("Model id        : \(Gemma4SwiftCore.verifiedModelId)")
        log("Prompt          : \(prompt.prefix(80))\(prompt.count > 80 ? "…" : "")")
        log("Max new tokens  : \(maxTokens)")

        // ─── 1. Input validation (cheap, no model needed) ────────────
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            fail("Prompt is empty after trimming whitespace.", code: 1)
        }
        // Roughly cap raw character length so we never silently exceed
        // the model's 32k position embedding budget. ~3 chars/token is a
        // conservative lower bound for English text; we re-check after
        // tokenization with the actual encoded length.
        let charBudget = 32_000 * 3
        guard prompt.count < charBudget else {
            fail("Prompt is \(prompt.count) chars, above char budget \(charBudget).", code: 1)
        }
        let formatted = Gemma4PromptFormatter.userTurn(prompt)
        guard formatted.contains("<|turn>user") && formatted.contains("<|turn>model") else {
            fail("Prompt formatter produced output without expected turn markers.", code: 1)
        }
        log("Input validation: ok")

        // ─── 2. Register the sidecar (idempotent) ────────────────────
        log("Registering Gemma4 sidecar handler…")
        await Gemma4Registration.registerIfNeeded().value

        // ─── 3. Load the model (downloads on first run) ──────────────
        log("Loading container (first run downloads ~1.5 GB)…")
        let loadStart = Date()
        let container: ModelContainer
        do {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))
        } catch {
            fail("Container load failed: \(error)", code: 2)
        }
        log(String(format: "Loaded in %.2fs", Date().timeIntervalSince(loadStart)))

        // ─── 4. Encode and validate token sequence ───────────────────
        let tokens = await container.encode(formatted)
        guard !tokens.isEmpty else {
            fail("Tokenizer returned 0 tokens — special-token registration broken?", code: 1)
        }
        guard tokens.count < 32_768 else {
            fail("Encoded length \(tokens.count) exceeds max_position_embeddings.", code: 1)
        }
        // Sanity-check: BOS (id 2) should be the very first token, otherwise
        // the chat-template bypass and tokenizer disagree about <bos> handling.
        if tokens.first != 2 {
            log("⚠️  warning: expected BOS=2 at index 0, got \(tokens.first ?? -1)")
        }
        log("Encoded \(tokens.count) tokens. First 10: \(Array(tokens.prefix(10)))")

        // ─── 5. Stream generation ────────────────────────────────────
        let input = LMInput(tokens: MLXArray(tokens.map { Int32($0) }))
        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: 0.8,
            topP: 0.95)

        log("Generating…")
        log("──────")
        let genStart = Date()
        var firstChunkAt: TimeInterval? = nil
        var totalTokens = 0
        var sawChunk = false
        do {
            let stream = try await container.generate(
                input: input, parameters: parameters)
            for await event in stream {
                switch event {
                case .chunk(let text):
                    if firstChunkAt == nil {
                        firstChunkAt = Date().timeIntervalSince(genStart)
                    }
                    sawChunk = true
                    print(text, terminator: "")
                    fflush(stdout)
                case .info(let info):
                    totalTokens = info.generationTokenCount
                case .toolCall:
                    break
                @unknown default:
                    break
                }
            }
        } catch {
            print()
            fail("Generation failed: \(error)", code: 2)
        }
        print()
        log("──────")
        let elapsed = Date().timeIntervalSince(genStart)
        log(String(format: "TTFC: %.2fs · gen: %.2fs · tokens: %d · tok/s: %.2f",
                   firstChunkAt ?? -1,
                   elapsed,
                   totalTokens,
                   totalTokens > 0 ? Double(totalTokens) / elapsed : 0))

        guard sawChunk else {
            fail("Generation stream produced no chunks.", code: 2)
        }
        log("✅ verify ok")
    }

    // MARK: - helpers

    static func parsePrompt() -> String {
        let args = CommandLine.arguments.dropFirst()
        if args.isEmpty {
            return "Tell me a short story about a curious fox."
        }
        return args.joined(separator: " ")
    }

    static func parseMaxTokens() -> Int {
        if let v = ProcessInfo.processInfo.environment["GEMMA4_VERIFY_MAX_TOKENS"],
           let n = Int(v), n > 0 {
            return n
        }
        return 80
    }

    static func log(_ message: String) {
        FileHandle.standardError.write(Data("[verify] \(message)\n".utf8))
    }

    static func fail(_ message: String, code: Int32) -> Never {
        FileHandle.standardError.write(Data("[verify] ❌ \(message)\n".utf8))
        exit(code)
    }
}
