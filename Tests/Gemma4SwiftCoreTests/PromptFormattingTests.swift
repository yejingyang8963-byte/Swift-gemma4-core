// PromptFormattingTests.swift
//
// Pure-string tests for Gemma4PromptFormatter — no tokenizer required.
// The opt-in network test in NetworkIntegrationTests.swift verifies that
// these strings, when fed through swift-transformers' tokenizer.encode,
// produce the same token IDs as Python apply_chat_template.
//
// SPDX-License-Identifier: MIT

import XCTest
@testable import Gemma4SwiftCore

@available(iOS 17.0, macOS 14.0, *)
final class PromptFormattingTests: XCTestCase {

    // MARK: - userTurn

    func test_userTurn_wrapsWithCorrectMarkers() {
        let result = Gemma4PromptFormatter.userTurn("Hello, what is your name?")
        XCTAssertEqual(
            result,
            "<bos><|turn>user\nHello, what is your name?<turn|>\n<|turn>model\n")
    }

    func test_userTurn_emptyMessage_stillWellFormed() {
        let result = Gemma4PromptFormatter.userTurn("")
        XCTAssertEqual(result, "<bos><|turn>user\n<turn|>\n<|turn>model\n")
    }

    func test_userTurn_multilineMessage_preservesNewlines() {
        let multiline = "Line one.\nLine two.\nLine three."
        let result = Gemma4PromptFormatter.userTurn(multiline)
        XCTAssertTrue(result.contains("Line one.\nLine two.\nLine three."))
        XCTAssertTrue(result.hasPrefix("<bos><|turn>user\n"))
        XCTAssertTrue(result.hasSuffix("<turn|>\n<|turn>model\n"))
    }

    // MARK: - userTurnWithThinking

    func test_userTurnWithThinking_includesSystemTurnAndThinkToken() {
        let result = Gemma4PromptFormatter.userTurnWithThinking(
            "Tell me a short fairy tale.")
        // System turn comes first with the <|think|> injection.
        XCTAssertTrue(result.contains("<|turn>system\n<|think|><turn|>\n"))
        // Then the user turn.
        XCTAssertTrue(result.contains("<|turn>user\nTell me a short fairy tale.<turn|>\n"))
        // Trailing model turn invitation.
        XCTAssertTrue(result.hasSuffix("<|turn>model\n"))
    }

    func test_userTurnWithThinking_disabled_omitsThinkToken() {
        let result = Gemma4PromptFormatter.userTurnWithThinking(
            "Hello",
            includeThinking: false)
        XCTAssertFalse(result.contains("<|think|>"))
        XCTAssertTrue(result.contains("<|turn>system\n<turn|>\n"))
    }

    // MARK: - conversation

    func test_conversation_singleUserTurn_matchesUserTurnHelper() {
        let conv = Gemma4PromptFormatter.conversation(
            turns: [(role: "user", content: "Hi")])
        XCTAssertEqual(conv, "<bos><|turn>user\nHi<turn|>\n<|turn>model\n")
    }

    func test_conversation_multipleTurns_areSeparatedAndOrdered() {
        let conv = Gemma4PromptFormatter.conversation(turns: [
            (role: "user", content: "Hello"),
            (role: "model", content: "Hi there!"),
            (role: "user", content: "How are you?"),
        ])
        XCTAssertTrue(conv.contains("<|turn>user\nHello<turn|>\n"))
        XCTAssertTrue(conv.contains("<|turn>model\nHi there!<turn|>\n"))
        XCTAssertTrue(conv.contains("<|turn>user\nHow are you?<turn|>\n"))
        XCTAssertTrue(conv.hasSuffix("<|turn>model\n"))
    }

    func test_conversation_bosFirstFalse_omitsLeadingBos() {
        let conv = Gemma4PromptFormatter.conversation(
            turns: [(role: "user", content: "Hi")],
            bosFirst: false)
        XCTAssertFalse(conv.hasPrefix("<bos>"))
        XCTAssertTrue(conv.hasPrefix("<|turn>user\n"))
    }
}
