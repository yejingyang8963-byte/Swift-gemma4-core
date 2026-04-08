// Gemma4PromptFormatter.swift
//
// Chat-template bypass for Gemma 4's `<|turn>...<turn|>` protocol.
//
// WHY THIS EXISTS
// ───────────────
// Gemma 4 ships an entirely new chat protocol with custom turn-marker
// special tokens (`<|turn>`, `<turn|>`, `<|think|>`, `<|channel>`,
// `<channel|>`) and a complex jinja2 chat_template that uses macros and
// namespaces. As of swift-jinja 1.x (the renderer used by
// swift-transformers and `tokenizer.applyChatTemplate`), this template is
// rendered incorrectly:
//
//   Python ground truth (16 tokens for [{role:user, content:"TEST"}]):
//     [2, 105, 9731, 107, 98, 106, 107,
//      105, 2364, 107, 20721, 106, 107,
//      105, 4368, 107]
//
//   swift-jinja output (11 tokens for the same input):
//     [2, 108, 105, 2364, 107, 20721, 106, 107, 105, 4368, 107]
//
// The system turn (5 tokens) is dropped AND the second token id is wrong
// (108 instead of 105). The model receives a token sequence it has never
// seen during training and degrades to raw-language-modeling — fluent
// local tokens, zero instruction following.
//
// FIX
// ───
// Bypass the jinja path entirely. Manually format the prompt with the
// turn markers as literal strings, then call `tokenizer.encode(text:)`,
// which respects the special tokens registered in `tokenizer.json`
// (verified: ids 2/105/106 all have `special=true`).
//
// VERIFICATION
// ────────────
// `Tests/Gemma4SwiftCoreTests/PromptFormattingTests` asserts the resulting
// token sequence matches Python `tokenizer.apply_chat_template` output
// byte-for-byte. The opt-in network test downloads the real tokenizer.json
// from HuggingFace to perform this check.
//
// SPDX-License-Identifier: MIT

import Foundation

/// Builds prompt strings in Gemma 4's `<|turn>` chat format.
///
/// All methods produce literal strings. Token-level handling is left to
/// the caller, who should pass the result through their tokenizer's
/// `encode(text:)` method (which recognizes `<bos>`, `<|turn>`, and
/// `<turn|>` as registered special tokens).
public enum Gemma4PromptFormatter {

    /// Wrap a single user message for the Gemma 4 chat protocol.
    ///
    /// The output looks like:
    /// ```
    /// <bos><|turn>user
    /// {message}<turn|>
    /// <|turn>model
    ///
    /// ```
    ///
    /// This is the simplest possible format — no system turn, no
    /// `<|think|>` injection. The model will produce the assistant
    /// response immediately, without an English chain-of-thought block.
    /// Use this for production text generation where you want clean,
    /// instruction-following output.
    ///
    /// - Parameter message: The user-facing prompt text. Special-token
    ///   characters in the message are NOT escaped — pass plain text.
    /// - Returns: A formatted prompt string ready for `tokenizer.encode`.
    public static func userTurn(_ message: String) -> String {
        "<bos><|turn>user\n\(message)<turn|>\n<|turn>model\n"
    }

    /// Wrap a system + user pair, opting in to Gemma 4's thinking mode.
    ///
    /// The output looks like:
    /// ```
    /// <bos><|turn>system
    /// <|think|><turn|>
    /// <|turn>user
    /// {user}<turn|>
    /// <|turn>model
    ///
    /// ```
    ///
    /// Adding the empty system turn with `<|think|>` causes the model to
    /// produce a `<|channel>thought ... <channel|>` block before the
    /// final answer. Useful for tasks where you want chain-of-thought
    /// reasoning. The caller is responsible for stripping the thought
    /// channel from the output if it's not wanted in the final UI.
    ///
    /// - Parameters:
    ///   - user: The user-facing prompt text.
    ///   - includeThinking: When `true` (the default), inject the
    ///     `<|think|>` token in the system turn.
    public static func userTurnWithThinking(
        _ user: String,
        includeThinking: Bool = true
    ) -> String {
        let thinkToken = includeThinking ? "<|think|>" : ""
        return "<bos><|turn>system\n\(thinkToken)<turn|>\n<|turn>user\n\(user)<turn|>\n<|turn>model\n"
    }

    /// Format a multi-turn conversation.
    ///
    /// Each turn is wrapped in its own `<|turn>{role}\n...<turn|>\n`
    /// block. The trailing `<|turn>model\n` invites the next assistant
    /// response. Pass `bosFirst: true` (the default) to prefix `<bos>`.
    ///
    /// - Parameters:
    ///   - turns: Ordered list of `(role, content)` pairs. Roles are
    ///     conventionally `"user"` and `"model"` in Gemma 4 — the
    ///     formatter does not validate this.
    ///   - bosFirst: Whether to prefix the entire string with `<bos>`.
    ///     Set to `false` if your tokenizer auto-adds BOS.
    public static func conversation(
        turns: [(role: String, content: String)],
        bosFirst: Bool = true
    ) -> String {
        var result = bosFirst ? "<bos>" : ""
        for turn in turns {
            result += "<|turn>\(turn.role)\n\(turn.content)<turn|>\n"
        }
        result += "<|turn>model\n"
        return result
    }
}
