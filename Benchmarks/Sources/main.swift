// benchmarks — micro-benchmarks for the lightweight Gemma4SwiftCore
// public surface that doesn't need real model weights.
//
// Each benchmark warms up for 3 iterations, then runs 50 timed
// iterations and reports min / mean / median / max / stddev.
//
// We deliberately do NOT benchmark full forward passes here — that
// would require downloading the 1.5 GB model and would dominate the
// run time with cache I/O. Instead we measure the parts of the
// library that are pure-CPU and reproducible:
//
//   1. Configuration JSON parsing
//   2. Prompt formatting (chat template bypass)
//   3. ProportionalRoPE construction (which builds the freqs array)
//
// For end-to-end forward-pass numbers, see the example app and the
// README "Performance" table.

import Foundation
import Gemma4SwiftCore

// MARK: - Tiny benchmark harness

struct BenchmarkResult {
    let name: String
    let iterations: Int
    let timesNs: [UInt64]

    var min: UInt64 { timesNs.min() ?? 0 }
    var max: UInt64 { timesNs.max() ?? 0 }
    var mean: Double {
        guard !timesNs.isEmpty else { return 0 }
        return Double(timesNs.reduce(0, +)) / Double(timesNs.count)
    }
    var median: UInt64 {
        let sorted = timesNs.sorted()
        return sorted[sorted.count / 2]
    }
    var stddev: Double {
        let m = mean
        let variance = timesNs
            .map { (Double($0) - m) * (Double($0) - m) }
            .reduce(0, +) / Double(timesNs.count)
        return variance.squareRoot()
    }
}

func benchmark(_ name: String, warmup: Int = 3, iterations: Int = 50, _ body: () throws -> Void) rethrows -> BenchmarkResult {
    for _ in 0 ..< warmup { try body() }
    var times: [UInt64] = []
    times.reserveCapacity(iterations)
    for _ in 0 ..< iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        try body()
        let end = DispatchTime.now().uptimeNanoseconds
        times.append(end - start)
    }
    return BenchmarkResult(name: name, iterations: iterations, timesNs: times)
}

func format(_ ns: Double) -> String {
    if ns >= 1_000_000 {
        return String(format: "%.2f ms", ns / 1_000_000)
    }
    if ns >= 1_000 {
        return String(format: "%.2f µs", ns / 1_000)
    }
    return String(format: "%.0f ns", ns)
}

func report(_ r: BenchmarkResult) {
    print(String(format: "%-50s  min=%-12s  median=%-12s  mean=%-12s  stddev=%s",
                 r.name,
                 format(Double(r.min)),
                 format(Double(r.median)),
                 format(r.mean),
                 format(r.stddev)))
}

// MARK: - Workloads

let tinyConfigJSON = """
{
  "model_type": "gemma4_text",
  "hidden_size": 32, "num_hidden_layers": 5, "intermediate_size": 64,
  "num_attention_heads": 4, "num_key_value_heads": 1,
  "head_dim": 8, "global_head_dim": 16,
  "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8,
  "vocab_size_per_layer_input": 100, "use_double_wide_mlp": true,
  "attention_k_eq_v": false, "enable_moe_block": false,
  "sliding_window": 16, "max_position_embeddings": 128,
  "rms_norm_eps": 1e-6, "vocab_size": 100, "tie_word_embeddings": true,
  "final_logit_softcapping": 30.0,
  "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
  "rope_parameters": {
    "sliding_attention": {"rope_type":"default","rope_theta":10000.0},
    "full_attention": {"rope_type":"proportional","rope_theta":1000000.0,"partial_rotary_factor":0.25}
  }
}
""".data(using: .utf8)!

print("Gemma4SwiftCore benchmarks  ·  v\(Gemma4SwiftCore.version)")
print(String(repeating: "─", count: 110))

let configBench = try benchmark("Gemma4TextConfiguration JSON decode (5 layers)") {
    _ = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: tinyConfigJSON)
}
report(configBench)

let promptBench = benchmark("Gemma4PromptFormatter.userTurn (32 char prompt)") {
    _ = Gemma4PromptFormatter.userTurn("Hello, what is your name today?")
}
report(promptBench)

let conversationBench = benchmark("Gemma4PromptFormatter.conversation (3 turns)") {
    _ = Gemma4PromptFormatter.conversation(turns: [
        (role: "user", content: "Hello"),
        (role: "model", content: "Hi there, how are you?"),
        (role: "user", content: "Tell me a short fairy tale."),
    ])
}
report(conversationBench)

print(String(repeating: "─", count: 110))
print("Done.")
