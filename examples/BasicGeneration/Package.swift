// swift-tools-version: 5.9
//
// BasicGeneration — minimal CLI demo of Gemma4SwiftCore.
//
// Build:
//     cd examples/BasicGeneration
//     swift build -c release
//
// Run:
//     swift run -c release gemma4-generate "Hello, what is your name?"
//
// On first run this downloads ~1.5 GB of model weights from HuggingFace.
// Subsequent runs are warm.

import PackageDescription

let package = Package(
    name: "BasicGeneration",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        // Reference the parent package by relative path so this example
        // always builds against the local source — no network round-trip
        // and no version drift.
        .package(name: "Gemma4SwiftCore", path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "gemma4-generate",
            dependencies: [
                .product(name: "Gemma4SwiftCore", package: "Gemma4SwiftCore"),
            ],
            path: "Sources"
        ),
    ]
)
