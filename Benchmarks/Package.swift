// swift-tools-version: 5.9
//
// Benchmarks — measures forward-pass perf of Gemma4SwiftCore.
//
// Run:
//     cd Benchmarks
//     swift run -c release benchmarks
//
// Uses XCTest's measure() under the hood for consistency with the rest
// of the test suite. swift-benchmark would also work but adds a heavy
// dependency for marginal benefit.

import PackageDescription

let package = Package(
    name: "Benchmarks",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        .package(name: "Gemma4SwiftCore", path: ".."),
    ],
    targets: [
        .executableTarget(
            name: "benchmarks",
            dependencies: [
                .product(name: "Gemma4SwiftCore", package: "Gemma4SwiftCore"),
            ],
            path: "Sources"
        ),
    ]
)
