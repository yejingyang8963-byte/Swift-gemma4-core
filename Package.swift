// swift-tools-version: 5.9
//
// Gemma4SwiftCore — Native Swift inference for Google Gemma 4
// https://github.com/yejingyang8963-byte/Swift-gemma4-core
//
// Copyright (c) 2026 Jingyang Ye
// Licensed under the MIT License — see LICENSE for details.

import PackageDescription

let package = Package(
    name: "Gemma4SwiftCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "Gemma4SwiftCore",
            targets: ["Gemma4SwiftCore"]
        ),
        .executable(
            name: "Gemma4Verify",
            targets: ["Gemma4Verify"]
        ),
    ],
    dependencies: [
        // Apple's MLX Swift bindings — the underlying tensor library.
        // Pinned to 0.30.x to satisfy mlx-swift-lm 2.30.x's dependency.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", "0.30.3" ..< "0.31.0"),
        // mlx-swift-lm provides the LLMModel protocol, KVCache types,
        // tokenizer integration, and the Hub download client we register against.
        // Pinned to 2.30.x (latest version with Swift 5.x tools support).
        // 2.31.3 bumped swift-tools-version to 6.1.0 which requires Xcode 16.3+
        // and breaks every CI runner that ships an older Xcode by default.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", "2.30.0" ..< "2.31.0"),
        // Build-time-only plugin used by `swift package generate-documentation`
        // in .github/workflows/docc.yml to publish API docs to GitHub Pages.
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "Gemma4SwiftCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/Gemma4SwiftCore"
        ),
        .executableTarget(
            name: "Gemma4Verify",
            dependencies: [
                "Gemma4SwiftCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/Gemma4Verify"
        ),
        .testTarget(
            name: "Gemma4SwiftCoreTests",
            dependencies: [
                "Gemma4SwiftCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/Gemma4SwiftCoreTests",
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
