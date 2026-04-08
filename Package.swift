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
    ],
    dependencies: [
        // Apple's MLX Swift bindings — the underlying tensor library.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.0"),
        // mlx-swift-lm provides the LLMModel protocol, KVCache types,
        // tokenizer integration, and the Hub download client we register against.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.31.0"),
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
