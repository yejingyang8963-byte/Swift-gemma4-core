<p align="center">
  <img src="docs/images/banner.svg" alt="Gemma4SwiftCore" width="720">
</p>

<h1 align="center">Gemma4SwiftCore</h1>

<p align="center">
  <strong>世界上第一个能跑的 Google Gemma 4 原生 Swift 实现。</strong><br>
  iPhone、iPad、Mac 全平台支持。100% 端侧推理。运行时无 Python。
</p>

<p align="center">
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9%2B-orange.svg" alt="Swift 5.9+"></a>
  <a href="#安装"><img src="https://img.shields.io/badge/Platform-iOS%2017%20%7C%20macOS%2014-blue.svg" alt="平台"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT 协议"></a>
  <a href="https://github.com/yejingyang8963-byte/Swift-gemma4-core/actions"><img src="https://img.shields.io/badge/Tests-passing-brightgreen.svg" alt="测试通过"></a>
  <a href="https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit"><img src="https://img.shields.io/badge/Model-Gemma%204%20E2B%204bit-purple.svg" alt="Gemma 4 E2B 4bit"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>简体中文</strong> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a>
</p>

---

## 这是什么？

`Gemma4SwiftCore` 是 Google [Gemma 4](https://huggingface.co/google) 文本
解码器的**纯 Swift 移植**。它以 sidecar 模型注册的方式接入 Apple 的
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)，让任意
HuggingFace Gemma 4 仓库（比如 `mlx-community/gemma-4-e2b-it-4bit`）
都能像加载 Llama 或 Qwen 一样直接加载——只是这次 Gemma 4 是**真的能跑**。

运行时**没有 Python**，**没有 CoreML 转换**步骤。从 token id 到 logits
全程跑在 Apple MLX 的 Metal kernel 上，完全端侧。

## 为什么需要这个？

2026 年 4 月本项目立项时，`mlx-swift-lm` 2.31.x **没有 Gemma 4 支持**。
最朴素的临时方案——借用 Gemma 3 文本实现 + 打补丁 config——会在权重
加载时因为缺失字段而崩溃。原因是 Gemma 4 跟 Gemma 3 在五个地方**结构上
不同**。而 swift-jinja 的 chat template 渲染路径会**静默地破坏 prompt**，
让模型变成「字面流畅但完全不听话」的乱码生成器。

本仓库一次性解决两个问题：把整个 decoder 用 Swift 从零移植，并且自带
一个 chat template 绕过方案，token 序列跟 Python `mlx-lm` 的
`tokenizer.apply_chat_template` **逐字节一致**。

## 核心创新

- 🧠 **Per-Layer Embedding (PLE)** —— Gemma 4 的招牌特性。每个 decoder
  层都从一个共享的 embedding 表里取一个 per-token 向量，经过一个小 MLP
  门控，作为第三个 residual 加回主路径。

- 🔗 **跨层 KV 共享** —— E2B 的 35 层 decoder 中后 20 层不计算自己的
  K/V，而是复用前面相同 attention 类型层的 K/V。我们用一个「donor 表」
  在 forward pass 里串起来，并且用一个**全局 rope offset** 保证生成
  阶段的位置编码不出错。

- 🎯 **Proportional RoPE** —— 为 Gemma 4 full-attention 层自创的部分
  旋转 RoPE 类。`mlx-swift-lm` 内置的 `initializeRope` 不认识这个
  rope type，我们写了 ``Gemma4ProportionalRoPE``，跟 Python 参考实现
  逐字节对齐。

- 💬 **Chat template 绕过** —— `swift-jinja` 1.x 渲染 Gemma 4 的 chat
  template 时**会丢 5 个 token、第二个 token id 都是错的**。我们绕过它
  整条路径，用 `<|turn>` 标记符直接拼字符串，然后通过
  `tokenizer.encode(text:)` 编码（特殊 token 已经在 tokenizer.json 里
  注册了）。

完整的技术细节见
[Architecture 文章](Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md)。

## 性能数据

在真实 iPhone（Apple A 系列芯片，7.4 GB RAM）上用
`mlx-community/gemma-4-e2b-it-4bit` 实测：

| 指标 | 实测值 | 目标 |
|---|---|---|
| 冷启动加载（含下载） | ~110 秒 | 一次性 |
| 热启动加载（缓存命中） | ~6 秒 | — |
| 加载后内存 | 341–392 MB | < 2 GB ✅ |
| 首音延迟 | **2.82 秒** | < 3 秒 ✅ |
| 生成速度 | 12–14 tok/s | — |

2.82 秒首音延迟是端到端经过 TTS 流水线测出来的（一个真实上线的应用），
模型已加载、333 token 的 system prompt。纯 forward pass 的吞吐更高。

## 安装

在你的 `Package.swift` 里加：

```swift
dependencies: [
    .package(
        url: "https://github.com/yejingyang8963-byte/Swift-gemma4-core.git",
        from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "Gemma4SwiftCore", package: "Swift-gemma4-core"),
        ]),
],
```

或者在 Xcode 里：**File → Add Package Dependencies...** → 粘贴仓库 URL。

## 快速上手

```swift
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

// 1. 注册 sidecar handler 到 mlx-swift-lm。幂等，多次调用安全。
await Gemma4Registration.registerIfNeeded().value

// 2. 从 HuggingFace 下载真实的 4-bit 权重。
//    模型 ~1.5 GB，首次下载之后会缓存。
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))

// 3. 用 chat template 绕过工具格式化 prompt。
//    千万不要用 tokenizer.applyChatTemplate——它在 Gemma 4 上是坏的。
let prompt = Gemma4PromptFormatter.userTurn("讲一个关于好奇小狐狸的短故事")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))

// 4. 流式生成。
let stream = try await container.generate(
    input: input,
    parameters: GenerateParameters(maxTokens: 200, temperature: 0.8, topP: 0.95))
for await event in stream {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

## 测试

```bash
# 纯 Swift 单元测试（Configuration、Sanitize、ProportionalRoPE 数学、
# PromptFormatter 字面量）。任何 Swift 环境都能跑：
swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"

# 完整测试套件（含 MLX 模块形状测试）。需要 Apple Silicon + Xcode：
xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'

# 可选的网络集成测试（下载真实 tokenizer 并对照 Python ground truth）：
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

## 可复现性证明

想自己验证我们的 chat template 绕过工具产出的 token 跟 Python `mlx-lm`
完全一致？在任意 Apple Silicon Mac 上跑 `scripts/python_baseline.py`：

```bash
python3 -m venv ~/.mlx-venv
source ~/.mlx-venv/bin/activate
pip install mlx-lm
python scripts/python_baseline.py
```

它会加载同一个模型、用同一个 prompt、并且把 token id 跟
`Gemma4PromptFormatter.userTurn` 的输出**逐字节并排打印**。两边一致。

## 常见问题

**问：我需要自己下载模型权重吗？**
需要。本包不打包模型权重（光 4-bit checkpoint 就 ~1.5 GB）。第一次调
`loadContainer` 时会通过 HuggingFace hub 客户端下载到平台标准的 caches
目录里。

**问：哪些设备能跑？**
任何 RAM ≥ 6 GB 的 Apple Silicon 设备。iPhone 14 / iPhone 13 Pro 及更
新机型，M1 Mac 及更新机型。

**问：这跟 Google 或 Apple 有官方关系吗？**
没有。Google 用自己的 license 发布 Gemma 4 权重。Apple 发布 mlx-swift
和 mlx-swift-lm。本包是一个独立的端口，依赖这两个开源项目。第三方
归属信息见 `NOTICE`。

**问：我能商用吗？**
能。本仓库的代码用 MIT 协议发布。但 Gemma 4 权重本身有 Google 自己的
license，发布商业产品前请先看清楚那个。

**问：为什么不等上游官方支持？**
你可以等。但上游一直没出，架构上有 5 个独立特性需要移植，chat template
那个问题就算上游出了也得绕。这个包就是让你**不用等**。

## 路线图

- **v0.2** — KV cache 量化、更长 context window 性能数据
- **v0.3** — Gemma 4 E4B 变种支持、流式生成 API
- **v1.0** — 公开 API 稳定、严格遵守 SemVer

## 引用

如果你在研究或商业项目里用了 `Gemma4SwiftCore`，请引用：

```bibtex
@software{ye2026gemma4swiftcore,
  author = {Ye, Jingyang},
  title  = {{Gemma4SwiftCore}: Native Swift Inference for Google Gemma 4},
  year   = {2026},
  url    = {https://github.com/yejingyang8963-byte/Swift-gemma4-core},
  license = {MIT}
}
```

## 致谢

- Apple [MLX](https://github.com/ml-explore/mlx) 和
  [mlx-swift](https://github.com/ml-explore/mlx-swift) 团队提供底层
  Metal 加速张量库
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) 贡献者
  提供 `LLMModel` 协议和 `KVCache` 类型
- Google 提供 Gemma 4 权重和
  [transformers 参考实现](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)

## 作者

由 **[Jingyang Ye 叶静阳](https://github.com/yejingyang8963-byte)**
开发并维护。

本项目脱胎于一个私有 iOS 应用——给孩子讲端侧 AI 睡前故事。决定开源
是因为：在苹果设备上跑 Gemma 4 这种事情，**不应该被任何一个闭源项目
垄断**。

## 许可证

MIT。完整文本见 [LICENSE](LICENSE)。

Copyright © 2026 Jingyang Ye.
