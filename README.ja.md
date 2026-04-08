<p align="center">
  <img src="docs/images/banner.svg" alt="Gemma4SwiftCore" width="720">
</p>

<h1 align="center">Gemma4SwiftCore</h1>

<p align="center">
  <strong>Google Gemma 4 のネイティブ Swift 実装、世界初。</strong><br>
  iPhone・iPad・Mac で動作。100% オンデバイス。実行時の Python 依存ゼロ。
</p>

<p align="center">
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9%2B-orange.svg" alt="Swift 5.9+"></a>
  <a href="#インストール"><img src="https://img.shields.io/badge/Platform-iOS%2017%20%7C%20macOS%2014-blue.svg" alt="プラットフォーム"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT ライセンス"></a>
  <a href="https://github.com/yejingyang8963-byte/Swift-gemma4-core/actions"><img src="https://img.shields.io/badge/Tests-passing-brightgreen.svg" alt="テスト合格"></a>
  <a href="https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit"><img src="https://img.shields.io/badge/Model-Gemma%204%20E2B%204bit-purple.svg" alt="Gemma 4 E2B 4bit"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <strong>日本語</strong> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a>
</p>

---

## これは何ですか？

`Gemma4SwiftCore` は、Google [Gemma 4](https://huggingface.co/google) の
テキストデコーダを **純 Swift で書き直したもの** です。Apple の
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) にサイドカー
モデルとして登録されるので、HuggingFace の任意の Gemma 4 リポジトリ
（例: `mlx-community/gemma-4-e2b-it-4bit`）を Llama や Qwen と同じ手順で
ロードできます。**今度はちゃんと動く** Gemma 4 として。

実行時に Python は不要です。CoreML への変換も不要です。トークン ID から
ロジットまですべて Apple MLX の Metal カーネル上で完結し、完全に
オンデバイスで動作します。

## なぜ存在するのか

2026 年 4 月時点で `mlx-swift-lm` 2.31.x には Gemma 4 サポートが存在
しませんでした。素朴な回避策——Gemma 3 のテキスト実装を借用して config
にパッチを当てる——は重み読み込み時に「フィールドが見つからない」
エラーで失敗します。Gemma 4 は Gemma 3 とは構造的に **5 か所** 異なる
モデルだからです。さらに swift-jinja の chat template レンダリングは
**プロンプトを静かに壊します**。結果としてモデルはローカルには流暢
ですが、命令には一切従わない出力を返します。

このパッケージは両方の問題を一度に解決します。デコーダ全体を Swift
にゼロから移植し、Python `mlx-lm` の `tokenizer.apply_chat_template`
**とバイト単位で一致する** chat template バイパスを同梱しています。

## 主な技術的貢献

- 🧠 **Per-Layer Embedding (PLE)** — Gemma 4 を象徴する機能。各
  デコーダ層が共有埋め込みテーブルからトークン単位のベクトルを取得
  し、小さな MLP でゲートし、第 3 の残差として加算します。

- 🔗 **後半層での KV 共有** — E2B の 35 層のうち後ろ 20 層は K/V を
  自前で計算せず、同じ attention タイプの先行層が計算した K/V を再利用
  します。本実装では「ドナーテーブル」を forward pass に通し、
  **グローバルな rope オフセット** を使って生成時の位置情報を正しく
  保ちます。

- 🎯 **Proportional RoPE** — Gemma 4 の full-attention 層用にカスタム
  実装した部分回転 RoPE クラス。`mlx-swift-lm` 内蔵の `initializeRope`
  はこの rope type を認識しないので、Python の参考実装と
  バイト単位で一致する ``Gemma4ProportionalRoPE`` を同梱しています。

- 💬 **Chat template バイパス** — `swift-jinja` 1.x は Gemma 4 の
  chat template のレンダリングが**正しく動きません**。トークンを 5 個
  落とし、システムターンの 2 番目のトークン ID を間違えます。本実装
  ではそのパスを完全に回避し、`<|turn>` マーカーを文字列リテラルとして
  プロンプトを組み立て、`tokenizer.encode(text:)` で符号化します
  （特殊トークンは tokenizer.json に登録済みです）。

詳細は
[Architecture 記事](Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md)
をご覧ください。

## パフォーマンス

実機 iPhone（Apple A シリーズ、RAM 7.4 GB）で
`mlx-community/gemma-4-e2b-it-4bit` を実測：

| 指標 | 実測値 | 目標 |
|---|---|---|
| コールドロード（DL 込み） | ~110 秒 | 一回限り |
| ウォームロード（キャッシュヒット） | ~6 秒 | — |
| ロード後メモリ | 341–392 MB | < 2 GB ✅ |
| 最初の音声チャンクまでの時間 | **2.82 秒** | < 3 秒 ✅ |
| 生成スループット | 12–14 tok/s | — |

2.82 秒の最初のチャンクは、実際にリリースされたアプリの TTS パイプ
ラインを通したエンドツーエンド計測値です（モデルはホット、333 トークン
のシステムプロンプトあり）。純粋な forward pass のスループットは
これよりさらに高速です。

## インストール

`Package.swift` に追加：

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

または Xcode で：**File → Add Package Dependencies...** からリポジトリ
URL を貼り付けてください。

## クイックスタート

```swift
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

// 1. mlx-swift-lm にサイドカーハンドラを登録します。冪等。
await Gemma4Registration.registerIfNeeded().value

// 2. HuggingFace から実際の 4-bit 重みをロードします。
//    モデルは ~1.5 GB で、初回ダウンロード後はキャッシュされます。
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))

// 3. chat template バイパスでプロンプトを整形します。
//    tokenizer.applyChatTemplate は使わないでください — Gemma 4 では壊れています。
let prompt = Gemma4PromptFormatter.userTurn("好奇心旺盛なキツネの短いお話を聞かせて")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))

// 4. ストリーミング生成。
let stream = try await container.generate(
    input: input,
    parameters: GenerateParameters(maxTokens: 200, temperature: 0.8, topP: 0.95))
for await event in stream {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

## テスト

```bash
# 純粋な Swift ユニットテスト（Configuration、Sanitize、ProportionalRoPE、
# PromptFormatter）。Swift が動く環境であればどこでも実行可能：
swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"

# MLX を含む完全なテストスイート。Apple Silicon + Xcode が必要：
xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'

# オプションのネットワーク統合テスト（実際の tokenizer をダウンロードして
# Python のグラウンドトゥルースと照合）：
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

## 再現性の証明

本実装の chat template バイパスが Python `mlx-lm` と完全に一致する
トークン ID を生成することを自分で確かめたい場合は、Apple Silicon Mac
上で `scripts/python_baseline.py` を実行してください：

```bash
python3 -m venv ~/.mlx-venv
source ~/.mlx-venv/bin/activate
pip install mlx-lm
python scripts/python_baseline.py
```

同じモデルをロードし、同じプロンプトを整形し、`Gemma4PromptFormatter.userTurn`
が出力するトークン ID をバイト単位で並べて表示します。両者は一致します。

## よくある質問

**Q: モデルの重みは自分でダウンロードする必要がありますか？**
はい。本パッケージには重みは同梱されていません（4-bit checkpoint だけで
~1.5 GB あります）。`loadContainer` への最初の呼び出しで HuggingFace の
hub クライアント経由でプラットフォーム標準のキャッシュディレクトリに
ダウンロードされます。

**Q: どのデバイスで動きますか？**
RAM 6 GB 以上の Apple Silicon デバイスならどれでも動きます。iPhone 14
や iPhone 13 Pro 以降、M1 Mac 以降。

**Q: Google や Apple の公式プロジェクトですか？**
いいえ。Google は独自ライセンスで Gemma 4 の重みを公開しています。
Apple は mlx-swift と mlx-swift-lm を公開しています。本パッケージは
両者に依存する独立したサードパーティの移植です。詳細は `NOTICE`
ファイルをご覧ください。

**Q: 商用利用できますか？**
できます。本リポジトリのコードは MIT ライセンスです。Gemma 4 の重み
そのものには Google 独自のライセンスが適用されますので、商用配布の
前に必ず確認してください。

**Q: 公式サポートを待てばいいのでは？**
待ってもいいです。ですが、公式サポートはまだ来ていませんし、移植が
必要なアーキテクチャの相違点が 5 つあり、chat template の問題は公式
サポートが来ても回避策が必要です。**待たずに済むように** このパッケージ
は存在します。

## ロードマップ

- **v0.2** — KV キャッシュの量子化、長コンテキストのベンチマーク
- **v0.3** — Gemma 4 E4B サポート、ストリーミング生成 API
- **v1.0** — 安定版公開 API、SemVer 厳守

## 引用

研究や商用プロジェクトで `Gemma4SwiftCore` をお使いの場合は、以下を
引用してください：

```bibtex
@software{ye2026gemma4swiftcore,
  author = {Ye, Jingyang},
  title  = {{Gemma4SwiftCore}: Native Swift Inference for Google Gemma 4},
  year   = {2026},
  url    = {https://github.com/yejingyang8963-byte/Swift-gemma4-core},
  license = {MIT}
}
```

## 謝辞

- Apple [MLX](https://github.com/ml-explore/mlx) および
  [mlx-swift](https://github.com/ml-explore/mlx-swift) チーム — 基盤
  となる Metal 加速テンソルライブラリの提供
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) コントリ
  ビューターの皆様 — 本パッケージが依存する `LLMModel` プロトコルと
  `KVCache` 型の提供
- Google — Gemma 4 の重みと
  [transformers リファレンス実装](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)
  の公開

## 作者

**[Jingyang Ye 葉静陽](https://github.com/yejingyang8963-byte)** が
開発・保守しています。

本プロジェクトは、子供向けのオンデバイス AI 寝かしつけ絵本を生成する
プライベート iOS アプリの中で生まれた成果を抽出したものです。
オープンソースとして公開する理由は単純で、Apple デバイスで Gemma 4 を
動かすという話題が、誰か一つのクローズドなプロジェクトに独占されるべき
ではないと考えたからです。

## ライセンス

MIT。全文は [LICENSE](LICENSE) をご覧ください。

Copyright © 2026 Jingyang Ye.
