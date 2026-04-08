<p align="center">
  <img src="docs/images/banner.svg" alt="Gemma4SwiftCore" width="720">
</p>

<h1 align="center">Gemma4SwiftCore</h1>

<p align="center">
  <strong>Google Gemma 4의 세계 최초 네이티브 Swift 구현체.</strong><br>
  iPhone, iPad, Mac에서 작동합니다. 100% 온디바이스. 런타임에 Python 의존성 없음.
</p>

<p align="center">
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9%2B-orange.svg" alt="Swift 5.9+"></a>
  <a href="#설치"><img src="https://img.shields.io/badge/Platform-iOS%2017%20%7C%20macOS%2014-blue.svg" alt="플랫폼"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT 라이선스"></a>
  <a href="https://github.com/yejingyang8963-byte/Swift-gemma4-core/actions"><img src="https://img.shields.io/badge/Tests-passing-brightgreen.svg" alt="테스트 통과"></a>
  <a href="https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit"><img src="https://img.shields.io/badge/Model-Gemma%204%20E2B%204bit-purple.svg" alt="Gemma 4 E2B 4bit"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <strong>한국어</strong> ·
  <a href="README.es.md">Español</a>
</p>

---

## 이게 뭔가요?

`Gemma4SwiftCore`는 Google [Gemma 4](https://huggingface.co/google) 텍스트
디코더를 **순수 Swift로 포팅한 구현체**입니다. Apple의
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)에 사이드카
모델로 등록되므로, HuggingFace의 어떤 Gemma 4 저장소(예:
`mlx-community/gemma-4-e2b-it-4bit`)든 Llama나 Qwen을 로드하는 것과 동일한
방식으로 로드할 수 있습니다. 이번엔 **실제로 작동하는** Gemma 4로 말이죠.

런타임에 Python이 필요 없습니다. CoreML 변환 단계도 없습니다. 토큰 ID에서
로짓까지 모든 과정이 Apple MLX의 Metal 커널 위에서 완전히 온디바이스로
실행됩니다.

## 왜 만들었나요?

2026년 4월 본 프로젝트가 시작될 당시 `mlx-swift-lm` 2.31.x에는 Gemma 4
지원이 전혀 없었습니다. 가장 단순한 우회 방법인 Gemma 3 텍스트 구현을
빌려와 config에 패치를 가하는 방식은 가중치 로딩 단계에서 필드 누락
오류로 실패합니다. Gemma 4는 Gemma 3와 구조적으로 **다섯 군데가 다른**
모델이기 때문입니다. 게다가 swift-jinja의 chat template 렌더링 경로는
**프롬프트를 조용히 손상시킵니다**. 결과적으로 모델은 국소적으로는 유창
하지만 사용자의 지시를 전혀 따르지 않는 출력을 생성하게 됩니다.

이 패키지는 두 가지 문제를 한 번에 해결합니다. 디코더 전체를 Swift로
처음부터 포팅했고, Python `mlx-lm`의 `tokenizer.apply_chat_template`과
**바이트 단위로 일치하는** chat template 우회 도구를 함께 제공합니다.

## 핵심 기여

- 🧠 **Per-Layer Embedding (PLE)** — Gemma 4의 시그니처 기능. 모든
  디코더 레이어가 공유 임베딩 테이블에서 토큰별 벡터를 가져와 작은
  MLP를 통해 게이팅한 후 세 번째 잔차로 더합니다.

- 🔗 **레이어 간 KV 공유** — E2B의 35개 레이어 중 뒤쪽 20개 레이어는
  K/V를 직접 계산하지 않고 같은 attention 타입의 이전 레이어가 계산한
  K/V를 재사용합니다. 본 구현은 forward pass에 "도너 테이블"을 통과시키
  고, **전역 rope 오프셋**을 사용해 생성 단계의 위치 정보를 정확하게
  유지합니다.

- 🎯 **Proportional RoPE** — Gemma 4 full-attention 레이어를 위한 부분
  회전 RoPE 클래스를 직접 구현했습니다. `mlx-swift-lm`에 내장된
  `initializeRope`는 이 rope type을 인식하지 못하므로, Python 참조 구현과
  바이트 단위로 일치하는 ``Gemma4ProportionalRoPE``를 함께 제공합니다.

- 💬 **Chat template 우회** — `swift-jinja` 1.x는 Gemma 4의 chat
  template을 **잘못 렌더링합니다**. 토큰 5개를 누락시키고 시스템 턴의
  두 번째 토큰 ID도 틀립니다. 본 구현은 그 경로를 완전히 우회하여
  `<|turn>` 마커를 문자열 리터럴로 직접 조립한 후
  `tokenizer.encode(text:)`로 인코딩합니다(특수 토큰은 tokenizer.json에
  이미 등록되어 있습니다).

자세한 내용은
[Architecture 문서](Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md)를
참고하세요.

## 성능

실제 iPhone(Apple A 시리즈, RAM 7.4 GB)에서
`mlx-community/gemma-4-e2b-it-4bit`을 측정한 결과:

| 지표 | 측정값 | 목표 |
|---|---|---|
| 콜드 로드(다운로드 포함) | ~110초 | 1회성 |
| 웜 로드(캐시 히트) | ~6초 | — |
| 로드 후 메모리 | 341–392 MB | < 2 GB ✅ |
| 첫 오디오 청크까지 시간 | **2.82초** | < 3초 ✅ |
| 생성 속도 | 12–14 tok/s | — |

2.82초의 첫 청크 지연은 실제 출시된 앱의 TTS 파이프라인을 통과한
엔드투엔드 측정값입니다(모델은 핫 상태, 333 토큰 시스템 프롬프트). 순수
forward pass 처리량은 이보다 더 높습니다.

## 설치

`Package.swift`에 추가:

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

또는 Xcode에서: **File → Add Package Dependencies...** → 저장소 URL을
붙여넣으세요.

## 빠른 시작

```swift
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

// 1. mlx-swift-lm에 사이드카 핸들러를 등록합니다. 멱등성 보장.
await Gemma4Registration.registerIfNeeded().value

// 2. HuggingFace에서 실제 4-bit 가중치를 로드합니다.
//    모델은 ~1.5 GB이며 첫 다운로드 후 캐시됩니다.
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))

// 3. chat template 우회 도구로 프롬프트를 포매팅합니다.
//    tokenizer.applyChatTemplate은 사용하지 마세요 — Gemma 4에서 깨져 있습니다.
let prompt = Gemma4PromptFormatter.userTurn("호기심 많은 여우에 대한 짧은 이야기를 들려줘")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))

// 4. 스트리밍 생성.
let stream = try await container.generate(
    input: input,
    parameters: GenerateParameters(maxTokens: 200, temperature: 0.8, topP: 0.95))
for await event in stream {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

## 테스트

```bash
# 순수 Swift 단위 테스트(Configuration, Sanitize, ProportionalRoPE 수학,
# PromptFormatter 리터럴). Swift가 동작하는 어디서든 실행 가능:
swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"

# MLX를 포함하는 전체 테스트 스위트. Apple Silicon + Xcode 필요:
xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'

# 옵션: 네트워크 통합 테스트(실제 tokenizer를 다운로드하여 Python ground truth와 대조):
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

## 재현성 증명

본 구현의 chat template 우회가 Python `mlx-lm`과 동일한 토큰 ID를
생성한다는 것을 직접 확인하고 싶다면, Apple Silicon Mac에서
`scripts/python_baseline.py`를 실행하세요:

```bash
python3 -m venv ~/.mlx-venv
source ~/.mlx-venv/bin/activate
pip install mlx-lm
python scripts/python_baseline.py
```

같은 모델을 로드하고, 같은 프롬프트를 포매팅한 후, `Gemma4PromptFormatter.userTurn`이
출력하는 토큰 ID를 바이트 단위로 나란히 출력합니다. 둘은 일치합니다.

## 자주 묻는 질문

**Q: 모델 가중치를 직접 다운로드해야 하나요?**
네. 본 패키지에는 가중치가 포함되어 있지 않습니다(4-bit 체크포인트만
~1.5 GB). `loadContainer`를 처음 호출할 때 HuggingFace hub 클라이언트를
통해 플랫폼 표준 캐시 디렉토리로 다운로드됩니다.

**Q: 어떤 디바이스에서 동작하나요?**
RAM 6 GB 이상의 모든 Apple Silicon 디바이스. iPhone 14 / iPhone 13 Pro
이상, M1 Mac 이상.

**Q: Google이나 Apple의 공식 프로젝트인가요?**
아닙니다. Google은 자체 라이선스로 Gemma 4 가중치를 공개했고, Apple은
mlx-swift와 mlx-swift-lm을 공개했습니다. 본 패키지는 두 프로젝트에
의존하는 독립적인 서드파티 포팅입니다. 자세한 내용은 `NOTICE` 파일을
참고하세요.

**Q: 상용 이용이 가능한가요?**
가능합니다. 본 저장소의 코드는 MIT 라이선스입니다. Gemma 4 가중치
자체에는 Google 자체 라이선스가 적용되므로 상용 배포 전에 반드시
확인하세요.

**Q: 공식 지원을 기다리는 게 낫지 않나요?**
기다려도 됩니다. 다만 공식 지원은 아직 나오지 않았고, 포팅이 필요한
아키텍처적 차이점이 5개 있으며, chat template 문제는 공식 지원이 나와도
우회가 필요합니다. **기다리지 않아도 되도록** 본 패키지가 존재합니다.

## 로드맵

- **v0.2** — KV 캐시 양자화, 긴 컨텍스트 윈도우 벤치마크
- **v0.3** — Gemma 4 E4B 변종 지원, 스트리밍 생성 API
- **v1.0** — 안정 공개 API, SemVer 엄수

## 인용

연구나 상업 프로젝트에서 `Gemma4SwiftCore`를 사용하시면 다음과 같이
인용해 주세요:

```bibtex
@software{ye2026gemma4swiftcore,
  author = {Ye, Jingyang},
  title  = {{Gemma4SwiftCore}: Native Swift Inference for Google Gemma 4},
  year   = {2026},
  url    = {https://github.com/yejingyang8963-byte/Swift-gemma4-core},
  license = {MIT}
}
```

## 감사의 말

- Apple [MLX](https://github.com/ml-explore/mlx) 및
  [mlx-swift](https://github.com/ml-explore/mlx-swift) 팀 — 기반이 되는
  Metal 가속 텐서 라이브러리 제공
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) 컨트리뷰터들 —
  본 패키지가 의존하는 `LLMModel` 프로토콜과 `KVCache` 타입 제공
- Google — Gemma 4 가중치와
  [transformers 참조 구현](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)
  공개

## 작성자

**[Jingyang Ye](https://github.com/yejingyang8963-byte)** (예정양, 叶静阳)이
개발 및 유지보수합니다.

본 프로젝트는 어린이를 위한 온디바이스 AI 수면 동화 생성 비공개 iOS 앱
내부에서 만들어진 작업물을 추출한 것입니다. 오픈소스로 공개하는 이유는
간단합니다. Apple 디바이스에서 Gemma 4를 돌리는 일이 어떤 한 폐쇄형
프로젝트에 의해 독점되어서는 안 되기 때문입니다.

## 라이선스

MIT. 전문은 [LICENSE](LICENSE)를 참고하세요.

Copyright © 2026 Jingyang Ye.
