<p align="center">
  <img src="docs/images/banner.svg" alt="Gemma4SwiftCore" width="720">
</p>

<h1 align="center">Gemma4SwiftCore</h1>

<p align="center">
  <strong>La primera implementación nativa en Swift de Google Gemma 4.</strong><br>
  Funciona en iPhone, iPad y Mac. 100% en el dispositivo. Sin Python en tiempo de ejecución.
</p>

<p align="center">
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9%2B-orange.svg" alt="Swift 5.9+"></a>
  <a href="#instalación"><img src="https://img.shields.io/badge/Platform-iOS%2017%20%7C%20macOS%2014-blue.svg" alt="Plataforma"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="Licencia MIT"></a>
  <a href="https://github.com/yejingyang8963-byte/Swift-gemma4-core/actions"><img src="https://img.shields.io/badge/Tests-passing-brightgreen.svg" alt="Tests aprobados"></a>
  <a href="https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit"><img src="https://img.shields.io/badge/Model-Gemma%204%20E2B%204bit-purple.svg" alt="Gemma 4 E2B 4bit"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <strong>Español</strong>
</p>

---

## ¿Qué es esto?

`Gemma4SwiftCore` es una **migración pura a Swift** del decodificador de
texto de [Gemma 4](https://huggingface.co/google) de Google. Se conecta
a [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) de Apple
como un registro de modelo paralelo, de modo que cualquier repositorio
de Gemma 4 en HuggingFace (por ejemplo, `mlx-community/gemma-4-e2b-it-4bit`)
puede cargarse de la misma forma que cargarías un modelo Llama o Qwen,
con la diferencia de que ahora Gemma 4 **realmente funciona**.

No hay Python en tiempo de ejecución. No hay un paso de conversión a
CoreML. Todo el camino desde los IDs de tokens hasta los logits corre
sobre los kernels Metal de MLX de Apple, completamente en el dispositivo.

## ¿Por qué existe esto?

Cuando este proyecto comenzó en abril de 2026, `mlx-swift-lm` 2.31.x no
tenía soporte para Gemma 4. La solución más obvia —reutilizar la
implementación de texto de Gemma 3 y parchear el config— falla al cargar
los pesos con un error de campo faltante, porque Gemma 4 es
estructuralmente distinto de Gemma 3 en cinco lugares. Y el camino del
chat template a través de swift-jinja **corrompe el prompt en silencio**,
dejando al modelo fluido pero incoherente.

Este paquete resuelve ambos problemas a la vez: porta todo el
decodificador a Swift desde cero y trae un puente de chat template que
produce secuencias de tokens **idénticas byte por byte** a la salida de
`tokenizer.apply_chat_template` de `mlx-lm` de Python.

## Innovaciones clave

- 🧠 **Per-Layer Embedding (PLE)** — el rasgo distintivo de Gemma 4.
  Cada capa del decodificador recibe un vector por token de una tabla
  de embeddings compartida, lo gatea a través de una pequeña MLP y lo
  añade como un tercer residual.

- 🔗 **Compartición de KV en la mitad posterior del decodificador** —
  las últimas 20 de las 35 capas en E2B reutilizan los tensores K/V de
  capas anteriores del mismo tipo de atención. Encadenamos una "tabla
  de donantes" a través del forward pass y usamos un **rope offset
  global** para mantener las posiciones correctas durante la generación.

- 🎯 **Proportional RoPE** — una clase RoPE personalizada de rotación
  parcial para las capas full-attention de Gemma 4. El `initializeRope`
  integrado de `mlx-swift-lm` no reconoce este tipo de RoPE; nosotros
  enviamos nuestra propia ``Gemma4ProportionalRoPE`` que coincide
  byte por byte con la implementación de referencia en Python.

- 💬 **Bypass del chat template** — `swift-jinja` 1.x renderiza el chat
  template de Gemma 4 **incorrectamente** (pierde 5 tokens, equivoca el
  ID del segundo token del turno de sistema). Nosotros saltamos esa
  ruta por completo y construimos el prompt como una cadena literal
  con marcadores `<|turn>`, luego lo codificamos vía
  `tokenizer.encode(text:)`, que respeta los tokens especiales
  registrados.

Para el análisis técnico completo, consulta el
[artículo de Architecture](Sources/Gemma4SwiftCore/Documentation.docc/Architecture.md).

## Rendimiento

Medido en un iPhone real (Apple A-series, 7.4 GB de RAM) con el
checkpoint `mlx-community/gemma-4-e2b-it-4bit`:

| Métrica | Valor | Objetivo |
|---|---|---|
| Carga en frío (descarga + init) | ~110 s | una sola vez |
| Carga en caliente (caché) | ~6 s | — |
| Memoria tras la carga | 341–392 MB | < 2 GB ✅ |
| Tiempo hasta el primer chunk de audio | **2.82 s** | < 3 s ✅ |
| Velocidad de generación | 12–14 tok/s | — |

La latencia de 2.82 s del primer chunk se midió de extremo a extremo a
través de la pipeline TTS de una app real ya publicada (modelo en
caliente, prompt de sistema de 333 tokens). El throughput puro del
forward pass es aún mayor.

## Instalación

Añade `Gemma4SwiftCore` a tu `Package.swift`:

```swift
dependencies: [
    .package(
        url: "https://github.com/yejingyang8963-byte/Swift-gemma4-core.git",
        from: "0.1.0"),
],
targets: [
    .target(
        name: "TuApp",
        dependencies: [
            .product(name: "Gemma4SwiftCore", package: "Swift-gemma4-core"),
        ]),
],
```

O en Xcode: **File → Add Package Dependencies...** y pega la URL del
repositorio.

## Inicio rápido

```swift
import Gemma4SwiftCore
import MLX
import MLXLLM
import MLXLMCommon

// 1. Registra el handler paralelo con mlx-swift-lm. Idempotente.
await Gemma4Registration.registerIfNeeded().value

// 2. Carga los pesos reales de 4-bit desde HuggingFace.
//    El modelo pesa ~1.5 GB y se cachea tras la primera descarga.
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: Gemma4SwiftCore.verifiedModelId))

// 3. Formatea el prompt usando el bypass del chat template.
//    NO uses tokenizer.applyChatTemplate — está roto en Gemma 4.
let prompt = Gemma4PromptFormatter.userTurn("Cuéntame un cuento corto sobre un zorro curioso.")
let tokens = await container.encode(prompt)
let input = LMInput(tokens: MLXArray(tokens))

// 4. Genera tokens en streaming.
let stream = try await container.generate(
    input: input,
    parameters: GenerateParameters(maxTokens: 200, temperature: 0.8, topP: 0.95))
for await event in stream {
    if case .chunk(let text) = event {
        print(text, terminator: "")
    }
}
```

## Tests

```bash
# Tests unitarios puros en Swift (Configuration, Sanitize, ProportionalRoPE,
# PromptFormatter). Funcionan donde sea que Swift funcione:
swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"

# Suite completa de tests incluyendo los de MLX. Requiere Apple Silicon + Xcode:
xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'

# Tests opcionales de integración con red (descargan el tokenizer real
# y verifican IDs de tokens contra la verdad de Python):
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

## Reproducibilidad

¿Quieres verificar que nuestro bypass del chat template produce los
mismos IDs de tokens que `mlx-lm` en Python? Ejecuta el script de
baseline en cualquier Mac con Apple Silicon:

```bash
python3 -m venv ~/.mlx-venv
source ~/.mlx-venv/bin/activate
pip install mlx-lm
python scripts/python_baseline.py
```

Carga el mismo modelo, formatea el mismo prompt e imprime los IDs de
tokens lado a lado con lo que produciría `Gemma4PromptFormatter.userTurn`.
Coinciden.

## Preguntas frecuentes

**P: ¿Tengo que descargar los pesos del modelo?**
Sí — no vienen empaquetados con esta librería (solo el checkpoint de
4-bit pesa ~1.5 GB). La primera llamada a `loadContainer` los descarga
vía el cliente del hub de HuggingFace al directorio de cachés
estándar de la plataforma.

**P: ¿En qué dispositivos puede correr esto?**
Cualquier dispositivo con Apple Silicon y al menos 6 GB de RAM. iPhone 14
o iPhone 13 Pro y posteriores, Mac M1 y posteriores.

**P: ¿Está esto afiliado a Google o a Apple?**
No. Google publica los pesos de Gemma 4 bajo su propia licencia. Apple
publica mlx-swift y mlx-swift-lm. Este paquete es una migración
independiente que depende de ambos. Consulta `NOTICE` para la atribución
de terceros.

**P: ¿Puedo usar esto en una app comercial?**
Sí — este código está bajo licencia MIT. Los pesos de Gemma 4 vienen
con su propia licencia de Google; revísala antes de publicar.

**P: ¿Por qué no esperar al soporte oficial?**
Podrías. Pero el soporte oficial todavía no llega, la arquitectura tiene
cinco características distintas que necesitan portarse, y el problema
del chat template seguiría necesitando un workaround. Este paquete
existe para que **no tengas que esperar**.

## Hoja de ruta

- **v0.2** — Cuantización del KV cache, benchmarks de contexto largo
- **v0.3** — Soporte para la variante Gemma 4 E4B, API de generación en streaming
- **v1.0** — API pública estable, compromiso con SemVer

## Citación

Si usas `Gemma4SwiftCore` en investigación o trabajo comercial,
por favor cita:

```bibtex
@software{ye2026gemma4swiftcore,
  author = {Ye, Jingyang},
  title  = {{Gemma4SwiftCore}: Native Swift Inference for Google Gemma 4},
  year   = {2026},
  url    = {https://github.com/yejingyang8963-byte/Swift-gemma4-core},
  license = {MIT}
}
```

## Agradecimientos

- El equipo de Apple [MLX](https://github.com/ml-explore/mlx) y
  [mlx-swift](https://github.com/ml-explore/mlx-swift) por la librería
  de tensores acelerada por Metal sobre la que está construido todo esto
- Los contribuidores de [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
  por el protocolo `LLMModel` y los tipos `KVCache` que este paquete
  utiliza
- Google por los pesos de Gemma 4 y la
  [implementación de referencia en transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)

## Autor

Construido y mantenido por
**[Jingyang Ye](https://github.com/yejingyang8963-byte)** (叶静阳).

Este proyecto es la destilación open source de un trabajo originalmente
hecho dentro de una app iOS privada que genera cuentos para dormir
infantiles con IA en el dispositivo. Lo libero porque correr Gemma 4 en
dispositivos Apple no debería estar bloqueado tras un proyecto cerrado.

## Licencia

MIT. Consulta [LICENSE](LICENSE) para el texto completo.

Copyright © 2026 Jingyang Ye.
