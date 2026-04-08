# Contributing to Gemma4SwiftCore

Thank you for considering a contribution! This document covers the
practical workflow and the project-specific conventions.

## Ground rules

1. **Stay focused on the Gemma 4 text decoder.** This package is
   intentionally narrow. PRs that add unrelated features (different
   model architectures, app-level utilities, UI helpers) will be closed.
   Open a separate package instead.
2. **Every public symbol gets a `///` doc comment.** DocC builds the
   public documentation from these — missing comments break the docs
   site and will block merge.
3. **Every source file is ≤ 200 lines.** If a change pushes a file
   over, split it via a Swift extension in a sibling file. The existing
   `Gemma4TextAttention` / `Gemma4TextAttention+Forward` and
   `Gemma4TextInner` / `Gemma4TextInner+PerLayerInputs` pairs are
   examples of the convention.
4. **Tests for everything.** Numerical changes need a numerical test;
   API changes need a behavior test. The MLX-dependent tests run via
   `xcodebuild test`; pure-Swift tests run via `swift test`.

## Setup

```bash
git clone https://github.com/yejingyang8963-byte/Swift-gemma4-core.git
cd Swift-gemma4-core
swift build
swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"
```

To run the full suite (including MLX-dependent tests) you need
Xcode 15.4+ on Apple Silicon:

```bash
xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'
```

To run the opt-in network test:

```bash
GEMMA4_TEST_NETWORK=1 swift test --filter NetworkIntegrationTests
```

## Submitting a change

1. Fork the repository on GitHub.
2. Create a topic branch from `main`:
   ```bash
   git checkout -b fix/some-bug
   ```
3. Make your changes. Keep commits **atomic** — one logical change per
   commit, descriptive English commit messages following the
   [Conventional Commits](https://www.conventionalcommits.org/) format
   (e.g. `fix(rope): handle rotated_dims = 0 edge case`).
4. Run the full test suite locally before pushing.
5. Open a PR against `main`. Fill out every section of the PR template.
   The CI matrix runs `swift build`, `swift test`, `xcodebuild test`,
   and `swiftlint`. All must pass before review.

## Numerical changes

Anything touching the forward-pass math (Attention, RoPE, PLE, the
KV-sharing rope offset) must be backed by **byte-level evidence** that
the new behavior matches Python `mlx-lm`. The standard procedure:

1. Modify `scripts/python_baseline.py` to dump the relevant
   intermediate (e.g. K/V tensors, hidden state means, top-k token IDs).
2. Run it on Apple Silicon and capture the output.
3. Add or update the corresponding Swift test to assert the same value.
4. Include both the Python output and the Swift test result in the PR
   description, side-by-side.

This is the only kind of evidence the maintainer accepts for numerical
correctness claims. PRs that say "I tested it and it works" without
reproducible Python ground truth will be sent back.

## Style

- Use 4-space indentation.
- One blank line between functions, two between top-level type
  declarations.
- Group module imports alphabetically: Foundation first, then MLX
  modules, then everything else.
- Use Swift 5.9+ features (parameter packs, `@MainActor`, typed
  throws when appropriate). Don't reach for Swift 6 strict concurrency
  unless the change actually requires it.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) 2.1.
Be excellent to each other.

## License

By contributing, you agree that your contributions will be licensed
under the [MIT License](LICENSE).
