# Pull Request

## Summary

<!-- One paragraph: what does this PR change and why? -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing code to fail)
- [ ] Documentation only (DocC, README, CHANGELOG, comments)
- [ ] CI / build / tooling
- [ ] Refactor (no behavior change)

## Checklist

- [ ] My code follows the project style — every public symbol has a `///` doc comment
- [ ] Every modified or new file is **≤ 200 lines**
- [ ] I added tests covering my change
- [ ] Pure-Swift tests pass: `swift test --filter "ConfigurationTests|SanitizeTests|ProportionalRoPETests|PromptFormattingTests"`
- [ ] Full test suite passes: `xcodebuild test -scheme Gemma4SwiftCore -destination 'platform=macOS,arch=arm64'`
- [ ] `swiftlint --strict` is clean
- [ ] `bash scripts/verify_release.sh` passes
- [ ] CHANGELOG.md is updated under the `## [Unreleased]` section
- [ ] No `BakuAI`, no proprietary prompts, no API keys, no certificates

## For numerical changes (forward pass / RoPE / KV sharing / sanitize)

Per [CONTRIBUTING.md](../CONTRIBUTING.md), numerical changes must be
backed by side-by-side Python `mlx-lm` evidence.

- [ ] I ran `scripts/python_baseline.py` (or a modified version) to capture the reference
- [ ] I added or updated a test that asserts the new value
- [ ] I pasted the Python output below

```
<paste python output here>
```

```
<paste swift test output here>
```

## Related issues

<!-- Closes #123 -->

## Notes for reviewer

<!-- Anything tricky, surprising, or worth flagging? -->
