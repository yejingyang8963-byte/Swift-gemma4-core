# Security Policy

## Supported versions

`Gemma4SwiftCore` is at `v0.1.x` — initial public release. The maintainer
will accept security fixes against:

| Version | Supported |
|---|:---:|
| 0.1.x   | ✅ |
| < 0.1.0 | ❌ |

When `v1.0.0` ships, this matrix will be updated to support the current
major version plus one previous major version.

## Reporting a vulnerability

**Please do not file public GitHub issues for security reports.** Public
issues are visible to everyone and give attackers a head-start before a
fix is available.

Instead, send a private report via one of these channels:

1. **GitHub private vulnerability reporting** —
   https://github.com/yejingyang8963-byte/Swift-gemma4-core/security/advisories/new
   (preferred — encrypted, tracked, and the standard place for this)
2. **Email** — open a placeholder issue titled "security: request
   private contact" and the maintainer will respond with a private
   email within 48 hours.

When reporting, please include:

- The version of `Gemma4SwiftCore` you tested against
  (`Gemma4SwiftCore.version`)
- A minimal reproduction (Swift code snippet or test case)
- The impact you believe the issue has
- Whether you have already disclosed the issue elsewhere

## What counts as a security issue

Because this package is a **library that loads model weights and runs
inference**, the threat model is narrower than a network service. Real
security issues include:

- **Memory safety bugs** in the Swift code that could be triggered by
  a malicious model file (out-of-bounds reads/writes, integer overflows
  in shape arithmetic, use-after-free)
- **Tensor parsing bugs** that crash on malformed safetensors weights
- **Path traversal** in any file-handling code paths (although this
  package does no file I/O of its own — `mlx-swift-lm` handles
  downloads)
- **Supply-chain compromise** of the published Git tags or releases

What does **NOT** count as a security issue:

- Model output content (the LLM might say strange things — that's a
  Google model behavior issue, not a `Gemma4SwiftCore` issue)
- Performance regressions (file a regular issue)
- Disagreements about API design (file a feature request)

## Response timeline

| Stage | Target |
|---|---|
| Acknowledgment of receipt | 48 hours |
| Initial assessment | 7 days |
| Fix in main branch | 14 days for critical, 30 days for high |
| Coordinated disclosure | 30 days after fix lands |

## Credit

We're happy to credit security researchers in the release notes and
the relevant `CHANGELOG.md` entry. If you prefer to remain anonymous,
just say so in your report.

## License of this policy

This security policy is licensed under the same MIT terms as the rest
of the project. Feel free to copy and adapt it for your own Swift
package.
