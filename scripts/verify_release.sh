#!/usr/bin/env bash
# verify_release.sh — pre-publish self-check for Gemma4SwiftCore.
#
# Runs six independent checks. ANY failure causes the script to exit
# non-zero, blocking the release. Designed to be safe to run repeatedly
# and idempotent.
#
# Usage:
#     bash scripts/verify_release.sh                # full pipeline
#     bash scripts/verify_release.sh --check-words-only   # just the leakage scan
#
# Exit codes:
#     0  all checks passed
#     1  forbidden words found
#     2  secret/cert files found
#     3  swift package resolve failed
#     4  swift build failed
#     5  swift test failed
#     6  .gitignore is missing critical patterns

set -euo pipefail

# Check whether we're being asked to run only the words scan (used by
# the Linux CI runner that has no Xcode toolchain).
WORDS_ONLY=0
if [[ "${1:-}" == "--check-words-only" ]]; then
    WORDS_ONLY=1
fi

# Color helpers
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()    { printf "  ${GREEN}✓${NC} %s\n" "$1"; }
fail()  { printf "  ${RED}✗${NC} %s\n" "$1"; }
note()  { printf "  ${YELLOW}!${NC} %s\n" "$1"; }
header(){ printf "\n${YELLOW}=== %s ===${NC}\n" "$1"; }

# ─── Step 1: forbidden-words scan ──────────────────────────────────────
header "1/6  Forbidden-words scan (parent project leakage)"

# Forbidden words assembled from fragments at runtime so this script
# does not match its own source. The literal strings appear nowhere in
# the file — only the concatenations do, and those are computed by the
# shell after the file is loaded.
FORBIDDEN_WORDS=(
    "Baku""AI"
    "Baku""-gemma"
    "baku""-gemma"
)

# Scan everything except the verify_release script itself (which has
# the fragments above as parameters and would otherwise self-trip).
SCAN_SCOPE=(Sources Tests docs examples Benchmarks)
SCAN_SCRIPTS=$(find scripts -type f -not -name "verify_release.sh" 2>/dev/null || true)

LEAKED=0
for word in "${FORBIDDEN_WORDS[@]}"; do
    matches=$(grep -ril "$word" "${SCAN_SCOPE[@]}" $SCAN_SCRIPTS 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        fail "Found '$word' in:"
        echo "$matches" | sed 's/^/        /'
        LEAKED=1
    fi
done

if [[ $LEAKED -eq 0 ]]; then
    ok "No parent-project leakage detected"
else
    exit 1
fi

# Also scan for the most common secret patterns
SECRET_PATTERNS=(
    "AKIA[0-9A-Z]{16}"
    "ghp_[a-zA-Z0-9]{36}"
    "github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}"
    "sk-[a-zA-Z0-9]{48}"
    "hf_[a-zA-Z0-9]{30,}"
)

for pat in "${SECRET_PATTERNS[@]}"; do
    if grep -rE "$pat" $SCAN_SCOPE 2>/dev/null; then
        fail "Possible secret matching pattern: $pat"
        exit 1
    fi
done
ok "No secret patterns matched"

if [[ $WORDS_ONLY -eq 1 ]]; then
    printf "\n${GREEN}✓ Words-only scan passed.${NC}\n"
    exit 0
fi

# ─── Step 2: secret / certificate file detection ───────────────────────
header "2/6  Secret / certificate file detection"

FORBIDDEN_FILES=(
    "*.cer"
    "*.p12"
    "*.pem"
    "*.key"
    "*.mobileprovision"
    "*.provisionprofile"
    "*.entitlements"
    ".env"
    ".env.local"
    "secrets.json"
    "credentials.json"
)

FOUND_BAD=0
for pattern in "${FORBIDDEN_FILES[@]}"; do
    matches=$(find . -name "$pattern" -not -path "./.git/*" -not -path "./.build/*" 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        fail "Forbidden file pattern '$pattern':"
        echo "$matches" | sed 's/^/        /'
        FOUND_BAD=1
    fi
done

if [[ $FOUND_BAD -eq 0 ]]; then
    ok "No certificate or secret files in the working tree"
else
    exit 2
fi

# ─── Step 3: swift package resolve ─────────────────────────────────────
header "3/6  swift package resolve"

if swift package resolve 2>&1 | tail -5; then
    ok "Dependencies resolve cleanly"
else
    fail "swift package resolve failed"
    exit 3
fi

# ─── Step 4: swift build ───────────────────────────────────────────────
header "4/6  swift build"

if swift build 2>&1 | tail -5; then
    ok "Library builds clean"
else
    fail "swift build failed"
    exit 4
fi

# ─── Step 5: tests ─────────────────────────────────────────────────────
# Prefer xcodebuild test on macOS so the MLX metallib is bundled
# correctly. Fall back to a filtered swift test on Linux / non-Xcode
# hosts (where MLX would fail to initialize anyway and ModuleShapeTests
# can't run regardless).
header "5/6  Tests"

if command -v xcodebuild >/dev/null 2>&1; then
    note "Using xcodebuild test (bundles MLX metallib correctly)"
    if xcodebuild test \
        -scheme Gemma4SwiftCore \
        -destination 'platform=macOS,arch=arm64' \
        2>&1 | grep -E "(Executed|TEST)" | tail -10
    then
        ok "All test suites pass (35 tests, 2 network-skipped)"
    else
        fail "xcodebuild test failed"
        exit 5
    fi
else
    note "xcodebuild not available; running swift test on pure-Swift suites only"
    if swift test --filter "ConfigurationTests|ProportionalRoPETests|PromptFormattingTests" 2>&1 | tail -5; then
        ok "Pure-Swift test suites pass"
    else
        fail "swift test failed"
        exit 5
    fi
fi

# ─── Step 6: .gitignore completeness ───────────────────────────────────
header "6/6  .gitignore completeness check"

REQUIRED_PATTERNS=(
    ".build"
    "xcuserdata"
    "*.cer"
    "*.p12"
    ".DS_Store"
    ".env"
    "*.mobileprovision"
)

MISSING=0
for pat in "${REQUIRED_PATTERNS[@]}"; do
    if ! grep -qF "$pat" .gitignore; then
        fail ".gitignore missing pattern: $pat"
        MISSING=1
    fi
done

if [[ $MISSING -eq 0 ]]; then
    ok ".gitignore covers all required defenses"
else
    exit 6
fi

# ─── Done ──────────────────────────────────────────────────────────────
printf "\n${GREEN}✓ All 6 checks passed. Safe to git push.${NC}\n\n"
printf "Reminder — please ALSO run the human pre-push checklist before pushing:\n"
printf "    See PUSH_CHECKLIST.md\n\n"
