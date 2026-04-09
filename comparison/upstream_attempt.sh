#!/usr/bin/env bash
# upstream_attempt.sh — reproducible proof that adrgrondin/mlx-swift-lm
# does not register a gemma4 model type, so it cannot load
# mlx-community/gemma-4-e2b-it-4bit.
#
# This script does NOT download model weights. It only clones the
# upstream source (~30 MB shallow) and greps the model registry.
#
# Exit codes:
#   0  upstream lacks gemma4 — Gemma4SwiftCore's claim holds
#   1  upstream registers gemma4 — our claim is OUT OF DATE, retire it
#   2  network / git failure
#
# SPDX-License-Identifier: MIT

set -euo pipefail

UPSTREAM_REPO="https://github.com/adrgrondin/mlx-swift-lm.git"
UPSTREAM_BRANCH="main"
WORK_DIR="$(mktemp -d -t gemma4-upstream-audit-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

REGISTRY_REL="Libraries/MLXLLM/LLMModelFactory.swift"

echo "── Gemma4SwiftCore upstream gap audit ─────────────────────────"
echo "Date     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Upstream : $UPSTREAM_REPO ($UPSTREAM_BRANCH)"
echo "Workdir  : $WORK_DIR"
echo

echo "[1/3] Shallow-cloning upstream …"
if ! git clone --depth 1 --branch "$UPSTREAM_BRANCH" "$UPSTREAM_REPO" \
        "$WORK_DIR/repo" 2>&1 | sed 's/^/    /'; then
    echo "ERROR: git clone failed (network?)" >&2
    exit 2
fi

REGISTRY_PATH="$WORK_DIR/repo/$REGISTRY_REL"
if [ ! -f "$REGISTRY_PATH" ]; then
    echo "ERROR: $REGISTRY_REL not found in upstream — directory layout changed." >&2
    exit 2
fi

echo
echo "[2/3] All model_type strings registered in upstream:"
# Upstream uses a Swift dictionary literal of the form
#     "model_type": create(SomeConfig.self, SomeModel.init),
# inside LLMTypeRegistry.shared. Match those entries.
TYPES=$(grep -oE '"[a-zA-Z0-9_-]+"[[:space:]]*:[[:space:]]*create\(' "$REGISTRY_PATH" \
        | sed -E 's/"([^"]+)".*/\1/' \
        | sort -u)
echo "$TYPES" | sed 's/^/    /'
TYPE_COUNT=$(echo "$TYPES" | wc -l | tr -d ' ')
echo
echo "Total model types registered: $TYPE_COUNT"

echo
echo "[3/3] Checking for gemma4 / gemma4_text …"
if echo "$TYPES" | grep -qE '^(gemma4|gemma4_text)$'; then
    echo
    echo "❌ UNEXPECTED: upstream NOW registers a gemma4 type."
    echo "   Our claim 'Gemma4SwiftCore is the only Swift package that"
    echo "   runs Gemma 4' is OUT OF DATE. Retire this audit and update"
    echo "   comparison/README.md."
    exit 1
fi

GEMMA_TYPES=$(echo "$TYPES" | grep -E '^gemma' || true)
echo "Gemma family in upstream:"
echo "${GEMMA_TYPES:-    (none)}" | sed 's/^/    /'
echo
echo "✅ Confirmed: upstream lacks 'gemma4' and 'gemma4_text'."
echo "   Gemma4SwiftCore remains the only Swift package that loads"
echo "   mlx-community/gemma-4-e2b-it-4bit."
exit 0
