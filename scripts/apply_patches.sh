#!/usr/bin/env bash
# Apply the BitNet-X patch series to the pinned 3rdparty/llama.cpp submodule.
#
# Fixes to the submodule are never committed inside the submodule (its pinned
# commit stays untouched); they live in patches/llama.cpp/NNNN-*.patch and are
# applied to the submodule *working tree* by this script.
#
#   scripts/apply_patches.sh            apply every patch that is not applied yet
#   scripts/apply_patches.sh --check    only report; exit 1 if something is not applied
#   scripts/apply_patches.sh --reverse  un-apply all applied patches (reverse order)
#
# Idempotent: a patch whose reverse applies cleanly is reported as "already
# applied" and skipped. A patch that neither applies nor reverse-applies is an
# error (wrong submodule commit or local edits) and nothing further is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/3rdparty/llama.cpp"
PDIR="$ROOT/patches/llama.cpp"
PIN="390c307752ab78fd8189f359d6954c9ba1be74af"
MODE="${1:-apply}"

if [ ! -e "$SUB/.git" ]; then
    echo "apply_patches: submodule not checked out: $SUB (run: git submodule update --init --recursive)" >&2
    exit 2
fi
HEAD_SHA="$(git -C "$SUB" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$PIN" ]; then
    echo "apply_patches: submodule HEAD is $HEAD_SHA, expected pin $PIN" >&2
    exit 2
fi

shopt -s nullglob
PATCHES=("$PDIR"/[0-9][0-9][0-9][0-9]-*.patch)
if [ "${#PATCHES[@]}" -eq 0 ]; then
    echo "apply_patches: no patches in $PDIR"
    exit 0
fi

is_applied()   { git -C "$SUB" apply --reverse --check "$1" >/dev/null 2>&1; }
can_apply()    { git -C "$SUB" apply --check "$1" >/dev/null 2>&1; }

rc=0
case "$MODE" in
apply)
    for p in "${PATCHES[@]}"; do
        n="$(basename "$p")"
        if is_applied "$p"; then
            echo "apply_patches: already applied  $n"
        elif can_apply "$p"; then
            git -C "$SUB" apply "$p"
            echo "apply_patches: applied          $n"
        else
            echo "apply_patches: CANNOT APPLY     $n" >&2
            git -C "$SUB" apply --check "$p" || true
            exit 1
        fi
    done
    ;;
--check)
    for p in "${PATCHES[@]}"; do
        n="$(basename "$p")"
        if is_applied "$p"; then
            echo "apply_patches: applied          $n"
        elif can_apply "$p"; then
            echo "apply_patches: NOT applied      $n"; rc=1
        else
            echo "apply_patches: CONFLICT         $n"; rc=1
        fi
    done
    ;;
--reverse)
    for ((i=${#PATCHES[@]}-1; i>=0; i--)); do
        p="${PATCHES[$i]}"; n="$(basename "$p")"
        if is_applied "$p"; then
            git -C "$SUB" apply --reverse "$p"
            echo "apply_patches: reverted         $n"
        else
            echo "apply_patches: not applied      $n"
        fi
    done
    ;;
*)
    echo "usage: $0 [--check|--reverse]" >&2
    exit 2
    ;;
esac
exit $rc
