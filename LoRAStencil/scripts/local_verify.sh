#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
arch="${LORASTENCIL_ARCH:-86}"
build_dir="${LORASTENCIL_BUILD_DIR:-"${repo_root}/build-sm86-check"}"

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
  -DLORASTENCIL_ENABLE_CHECKS=ON

cmake --build "${build_dir}" -j "${LORASTENCIL_BUILD_JOBS:-24}"

run_case() {
  echo
  echo "==> $*"
  "$@"
}

run_case "${build_dir}/lorastencil_1d" 1d1r 1024 1
run_case "${build_dir}/lorastencil_1d" 1d2r 1024 1

run_case "${build_dir}/lorastencil_2d" star2d1r 64 64 1
run_case "${build_dir}/lorastencil_2d" box2d1r 64 64 1
run_case "${build_dir}/lorastencil_2d" star2d3r 64 64 1
run_case "${build_dir}/lorastencil_2d" box2d3r 64 64 1

run_case "${build_dir}/lorastencil_3d" star3d1r 8 8 64 1
run_case "${build_dir}/lorastencil_3d" box3d1r 8 8 64 1

if command -v cuobjdump >/dev/null 2>&1; then
  if cuobjdump --dump-sass "${build_dir}/lorastencil_2d" | grep -q "DMMA\\.884"; then
    echo
    echo "SASS check: found DMMA.884 in lorastencil_2d."
  else
    echo "SASS check failed: DMMA.884 not found in lorastencil_2d." >&2
    exit 3
  fi
else
  echo
  echo "SASS check skipped: cuobjdump not found."
fi
