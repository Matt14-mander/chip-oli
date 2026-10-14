#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 2 ]]; then
  echo "Usage: ./scripts/apply_overlay.sh /path/to/mjlab /path/to/RoboJuDo" >&2
  exit 1
fi

MJLAB_TARGET="$(cd "$1" && pwd)"
ROBOJUDO_TARGET="$(cd "$2" && pwd)"

copy_tree() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "$dst_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src_dir/" "$dst_dir/"
  else
    cp -a "$src_dir/." "$dst_dir/"
  fi
}

copy_tree "$ROOT_DIR/train/mjlab_overlay" "$MJLAB_TARGET"
copy_tree "$ROOT_DIR/deploy/robojudo_overlay" "$ROBOJUDO_TARGET"
copy_tree "$ROOT_DIR/shared/chip_common" "$MJLAB_TARGET/src/mjlab/chip_common"
copy_tree "$ROOT_DIR/shared/chip_common" "$ROBOJUDO_TARGET/robojudo/chip_common"
copy_tree "$ROOT_DIR/robot_assets/oli" "$MJLAB_TARGET/src/mjlab/asset_zoo/robots/oli/assets"
copy_tree "$ROOT_DIR/robot_assets/oli" "$ROBOJUDO_TARGET/assets/oli"

echo "Overlay applied."
echo "mjlab target: $MJLAB_TARGET"
echo "RoboJuDo target: $ROBOJUDO_TARGET"
