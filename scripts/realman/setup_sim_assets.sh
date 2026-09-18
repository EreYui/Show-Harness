#!/usr/bin/env bash
set -euo pipefail

asset_root="${RMC_AIDAL_ROOT:-$HOME/Ecosystem_Cases_RMC_AIDA_L}"
repository="https://github.com/RealManRobot/Ecosystem_Cases.git"
package_path="模型文件/URDF/具身URDF/Embodied lifting robot_two wheels_RM65-B-V"

if [[ -f "$asset_root/$package_path/urdf/Embodied lifting robot_two wheels_RM65-B-V.urdf" ]]; then
  echo "RealMan RMC-AIDA-L RM65-B-V model already present: $asset_root"
  exit 0
fi
if [[ -e "$asset_root" ]]; then
  echo "Refusing to replace incomplete path: $asset_root" >&2
  exit 1
fi

git clone --depth 1 --branch RMC-AIDA-L --filter=blob:none --sparse "$repository" "$asset_root"
git -C "$asset_root" sparse-checkout set "$package_path"
echo "Installed official RMC-AIDA-L RM65-B-V model under $asset_root"
echo "Run: python scripts/run_realman_dual_isaaclab.py --scripted --dump-views"
