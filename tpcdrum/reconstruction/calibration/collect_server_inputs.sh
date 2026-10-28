#!/usr/bin/env bash
set -eu

# Standard-library Python only. Use the same initialized simulation environment
# as run_sim.exe if requesting the optional native Geant4 material export.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../.." && pwd)"
python_cmd="${PYTHON_BIN:-python3}"
output_dir="${RECO_EXPORT_DIR:-$HOME/workspace/reco_inputs_$(date +%Y%m%d_%H%M%S)}"

exec "$python_cmd" "$script_dir/export_collect_inputs.py" \
  --repo "$repo_dir" --output "$output_dir" "$@"
