#!/usr/bin/env bash
# Copy only this NEW macro directory into an existing server checkout.
# Never pull, reset, clean, overwrite task sources, or change simulation data.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash %s /path/to/existing/lilak\n' "$0" >&2
    exit 2
fi

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
target_repo="$(cd -- "$1" && pwd -P)"
target_parent="$target_repo/tpcdrum/simulation"
destination="$target_parent/hoyle_prepare_20261027"

if [[ ! -d "$target_parent" ]]; then
    printf 'Expected existing simulation directory: %s\n' "$target_parent" >&2
    exit 1
fi
if [[ -e "$destination" || -L "$destination" ]]; then
    printf 'Stopped: destination already exists; nothing was overwritten: %s\n' "$destination" >&2
    exit 1
fi

# mkdir also refuses a destination created between the check and this command.
# If copying fails, leave the new partial directory for inspection; do not
# remove or modify anything elsewhere in the existing checkout.
mkdir -- "$destination"
cp -a -- "$source_dir/." "$destination/"
printf 'Added macro directory: %s\n' "$destination"
printf 'Existing files and .git metadata were not changed.\n'
printf 'MC selection can now run there with the existing LILAK dictionaries.\n'
printf 'Digitization additionally needs your previously built SiEdep and metadata tasks.\n'
