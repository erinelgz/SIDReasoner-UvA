#!/bin/bash
# Fetch the public Amazon data archive linked by HappyPointer/SIDReasoner.
# This archive is used only for provenance/compatibility checking; it never
# overwrites the project's existing frozen data files.

set -euo pipefail

artifact_root="${ARTIFACT_ROOT:-/gpfs/work5/0/prjs2120/groups/group_06/checkpoints/SIDReasoner/author_final_baseline_test_20260623}"
archive_dir="${artifact_root}/source_data"
archive_path="${archive_dir}/Amazon.zip"
source_url="https://drive.google.com/file/d/1etg1e8oStGOjsg1Vr15vFnjlTMUx4Htz/view?usp=sharing"

mkdir -p "$archive_dir"

if [[ ! -f "$archive_path" ]]; then
    uvx --from gdown gdown --no-cookies --continue -O "$archive_path" "$source_url"
fi

unzip -tq "$archive_path" >/dev/null
archive_sha256="$(sha256sum "$archive_path" | awk '{print $1}')"
printf 'Author public Amazon archive: %s\n' "$archive_path"
printf 'Archive SHA-256: %s\n' "$archive_sha256"

for member in \
    'Amazon/test/Office_Products_5_2016-10-2018-11.csv' \
    'Amazon/info/Office_Products_5_2016-10-2018-11.txt' \
    'Amazon/index/Office_Products.index.json' \
    'Amazon/index/Office_Products.item.json'; do
    printf '%s  %s\n' "$(unzip -p "$archive_path" "$member" | sha256sum | awk '{print $1}')" "$member"
done
