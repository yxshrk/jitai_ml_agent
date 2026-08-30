#!/usr/bin/env bash

# Download and verify the public KuaiRand-Pure data used by the Project 2 starter kit.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kit_dir="${project_dir}/kuairand-starter-kit"
archive_path="${kit_dir}/KuaiRand-Pure.tar.gz"
partial_path="${archive_path}.partial"
data_dir="${kit_dir}/KuaiRand-Pure/data"
data_url="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
required_files=(
  "video_features_basic_pure.csv"
  "log_standard_4_08_to_4_21_pure.csv"
  "log_standard_4_22_to_5_08_pure.csv"
)

dataset_complete() {
  for file_name in "${required_files[@]}"; do
    [[ -s "${data_dir}/${file_name}" ]] || return 1
  done
}

if [[ ! -d "${kit_dir}" ]]; then
  echo "Starter kit not found at ${kit_dir}" >&2
  exit 1
fi

if ! dataset_complete; then
  if [[ -f "${archive_path}" ]] && ! tar -tzf "${archive_path}" >/dev/null 2>&1; then
    if [[ -e "${partial_path}" ]]; then
      echo "Both a failed archive and a partial download exist; resolve them before retrying." >&2
      exit 1
    fi
    mv "${archive_path}" "${partial_path}"
  fi

  if [[ ! -f "${archive_path}" ]]; then
    echo "Downloading KuaiRand-Pure data..."
    curl --fail --location --retry 3 --retry-delay 2 --continue-at - --output "${partial_path}" "${data_url}"
    tar -tzf "${partial_path}" >/dev/null
    mv "${partial_path}" "${archive_path}"
  fi

  tar -xzf "${archive_path}" -C "${kit_dir}"
fi

if ! dataset_complete; then
  echo "Dataset is incomplete after extraction: expected files under ${data_dir}" >&2
  exit 1
fi

echo "KuaiRand-Pure data is ready at ${data_dir}"
