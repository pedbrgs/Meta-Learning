#!/bin/bash
set -e

mkdir -p /meta-learning/data

ZIP_PATH="/meta-learning/data/Benchmark.zip"
FLAG="/meta-learning/data/.downloaded"

if [ ! -f "$FLAG" ]; then
  echo "Downloading datasets from OneDrive..."

  wget \
    --content-disposition \
    --trust-server-names \
    -O "$ZIP_PATH" \
    "https://www.dropbox.com/scl/fi/cpw05xojxbzqh3e2fq56m/Benchmark.zip?rlkey=ib4rr53dobgiof2dy9uxji02u&dl=1"

  echo "Validating ZIP file..."
  if ! unzip -t "$ZIP_PATH" >/dev/null 2>&1; then
    echo "ERROR: Downloaded file is not a valid ZIP (OneDrive HTML page?)"
    file "$ZIP_PATH"
    exit 1
  fi

  echo "Extracting dataset..."
  unzip "$ZIP_PATH" -d /meta-learning/data

  rm "$ZIP_PATH"
  touch "$FLAG"

  echo "Dataset downloaded and extracted successfully."
else
  echo "Dataset already downloaded."
fi
