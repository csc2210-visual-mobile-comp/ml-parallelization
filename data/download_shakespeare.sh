#!/bin/bash
set -e

OUT_FILE="shakespeare.txt"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

echo "Downloading Tiny Shakespeare..."
wget -O "${OUT_FILE}" "${URL}"

echo "Done."
echo "Saved to $(pwd)/${OUT_FILE}"
echo "Lines: $(wc -l < ${OUT_FILE})"
echo "Bytes: $(wc -c < ${OUT_FILE})"
