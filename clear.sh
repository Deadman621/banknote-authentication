#!/usr/bin/env bash

set -euo pipefail

ROOT="${1:-.}"

declare -A REPLACEMENTS=(
  ["from utils."]="from src.utils."
  ["from preprocessing."]="from src.preprocessing."
  ["from core."]="from src.core."
  ["from models."]="from src.models."
  ["from datasets."]="from src.datasets."
  ["from training."]="from src.training."
  ["from evaluation."]="from src.evaluation."
  ["from explainability."]="from src.explainability."
  ["from inference."]="from src.inference."
  ["from augmentation."]="from src.augmentation."
  ["from checkpoint."]="from src.checkpoint."
  ["from dashboard."]="from src.dashboard."
  ["from modules."]="from src.modules."
)

echo "Updating imports under ${ROOT}..."

find "$ROOT" \
    \( -path "*/.git" -o -path "*/.venv" -o -path "*/__pycache__" \) -prune \
    -o -name "*.py" -print |
while read -r file; do
    for old in "${!REPLACEMENTS[@]}"; do
        new="${REPLACEMENTS[$old]}"
        sed -i "s|^${old}|${new}|g" "$file"
    done
done

echo "Done."