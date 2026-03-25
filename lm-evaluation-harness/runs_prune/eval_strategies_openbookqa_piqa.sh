#!/usr/bin/env bash
# Run openbookqa + piqa only for multiple strategies (faster than full eval)
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

EXPERT_CAPACITY="${EXPERT_CAPACITY:-1.0}"
PRETRAINED="${PRETRAINED:-./models/OLMoE-1B-7B-0924}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
STRATEGIES="${STRATEGIES:-score score_sq score_rank_boost score_margin_add}"

if [[ ! -d "${PRETRAINED}" ]]; then
  echo "Model path does not exist: ${PRETRAINED}" >&2
  exit 1
fi

echo "PRETRAINED=${PRETRAINED}"
echo "EXPERT_CAPACITY=${EXPERT_CAPACITY}"
echo "STRATEGIES=${STRATEGIES}"

for STRATEGY in ${STRATEGIES}; do
  OUTPUT_PATH="${PRETRAINED}/expert_capacity-${EXPERT_CAPACITY}/${STRATEGY}"
  mkdir -p "${OUTPUT_PATH}"
  echo ""
  echo "=== Running STRATEGY=${STRATEGY} ==="
  python -m lm_eval \
    --model hf \
    --model_args "pretrained=${PRETRAINED},expert_capacity=${EXPERT_CAPACITY},strategy=${STRATEGY},parallelize=True,trust_remote_code=True,dtype=bfloat16" \
    --tasks openbookqa,piqa \
    --num_fewshot 0 \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    2>&1 | tee "${OUTPUT_PATH}/eval.log"
  echo "Done: ${STRATEGY}"
done

echo ""
echo "=== Summary: extract acc_norm from JSON results ==="
for STRATEGY in ${STRATEGIES}; do
  OUTPUT_PATH="${PRETRAINED}/expert_capacity-${EXPERT_CAPACITY}/${STRATEGY}"
  for task in openbookqa piqa; do
    json_file=$(ls -t "${OUTPUT_PATH}/${task}.json"/results_*.json 2>/dev/null | head -1)
    if [[ -n "${json_file}" ]]; then
      acc_norm=$(python3 -c "import json; d=json.load(open('${json_file}')); print(d['results']['${task}']['acc_norm,none'])" 2>/dev/null || echo "N/A")
      echo "  ${STRATEGY} ${task}: acc_norm=${acc_norm}"
    fi
  done
done
