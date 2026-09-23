#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

python3 -m pip install --no-cache-dir --root-user-action=ignore -q \
  "transformers>=4.46,<6" "datasets>=3" "accelerate>=1"

nproc="${NPROC_PER_NODE:-gpu}"
nodes="${PET_NNODES:-${KUBE_NODE_SIZE:-1}}"
if [ "$nodes" -gt 1 ] && [ -z "${PET_MASTER_ADDR:-}" ]; then
  job="${KUBE_TRAINJOB_NAME:-${HOSTNAME%-trainer-*}}"
  exec torchrun --nnodes="$nodes" --node_rank="${JOB_COMPLETION_INDEX:-${HOSTNAME##*-}}" \
    --master_addr="${job}-trainer-0-0.${job}" --master_port=29500 \
    --nproc_per_node="$nproc" run_glue.py
fi
exec torchrun --nproc_per_node="$nproc" run_glue.py
