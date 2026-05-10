#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/Users/peremarti-puig/Desktop/Image_Captioning/.venv/bin/python}"
GLOVE="${GLOVE:-$ROOT/data/embeddings/glove.6B.300d.txt}"
W2V="${W2V:-$ROOT/data/embeddings/GoogleNews-vectors-negative300.bin}"

mkdir -p .mplconfig checkpoints checkpoints_attention data/flickr30k_hf data/embeddings

run_base_f8k() {
  local run="$1"; shift
  MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1   "$PY" -m src.baseline.train     --images-dir data/flickr8k/Images     --captions-csv data/flickr8k/captions.txt     --vocab-path data/flickr8k/vocab.pkl     --checkpoints-dir checkpoints/"$run"     --vocab-threshold 5 --embed-size 256 --hidden-size 512 --num-layers 1     --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999     --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20     --scheduler plateau --semantic-temp 10.0     --wandb --wandb-entity learning6 --wandb-project Clean     --run-name "$run" "$@"
}

run_attn_f8k() {
  local run="$1"; shift
  MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1   "$PY" -m src.attention.train     --images-dir data/flickr8k/Images     --captions-csv data/flickr8k/captions.txt     --vocab-path data/flickr8k/vocab.pkl     --checkpoints-dir checkpoints_attention/"$run"     --vocab-threshold 5 --embed-size 256 --hidden-size 512 --attention-dim 256     --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999     --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20     --scheduler plateau --ds-lambda 1.0 --label-smoothing 0.0 --semantic-temp 10.0     --wandb --wandb-entity learning6 --wandb-project Clean     --run-name "$run" "$@"
}

run_base_f30hf() {
  local run="$1"; shift
  MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1   "$PY" -m src.baseline.train     --images-dir data/flickr30k_hf/Images --captions-csv data/flickr30k_hf/captions.txt     --vocab-path data/flickr30k_hf/vocab.pkl     --checkpoints-dir checkpoints/"$run"     --vocab-threshold 5 --embed-size 256 --hidden-size 512 --num-layers 1     --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999     --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20     --scheduler plateau --semantic-temp 10.0     --wandb --wandb-entity learning6 --wandb-project Clean     --run-name "$run" "$@"
}

run_attn_f30hf() {
  local run="$1"; shift
  MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1   "$PY" -m src.attention.train     --images-dir data/flickr30k_hf/Images --captions-csv data/flickr30k_hf/captions.txt     --vocab-path data/flickr30k_hf/vocab.pkl     --checkpoints-dir checkpoints_attention/"$run"     --vocab-threshold 5 --embed-size 256 --hidden-size 512 --attention-dim 256     --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999     --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20     --scheduler plateau --ds-lambda 1.0 --label-smoothing 0.0 --semantic-temp 10.0     --wandb --wandb-entity learning6 --wandb-project Clean     --run-name "$run" "$@"
}

run_backbone() {
  run_base_f8k baseline-flickr8k-resnet50-scratch-ce-lr1e3-20ep
  run_base_f8k baseline-flickr8k-resnet152-scratch-ce-lr1e3-20ep --backbone resnet152
  run_attn_f8k attention-flickr8k-resnet50-scratch-ce-lr1e3-20ep
  run_attn_f8k attention-flickr8k-resnet152-scratch-ce-lr1e3-20ep --backbone resnet152
}

run_embeddings() {
  run_attn_f8k attention-flickr8k-resnet50-glove300d-ce-lr1e3-20ep --glove-path "$GLOVE" --no-semantic-loss
  run_attn_f8k attention-flickr8k-resnet50-word2vec-ce-lr1e3-20ep --word2vec-path "$W2V" --word2vec-binary --no-semantic-loss
  run_base_f8k baseline-flickr8k-resnet50-glove300d-ce-lr1e3-20ep --glove-path "$GLOVE" --no-semantic-loss
  run_base_f8k baseline-flickr8k-resnet50-word2vec-ce-lr1e3-20ep --word2vec-path "$W2V" --word2vec-binary --no-semantic-loss
}

run_loss() {
  run_attn_f8k attention-flickr8k-resnet50-scratch-ce-lr1e3-20ep
  run_attn_f8k attention-flickr8k-resnet50-scratch-ce-ls01-lr1e3-20ep --label-smoothing 0.1
  run_base_f8k baseline-flickr8k-resnet50-scratch-ce-lr1e3-20ep
  run_base_f8k baseline-flickr8k-resnet50-scratch-ce-ls01-lr1e3-20ep --label-smoothing 0.1
}

run_dataset() {
  run_base_f30hf baseline-flickr30k-resnet50-scratch-ce-lr1e3-20ep
  run_attn_f30hf attention-flickr30k-resnet50-scratch-ce-lr1e3-20ep
  run_coco
}

run_learning_rate() {
  run_base_f8k baseline-flickr8k-resnet50-scratch-ce-lr5e4-20ep --lr 0.0005
  run_base_f8k baseline-flickr8k-resnet50-scratch-ce-lr2e3-20ep --lr 0.002
  run_attn_f8k attention-flickr8k-resnet50-scratch-ce-lr5e4-20ep --lr 0.0005
  run_attn_f8k attention-flickr8k-resnet50-scratch-ce-lr2e3-20ep --lr 0.002
}

run_implementation_first() {
  run_base_f8k baseline-flickr8k-efficientnetb0-scratch-ce-lr1e3-20ep --backbone efficientnet_b0
  run_attn_f8k attention-flickr8k-efficientnetb0-scratch-ce-lr1e3-20ep --backbone efficientnet_b0

  run_base_f8k baseline-flickr8k-resnet50-scratch-cycliclr-20ep --scheduler cyclic --base-lr 1e-4 --max-lr 1e-3 --step-size-up-epochs 4
  run_attn_f8k attention-flickr8k-resnet50-scratch-cycliclr-20ep --scheduler cyclic --base-lr 1e-4 --max-lr 1e-3 --step-size-up-epochs 4

  run_base_f8k baseline-flickr8k-resnet50-scratch-bidir-ce-lr1e3-20ep --decoder-direction bidir --skip-test-captioning
  run_attn_f8k attention-flickr8k-resnet50-scratch-bidir-ce-lr1e3-20ep --decoder-direction bidir --skip-test-captioning
}


run_base_coco() {
  local run="$1"; shift
  if [[ ! -f data/coco2017/captions.txt || ! -d data/coco2017/Images ]]; then
    echo "[skip] COCO2017 is not prepared yet. Run scripts/prepare_coco2017.py after downloading COCO."
    return 0
  fi
  MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1   "$PY" -m src.baseline.train     --images-dir data/coco2017/Images     --captions-csv data/coco2017/captions.txt     --vocab-path data/coco2017/vocab.pkl     --checkpoints-dir checkpoints/"$run"     --vocab-threshold 5 --embed-size 256 --hidden-size 512 --num-layers 1     --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999     --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20     --scheduler plateau --semantic-temp 10.0     --wandb --wandb-entity learning6 --wandb-project Clean     --run-name "$run" "$@"
}

run_attn_coco() {
  local run="$1"; shift
  if [[ ! -f data/coco2017/captions.txt || ! -d data/coco2017/Images ]]; then
    echo "[skip] COCO2017 is not prepared yet. Run scripts/prepare_coco2017.py after downloading COCO."
    return 0
  fi
  MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1   "$PY" -m src.attention.train     --images-dir data/coco2017/Images     --captions-csv data/coco2017/captions.txt     --vocab-path data/coco2017/vocab.pkl     --checkpoints-dir checkpoints_attention/"$run"     --vocab-threshold 5 --embed-size 256 --hidden-size 512 --attention-dim 256     --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999     --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20     --scheduler plateau --ds-lambda 1.0 --label-smoothing 0.0 --semantic-temp 10.0     --wandb --wandb-entity learning6 --wandb-project Clean     --run-name "$run" "$@"
}

run_coco() {
  run_base_coco baseline-coco2017-resnet50-scratch-ce-lr1e3-20ep
  run_attn_coco attention-coco2017-resnet50-scratch-ce-lr1e3-20ep
}

run_all() {
  run_backbone
  run_embeddings
  run_loss
  run_dataset
  run_learning_rate
  run_implementation_first
}

run_coco_template() {
  cat <<'EOF'
Prepare data/coco2017/Images, data/coco2017/captions.txt, and data/coco2017/vocab.pkl first, then use:

MPLCONFIGDIR="$ROOT/.mplconfig" PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" -m src.baseline.train   --images-dir data/coco2017/Images   --captions-csv data/coco2017/captions.txt   --vocab-path data/coco2017/vocab.pkl   --checkpoints-dir checkpoints/baseline-coco2017-resnet50-scratch-ce-lr1e3-20ep   --vocab-threshold 5 --embed-size 256 --hidden-size 512 --num-layers 1   --dropout 0.5 --backbone resnet50 --epochs 20 --patience 999   --batch-size 32 --num-workers 2 --lr 0.001 --log-step 20   --wandb --wandb-entity learning6 --wandb-project Clean   --run-name baseline-coco2017-resnet50-scratch-ce-lr1e3-20ep
EOF
}

usage() {
  cat <<'EOF'
Usage: scripts/run_clean_ablations.sh <group>

Groups:
  all
  coco
  backbone
  embeddings
  loss
  dataset
  learning-rate
  implementation-first
  coco-template
EOF
}

case "${1:-}" in
  all) run_all ;;
  coco) run_coco ;;
  backbone) run_backbone ;;
  embeddings) run_embeddings ;;
  loss) run_loss ;;
  dataset) run_dataset ;;
  learning-rate) run_learning_rate ;;
  implementation-first) run_implementation_first ;;
  coco-template) run_coco_template ;;
  *) usage; exit 1 ;;
esac
