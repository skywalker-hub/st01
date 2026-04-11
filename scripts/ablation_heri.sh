#!/bin/bash

COMMON_ARGS=(
    --dataset "math500"
    --model_name "./models/Qwen/Qwen2.5-3B-Instruct"
    --max_topk 10
    --max_generated_tokens 4096
    --temperature 0.6
    --top_p 0.95
    --top_k 30
    --min_p 0.0
    --after_thinking_temperature 0.6
    --after_thinking_top_p 0.95
    --after_thinking_top_k 30
    --after_thinking_min_p 0.0
    --early_stopping_entropy_threshold 0.1
    --early_stopping_length_threshold 64
    --mem_fraction_static 0.8
    --start_idx 0
    --end_idx 100000
    --num_gpus 1
    --num_samples 1
    --enable_soft_thinking
)

# ============================================
# Baseline: no soft thinking
# ============================================
echo "===== Baseline (no soft thinking) ====="
python run_sglang_softthinking.py \
    --dataset "math500" \
    --model_name "./models/Qwen/Qwen2.5-3B-Instruct" \
    --max_generated_tokens 4096 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 100000 \
    --num_gpus 1 \
    --num_samples 1

# ============================================
# Ablation 1: sweep gamma (fix theta=0.2)
# ============================================
for GAMMA in 0 0.5 1.0 2.0 3.0 5.0; do
    echo "===== gamma=${GAMMA}, theta=0.2 ====="
    HERI_GAMMA=${GAMMA} HERI_THETA=0.2 \
        python run_sglang_softthinking.py "${COMMON_ARGS[@]}"
done

# ============================================
# Ablation 2: sweep theta (fix gamma=3.0)
# ============================================
for THETA in 0.0 0.1 0.2 0.3 0.4 1.0; do
    echo "===== gamma=3.0, theta=${THETA} ====="
    HERI_GAMMA=3.0 HERI_THETA=${THETA} \
        python run_sglang_softthinking.py "${COMMON_ARGS[@]}"
done
