ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 src/open_r1/rag_grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_lamp5.yaml
