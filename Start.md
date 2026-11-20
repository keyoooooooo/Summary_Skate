conda activate LLM

cd /data/coding

vllm serve ./models/Qwen3.8-27B-FP8 \
    --served-model-name Qwen3.8-27B-FP8 \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90