# botcalltest


python -m vllm.entrypoints.openai.api_server --model /data/model/Qwen2.5-3B-Instruct \
--max-num-seqs=256 \
--max-model-len=14096 \
--max-num-batched-tokens=14096 \
--tensor-parallel-size=2 \
--block-size=128 \
--host=0.0.0.0 \
--port=9001 \
--gpu-memory-utilization=0.9 \
--trust-remote-code \
--enforce-eager \
