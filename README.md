# botcalltest

docker run -u root -itd --privileged=true \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
-v /etc/localtime:/etc/localtime  \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /var/log/npu/:/usr/slog \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /data/model:/data/model \
-v /home/:/home/ \
--net=host \
--name llm_mly \
afb81789b460 \
/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=4,5

python3 -m vllm.entrypoints.openai.api_server --model /data/model/Qwen2.5-3B-Instruct \
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


 curl -X POST http://127.0.0.1:9000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "/data/model/Qwen2.5-3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "请介绍一下你自己"
        }
    ],
    "max_tokens": 100,
    "top_k": -1,
    "top_p": 1,
    "temperature": 0,
    "ignore_eos": false,
    "stream": false
}'


POST /v1/chat/completions
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "请给出用户信息 JSON"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"}
      },
      "required": ["name", "age"]
    }
  },
  "guided_decoding_backend": "xgrammar"
}

https://www.doubao.com/thread/w0b67552341565421



curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-instruct",
    "prompt": "生成一个人的信息JSON",
    "max_tokens": 100,
    "temperature": 0,
    "grammar": {
      "type": "json_schema",
      "value": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}}}"
    }
  }'

