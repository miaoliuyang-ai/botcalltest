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


  https://help.aliyun.com/zh/model-studio/json-mode#3a11734087e4d



from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)


completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)


from pydantic import BaseModel
from enum import Enum


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"




class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType




json_schema = CarDescription.model_json_schema()


completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    extra_body={"guided_json": {
  "title": "CarDescription",
  "type": "object",
  "properties": {
    "brand": { "type": "string" },
    "model": { "type": "string" },
    "car_type": {
      "type": "string",
      "enum": ["sedan", "SUV", "Truck", "Coupe"]
    }
  },
  "required": ["brand", "model", "car_type"]
}},
)
print(completion.choices[0].message.content)



{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful expert math tutor."},
    {"role": "user", "content": "Solve 8x + 31 = 2."}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "MathResponse",
      "schema": {
        "type": "object",
        "properties": {
          "steps": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "explanation": {"type": "string"},
                "output": {"type": "string"}
              },
              "required": ["explanation", "output"]
            }
          },
          "final_answer": {"type": "string"}
        },
        "required": ["steps", "final_answer"]
      }
    }
  },
  "guided_decoding_backend": "outlines"
}



{
  "model": "meta-llama/Llama-3.1-8B-Instruct",  // 模型名称
  "messages": [
    {
      "role": "system",
      "content": "你是一个数学解题助手，必须返回 JSON 格式结果，包含以下字段：\n- steps：数组，每个元素是包含 explanation（步骤说明）和 output（步骤结果）的对象\n- final_answer：字符串，最终答案"
    },
    {
      "role": "user",
      "content": "Solve 2x + 5 = 15"
    }
  ],
  "response_format": {
    "type": "json_object"  // 指定输出为 JSON 对象
  },
  "guided_decoding_backend": "outlines"  // 可选，增强 JSON 格式准确性
}


url: http://192.168.1.174:19001/v1/chat/completions
入参：
{
    "model": "/data/model/Qwen/Qwen2.5/Qwen2.5-3B-Instruct",
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
}
