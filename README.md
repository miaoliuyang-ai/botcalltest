
python -m vllm.entrypoints.openai.api_server 
  --model /model/Qwen2.5-VL-72B-Instruct 
  --max-num-seqs=256 
  --max-model-len=12288 
  --max-num-batched-tokens=12288 
  --tensor-parallel-size=4 
  --block-size=128 
  --host=0.0.0.0 
  --port=19080 
  --gpu-memory-utilization=0.95 
  --trust-remote-code 
  --enforce-eager 
  --api-key xiVrwO39gXC3LRT876A23eLeIRg2bFLHGOtcjghEoPplw3SupsYkg3q9Jm5Fnsd04hGQkeL3XwM6zx-0WX3RsA

docker login -u cn-southwest-2@HST3URII19WGPSLZD01G -p 7d2ab75df4816d39827b96a77bbbb949ce8b15ae6cf8fd84385a7794b1359557 swr.cn-southwest-2.myhuaweicloud.com
docker pull swr.cn-southwest-2.myhuaweicloud.com/botcall/hwbotcall:1.1
