# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Optional
from pydantic import BaseModel
import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from hyperpyyaml import load_hyperpyyaml
import copy
import time

model_path = '/home/lzh/workspace/botcall/tts/cosyvoice2/weight/CosyVoice2-0.5B'
model_input = torch.load(f"{model_path}/voice_4.pt")
voice_templates = {
    "温柔女": copy.deepcopy(model_input['温柔女']),
}
warmup_txt = '收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
class TTSRequest(BaseModel):
    text: str
    voice_type: Optional[str] = "温柔女"

app = FastAPI()

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    cosyvoice = CosyVoice2(model_path, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm() #删除推理过程中的weight_norm
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend)
    
    with torch.no_grad():
        print('warm up start')
        for _ in range(args.warm_up_times):
            next(cosyvoice.inference_instruct2_fast(warmup_txt, pre_model_input=model_input['温柔女'], stream=args.stream))
        print('warm up end')

    @app.post("/tts/fast/stream")
    async def http_tts_stream(request: TTSRequest):
        # 检查参数
        if not request.text or not request.voice_type:
            raise HTTPException(status_code=400, detail="Missing text or voice_type")

        # 获取预加载模型输入
        try:
            pre_model_input = model_input[request.voice_type]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unsupported voice_type: {request.voice_type}")

        # 定义生成器函数用于 StreamingResponse
        def audio_stream():
            start_time = time.time()
            with torch.no_grad():
                for i, chunk in enumerate(cosyvoice.inference_instruct2_fast(
                    request.text,
                    pre_model_input=pre_model_input,
                    stream=True
                )):
                    audio_data = chunk['tts_speech'].detach().cpu().numpy()
                    t = time.time() - start_time
                    if i == 0:
                        print(f"首段时间: {t:.3f} 秒")
                    
                    yield audio_data.tobytes()
                    del chunk['tts_speech']
                    del audio_data
            torch.npu.empty_cache()

        # 设置响应头为 raw float32 音频流
        headers = {
            'Content-Type': 'audio/x-raw',
            'Content-Encoding': 'identity'
        }

        return StreamingResponse(audio_stream(), media_type='audio/x-raw', headers=headers)
    uvicorn.run(app, host="0.0.0.0", port=8011)
    

   