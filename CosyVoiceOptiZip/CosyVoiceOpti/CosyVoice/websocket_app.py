# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse,HTMLResponse
import io
import uvicorn
from typing import Optional
from pydantic import BaseModel
import json

import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from hyperpyyaml import load_hyperpyyaml
import copy
import numpy as np
import time
import socket
import asyncio

app = FastAPI()

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="CosyVoice infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--warm_up_times', default=2, type=int, help='warm up times')
    parser.add_argument('--infer_count', default=20, type=int, help='infer loop count')
    parser.add_argument('--stream', action="store_true", help='stream infer')
    args = parser.parse_args()

    cosyvoice = CosyVoice2(args.model_path, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm() #删除推理过程中的weight_norm
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend)

    target_sample = 16000
    # 输入数据加载
    warmup_txt = '收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    model_input = torch.load("/home/lzh/workspace/botcall/tts/cosyvoice2/weight/CosyVoice2-0.5B/fast2.pt")

    with torch.no_grad():
        print('warm up start')
        for _ in range(args.warm_up_times):
            next(cosyvoice.inference_instruct2_fast(warmup_txt, pre_model_input=model_input['温柔女'], stream=args.stream))
        print('warm up end')
    
    class TTSRequest(BaseModel):
        text: str
        voice_type: Optional[str] = "温柔的中文客服女生声音"

    @app.websocket("/ws/tts/fast")
    async def websocket_tts(websocket: WebSocket):
        await websocket.accept()
        websocket._receive_max_size = 10 * 1024 * 1024  # 10MB
        try:
            # 接收客户端发送的请求参数
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # 验证必要参数
            if "text" not in request or "voice_type" not in request:
                await websocket.send_json({"error": "Missing text or voice_type"})
                return
            
            start_time = time.time()
            with torch.no_grad():
                pre_model_input = model_input[request["voice_type"]]
                for i, j in enumerate(cosyvoice.inference_instruct2_fast(
                    request["text"], pre_model_input=pre_model_input,
                    stream=args.stream
                )):
                    t = time.time() - start_time
                    if i==0:
                        print(f"首段时间:{t:0.3f}秒")
                    audio_data = j['tts_speech']
                    segment = {
                        "audio_data": audio_data.cpu().numpy().tolist()
                    }
                    # 实时发送每个音频片段
                    await websocket.send_json(segment)
                    
            # 发送结束标记
            await websocket.send_json({"status": "complete"})
            
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()

    @app.websocket("/ws/tts/fast/byte")
    async def websocket_tts(websocket: WebSocket):
        await websocket.accept()
        try:
            # 接收客户端发送的请求参数
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # 验证必要参数
            if "text" not in request or "voice_type" not in request:
                await websocket.send_json({"error": "Missing text or voice_type"})
                return
            # await websocket.send_bytes(b"\x00"*1024)  # 1KB空数据
            # await asyncio.sleep(0.01) 
            start_time = time.time()
            with torch.no_grad():
                pre_model_input = model_input[request["voice_type"]]
                for i,chunk in enumerate(cosyvoice.inference_instruct2_fast(
                    request["text"], 
                    pre_model_input=pre_model_input,
                    stream=args.stream
                )):
                    audio_data = chunk['tts_speech'].cpu().numpy()
                    t = time.time() - start_time
                    if i==0:
                        print(f"首段时间:{t:0.3f}秒")
                    await websocket.send_bytes(audio_data.tobytes())
                    t = time.time() - start_time
                    print(f"发送时间:{t:0.3f}秒")
                    
            # 发送结束标记
            await websocket.send_json({"status": "complete"})
            
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()
    uvicorn.run(app, host="0.0.0.0", port=8012)