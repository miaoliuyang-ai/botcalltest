import os
import json
import asyncio
import uuid
import yaml
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import time
from tts_module import TTSModule
import sys
sys.path.append("/mnt/sfs_turbo/botcall")
from asr_module import ASRModule
from llm_module import LLMModule

# 加载配置
with open("/mnt/sfs_turbo/botcall/config.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="AI外呼助手调度中心")

# 初始化模块
asr = ASRModule(config['asr'])
llm = LLMModule(config['llm'])
tts = TTSModule(config['tts'])

class ConnectionState:
    def __init__(self):
        self.asr_cache = {}
        self.audio_buffer = np.array([], dtype=np.float32)
        self.voice_type = "温柔女"  # 默认音色
        self.timers = {
            "receive_asr_start": None,      # 收到第一段音频
            "receive_asr_end": None,        # 收到最后一段音频
            "audio_chunk_buffer":[],
            "asr_chunk":[],
            "asr_finish": None,     # ASR推理完成
            "llm_first": None,      # LLM首句生成
            "llm_finish": None,     # LLM生成完成
            "tts_first": None,      # TTS首段生成
            "tts_finish": None,     # TTS生成完成
        }
        self.recognized_text = ""  # 存储完整识别的文本

@app.websocket("/ws/call")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = ConnectionState()
    client_id = str(uuid.uuid4())
    
    try:
        # 接收音色参数
        init_data = await websocket.receive_text()
        init_params = json.loads(init_data)
        state.voice_type = init_params.get("voice_type", state.voice_type)
        
        # 处理音频流
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                # 记录ASR开始时间
                if state.timers["receive_asr_start"] is None:
                    state.timers["receive_asr_start"] = time.time()
                
                # 记录接受ASR结束时间（每次更新）
                state.timers["receive_asr_end"] = time.time()

                new_audio = asr.process_audio(data["bytes"])
                state.audio_buffer = np.concatenate([state.audio_buffer, new_audio])
                asr_start = time.time()
                state.timers["audio_chunk_buffer"].append(asr_start - state.timers["receive_asr_end"])
                # 执行ASR识别
                text, state.audio_buffer = asr.recognize(
                    state.audio_buffer,
                    state.asr_cache
                )
                state.timers["asr_chunk"].append(time.time() - asr_start)
                if text:
                    # 累积识别的文本
                    state.recognized_text += text
            elif data["type"] == "websocket.disconnect":
                break
            elif "text" in data and data["text"].upper() == "EOS":  # 结束信号
                # 处理剩余音频
                if len(state.audio_buffer) > 0:
                    final_text, _ = asr.recognize(
                        state.audio_buffer,
                        state.asr_cache,
                        is_final=True
                    )
                    if final_text:
                        state.recognized_text += final_text
                
                # 记录ASR结束时间
                state.timers["asr_finish"] = time.time()
                
                # 调用LLM生成回复
                async for sentence in llm.generate_stream(state.recognized_text):
                    # 记录首个LLM响应时间
                    if state.timers["llm_first"] is None:
                        state.timers["llm_first"] = time.time()
                
                    # 记录LLM完成时间
                    state.timers["llm_finish"] = time.time()

                    # 调用TTS合成语音
                    tts_gen = tts.synthesize_stream(sentence, state.voice_type)
                
                    try:
                        async for audio_chunk in tts_gen:
                            # 记录首个TTS响应时间
                            if state.timers["tts_first"] is None:
                                state.timers["tts_first"] = time.time()
                        
                            await websocket.send_bytes(audio_chunk)
                    except Exception as e:
                        print(f"tts 错误: {str(e)}")
                
                # 记录TTS完成时间
                state.timers["tts_finish"] = time.time()

                # 发送时间统计给客户端
                timers_str = state.timers
                await websocket.send_text(json.dumps({"timers": timers_str}))
                break        
    except WebSocketDisconnect:
        print(f"客户端断开: {client_id}")
    except Exception as e:
        print(f"处理错误: {str(e)}")
    finally:
        await websocket.close()

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok", "timestamp": time.time()})

if __name__ == "__main__":
    server_config = config['server']
    uvicorn.run(
        app,
        host=server_config['host'],
        port=server_config['port'],
        log_level="info"
    )