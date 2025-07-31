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
from collections import deque
import logging
import sys
import queue
import atexit

# 添加botcall路径
sys.path.append("/mnt/sfs_turbo/botcall")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log")
    ]
)
logger = logging.getLogger("AI-Call-Server")

# 加载配置
with open("/mnt/sfs_turbo/botcall/config.yaml") as f:
    config = yaml.safe_load(f)

# 从配置文件读取工作进程数量
NUM_WORKERS = config.get('server', {}).get('workers', 2)  # 每卡默认两进程
CHUNK_TIMEOUT = 30

# 加载模型
from asr_module import ASRModule
from llm_module import LLMModule
from tts_module import TTSModule

asr = ASRModule(config['asr'])
llm = LLMModule(config['llm'])
tts = TTSModule(config['tts'])

app = FastAPI(title="AI外呼助手调度中心")

class ConnectionState:
    def __init__(self):
        self.asr_cache = {}
        self.audio_buffer = np.array([], dtype=np.float32)
        self.voice_type = "温柔女"  # 默认音色
        self.timers = {
            "receive_asr_start": None,      # 收到第一段音频
            "receive_asr_end": None,        # 收到最后一段音频
            "audio_chunk_buffer": deque(maxlen=100),
            "asr_chunk": deque(maxlen=100),
            "asr_finish": None,     # ASR推理完成
            "llm_first": None,       # LLM首句生成
            "llm_finish": None,      # LLM生成完成
            "tts_first": None,       # TTS首段生成
            "tts_finish": None,      # TTS生成完成
        }
        self.recognized_text = ""    # 存储完整识别的文本
        self.data_queue = asyncio.Queue(maxsize=10)  # 处理队列
        self.processing_task = None
        self.eos_received = asyncio.Event()          # 结束信号

def proc_tag():
    return f"[PID:{os.getpid()}]"

@app.websocket("/ws/call")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = ConnectionState()
    client_id = str(uuid.uuid4())
    logger.info(f"{proc_tag()} 客户端连接: {client_id}")
    
    # 启动后台处理任务
    state.processing_task = asyncio.create_task(audio_processing_task(state, websocket))
    
    try:
        # 接收音色参数
        init_data = await websocket.receive_text()
        init_params = json.loads(init_data)
        state.voice_type = init_params.get("voice_type", state.voice_type)
        logger.info(f"{proc_tag()} 客户端 {client_id} 设置音色: {state.voice_type}")
        
        # 处理音频流
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                # 记录ASR开始时间
                if state.timers["receive_asr_start"] is None:
                    state.timers["receive_asr_start"] = time.time()
                
                # 记录接受ASR结束时间（每次更新）
                state.timers["receive_asr_end"] = time.time()
                
                # 将数据放入处理队列
                await state.data_queue.put(("audio", data["bytes"]))
                
            elif data["type"] == "websocket.disconnect":
                logger.info(f"{proc_tag()} 客户端主动断开: {client_id}")
                break
                
            elif "text" in data and data["text"].upper() == "EOS":  # 结束信号
                logger.info(f"{proc_tag()} 收到EOS信号: {client_id}")
                # 通知处理任务结束
                await state.data_queue.put(("eos", None))
                # 等待处理完成
                await state.eos_received.wait()
                
                # 发送时间统计给客户端
                # 转换deque为列表用于JSON序列化
                state.timers["audio_chunk_buffer"] = list(state.timers["audio_chunk_buffer"])
                state.timers["asr_chunk"] = list(state.timers["asr_chunk"])
                
                await websocket.send_text(json.dumps({"timers": state.timers}))
                logger.info(f"{proc_tag()} 发送时间统计: {client_id}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"{proc_tag()} 客户端断开: {client_id}")
    except Exception as e:
        logger.error(f"{proc_tag()} 处理错误: {str(e)}")
    finally:
        # 清理资源
        if state.processing_task:
            state.processing_task.cancel()
            try:
                await state.processing_task
            except asyncio.CancelledError:
                pass
        await websocket.close()
        logger.info(f"{proc_tag()} 连接关闭: {client_id}")

async def audio_processing_task(state: ConnectionState, websocket: WebSocket):
    """后台音频处理任务"""
    try:
        while True:
            try:
                item_type, data = await asyncio.wait_for(state.data_queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                if len(state.audio_buffer) > 0:
                    logger.info(f"{proc_tag()} [ASR] 超时，处理剩余音频缓冲区")
                    await process_audio_buffer(state, websocket)
                continue
                
            if item_type == "audio":
                start_time = time.time()
                new_audio = np.frombuffer(data, dtype=np.float32)
                state.audio_buffer = np.concatenate([state.audio_buffer, new_audio])
                state.timers["audio_chunk_buffer"].append(time.time() - start_time)
                logger.info(f"{proc_tag()} [ASR] 收到音频数据，当前缓冲区长度={len(state.audio_buffer)}")
                if len(state.audio_buffer) >= 16000 * 0.6:
                    logger.info(f"{proc_tag()} [ASR] 缓冲区达到阈值，触发处理")
                    await process_audio_buffer(state, websocket)
                    
            elif item_type == "eos":
                if len(state.audio_buffer) > 0:
                    logger.info(f"{proc_tag()} [ASR] 收到EOS，处理剩余音频缓冲区")
                    await process_audio_buffer(state, websocket, is_final=True)
                else:
                    # 如果没有音频数据，直接处理LLM和TTS（流式推送）
                    await process_llm_and_tts(state, websocket)
                state.eos_received.set()
                break
                
            state.data_queue.task_done()
            
    except asyncio.CancelledError:
        logger.info(f"{proc_tag()} 处理任务被取消")
    except Exception as e:
        logger.error(f"{proc_tag()} 处理任务错误: {str(e)}")

async def process_audio_buffer(state: ConnectionState, websocket: WebSocket, is_final=False):
    """处理音频缓冲区"""
    if len(state.audio_buffer) == 0:
        logger.info(f"{proc_tag()} [ASR] 跳过空缓冲区处理")
        return
        
    logger.info(f"{proc_tag()} [ASR] 开始识别，is_final={is_final}, 缓冲区长度={len(state.audio_buffer)}")
    
    start_time = time.time()
    text, remaining_buffer = asr.recognize(state.audio_buffer, state.asr_cache, is_final)
    duration = time.time() - start_time
    logger.info(f"{proc_tag()} [ASR] 任务完成，耗时={duration:.3f}s，识别文本：'{text}', 长度: {len(text)}")
    state.timers["asr_chunk"].append(duration)
    
    if text:
        state.recognized_text += text
        logger.info(f"{proc_tag()} [ASR] 累计识别文本：'{state.recognized_text}'")
    state.audio_buffer = remaining_buffer
    
    if is_final:
        state.timers["asr_finish"] = time.time()
        logger.info(f"{proc_tag()} [ASR] 完成，累计识别文本：{state.recognized_text}")
        await process_llm_and_tts(state, websocket)

async def process_llm_and_tts(state: ConnectionState, websocket: WebSocket):
    """处理LLM和TTS"""
    logger.info(f"{proc_tag()} [LLM/TTS] 开始处理LLM和TTS")
    
    first_llm = True
    tts_first_set = False
    async for sentence in llm.generate_stream(state.recognized_text):
        logger.info(f"{proc_tag()} [LLM] 生成句子：{sentence}")
        if first_llm:
            state.timers["llm_first"] = time.time()
            logger.info(f"{proc_tag()} [LLM] 首句生成，llm_first={state.timers['llm_first']}")
            first_llm = False
        async for audio_chunk in tts.synthesize_stream(sentence, state.voice_type):
            logger.info(f"{proc_tag()} [TTS] 收到音频块，长度={len(audio_chunk) if audio_chunk else None}")
            if not tts_first_set:
                state.timers["tts_first"] = time.time()
                logger.info(f"{proc_tag()} [TTS] 首音频块生成，tts_first={state.timers['tts_first']}")
                tts_first_set = True
            await websocket.send_bytes(audio_chunk)
    state.timers["llm_finish"] = time.time()
    state.timers["tts_finish"] = time.time()
    logger.info(f"{proc_tag()} [TTS] 完成，tts_finish={state.timers['tts_finish']}")


@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "ok", 
        "timestamp": time.time(),
        "modules": {
            "asr": "active",
            "llm": "active",
            "tts": "active"
        }
    })

if __name__ == "__main__":
    # 注册退出处理
    atexit.register(shutdown_workers)
    
    server_config = config['server']
    uvicorn.run(
        app,
        host=server_config['host'],
        port=server_config['port'],
        log_level="info",
        timeout_keep_alive=30,
        ws_ping_interval=10,
        ws_ping_timeout=30
    )