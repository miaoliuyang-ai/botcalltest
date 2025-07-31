import asyncio
import websockets
import json
import soundfile as sf
import librosa
import numpy as np
import time
from datetime import datetime

async def test_call():
    # 时间记录变量
    timings = {
        "send_asr_start": None,
        "send_end_chunk":None,
        "send_asr_end": None,
        "recv_tts_first": None,
        "recv_tts_end": None,
        "server_timers": None
    }
    # 1. 准备测试音频l
    audio_path = "/mnt/sfs_turbo/1.wav"
    y, sr = librosa.load(audio_path, sr=16000)
    y = y.astype(np.float32)
    
    # 2. 连接服务
    async with websockets.connect("ws://192.168.0.10:8000/ws/call") as websocket:
        # 发送音色参数
        await websocket.send(json.dumps({
            "voice_type": "温柔女"
        }))
        
        # 3. 模拟音频流发送
        chunk_size = int(16000*0.6)  # 600ms的音频

         # 记录ASR开始发送时间
        timings["send_asr_start"] = time.time()
        
        # 分块发送音频
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i+chunk_size].tobytes()
            await websocket.send(chunk)
            timings["send_end_chunk"] = time.time()
            await asyncio.sleep(0.05)  # 模拟实时
        # 记录ASR发送完成时间
        timings["send_asr_end"] = time.time()
        # 4. 发送结束信号
        await websocket.send("EOS")
        print("已发送EOS信号，等待响应...")
        
        # 4. 接收TTS响应
        # output_audio = bytearray()
        output_audio = []
        while True:
            try:
                data = await websocket.recv()
                if isinstance(data, bytes):
                    d = np.frombuffer(data, dtype=np.float32)
                    output_audio.append(d)
                    # 记录首个TTS包时间
                    if timings["recv_tts_first"] is None:
                        timings["recv_tts_first"] = time.time()
                    # 更新最后一个TTS数据包到达时间
                    timings["recv_tts_end"] = time.time()
                else:
                    msg = json.loads(data)
                    if "timers" in msg:
                        timings["server_timers"] = msg["timers"]
            except websockets.exceptions.ConnectionClosed:
                print("websockets.exceptions.ConnectionClosed")
                break
        
        # 5. 保存输出音频
        if output_audio:
            np_audio = np.concatenate(output_audio)
            sf.write("output.wav", np_audio, 24000)
        
         # 7. 计算时间差
        if timings["server_timers"]:
            # 将服务端时间字符串转换为时间戳
            server_ts = timings["server_timers"]
            
            # 计算时间差
            time_diffs = {
                "asr_start_diff": server_ts["receive_asr_start"] - timings["send_asr_start"],
                "asr_end_diff": server_ts["receive_asr_end"] - timings["send_asr_end"],
                "asr_end_diff_2": server_ts["receive_asr_end"] - timings["send_end_chunk"],
                "asr_end_finish" : server_ts["asr_finish"] - server_ts["receive_asr_end"],
                "asr_finish_diff": server_ts["asr_finish"] - timings["send_asr_end"],
                "asr_end_finish_diff": server_ts["asr_finish"] - timings["send_end_chunk"],
                "llm_first_diff": server_ts["llm_first"] - server_ts["asr_finish"],
                "llm_finish_diff": server_ts["llm_finish"] - server_ts["asr_finish"],
                "tts_first_chunk_diff": server_ts["tts_first"] - server_ts["llm_first"],
                "tts_clinet_receive_first_chunk_diff": timings["recv_tts_first"] - server_ts["tts_first"],
                "asr_end_chunk_tts_first_chunk_diff": timings["recv_tts_first"] - timings["send_end_chunk"],
                "asr_end_tts_first_chunk_diff": timings["recv_tts_first"] - timings["send_asr_end"],
                "tts_finish_diff": server_ts["tts_finish"] - timings["recv_tts_end"]
            }
        # 8. 打印性能数据
        print(f"\n客户端时间统计:")
        print(f"ASR开始发送: {timings['send_asr_start']}")
        print(f"ASR发送完成: {timings['send_asr_end']}")
        print(f"首个TTS接收: {timings['recv_tts_first']}")
        print(f"所有TTS接收完成: {timings['recv_tts_end']}")
        
        print(f"\n服务端时间统计:")
        for k, v in timings["server_timers"].items():
            print(f"{k}: {v}")
        
        print(f"\n时间差计算(服务端时间 - 客户端时间):")
        for k, v in time_diffs.items():
            print(f"{k}: {v:.4f}s")
        
        # 整体流水线耗时
        if timings["recv_tts_end"] and timings["send_asr_start"]:
            total_time = timings["recv_tts_end"] - timings["send_asr_start"]
            print(f"\n端到端总耗时: {total_time:.4f}s")
        
        

if __name__ == "__main__":
    asyncio.run(test_call())