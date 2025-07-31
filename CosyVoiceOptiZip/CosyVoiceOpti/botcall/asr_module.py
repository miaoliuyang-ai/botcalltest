import numpy as np
import torch
from funasr import AutoModel
from typing import Dict, Any

class ASRModule:
    def __init__(self, config: dict):
        self.config = config
        print("开始加载ASR模型...")
        self.asr_model = AutoModel(
            model=config['model_path'],
            model_revision=config['model_revision'],
            device=config['device'],
            disable_update=True
        )
        print("ASR模型加载完成")  # 添加调试输出
        self.chunk_stride = config['chunk_size'][1] * 960
        
    def process_audio(self, audio_data: bytes) -> np.ndarray:
        """转换音频数据为numpy数组"""
        try:
            return np.frombuffer(audio_data, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"音频处理错误: {str(e)}")
    
    def recognize(
        self,
        audio_buffer: np.ndarray,
        cache: Dict,
        is_final: bool = False
    ) -> str:
        """执行语音识别"""
        total_samples = len(audio_buffer)
        total_chunks = total_samples // self.chunk_stride
        results = []
        
        for i in range(total_chunks):
            start = i * self.chunk_stride
            end = (i + 1) * self.chunk_stride
            speech_chunk = audio_buffer[start:end]
            
            res = self.asr_model.generate(
                input=speech_chunk,
                cache=cache,
                is_final=is_final and (i == total_chunks - 1),
                chunk_size=self.config['chunk_size'],
                encoder_chunk_look_back=self.config['encoder_chunk_look_back'],
                decoder_chunk_look_back=self.config['decoder_chunk_look_back']
            )
            
            if res and "text" in res[0]:
                results.append(res[0]["text"])
        
        # 处理剩余音频
        remaining = total_samples % self.chunk_stride
        new_buffer = audio_buffer[-remaining:] if remaining > 0 else np.array([], dtype=np.float32)
        
        return "".join(results), new_buffer