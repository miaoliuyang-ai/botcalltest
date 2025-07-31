import os
import copy
import torch
import asyncio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from typing import Dict, Any, AsyncGenerator
from cosyvoice.cli.cosyvoice import CosyVoice2

class TTSModule:
    def __init__(self, tts_config: dict):
        self.tts_config = tts_config
        self.model = None
        self.voice_types = torch.load(os.path.join(self.tts_config['model_path'],self.tts_config['voice_template_file']))
        # 新增：首句分割配置
        self.first_tts_short_split = tts_config.get('first_tts_short_split', False)
        self.first_tts_short_len = tts_config.get('first_tts_short_len', 10)
        self._init_model()
    
    def _init_model(self):
        """初始化TTS模型"""
        print("开始加载TTS模型...")
        torch_npu.npu.set_compile_mode(jit_compile=False)
        self.model = CosyVoice2(self.tts_config['model_path'], load_om=True, fp16=True)
        self.model.model.llm.eval()
        self.model.model.llm.llm.model.model.half()
        
        # 配置NPU优化
        self.model.model.hift.remove_weight_norm()
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=config)
        self.model.model.hift.decode = torch.compile(
            self.model.model.hift.decode,
            dynamic=True,
            fullgraph=True,
            backend=npu_backend
        )

        with torch.no_grad():
            print('warm up start')
            for _ in range(self.tts_config['warm_up_times']):
                next(self.model.inference_instruct2_fast(self.tts_config['warmup_text'], pre_model_input=self.load_voice("温柔女"), stream=True))
            for chunk in self.model.inference_instruct2_fast(
                self.tts_config['warmup_text']*5,
                pre_model_input=self.load_voice("温柔女"),
                stream=True
            ):
                continue
            print('warm up end')
    
    def load_voice(self, voice_type: str) -> Dict[str, Any]:
        """加载指定音色的预输入"""
        if voice_type not in self.voice_types:
            voice_type = self.voice_types[self.tts_config['default_voice']] 
        return copy.deepcopy(self.voice_types[voice_type])
    
    async def synthesize_stream(
        self, 
        text: str, 
        voice_type: str
    ) -> AsyncGenerator[bytes, None]:
        """流式语音合成"""
        pre_input = self.load_voice(voice_type)
        
        with torch.no_grad():
            # 创建同步生成器
            sync_gen = self.model.inference_instruct2_fast(
                text, 
                pre_model_input=pre_input,
                stream=True,
                # 新增参数传递
                first_tts_short_split=self.first_tts_short_split,
                first_tts_short_len=self.first_tts_short_len
            )
            
            # 在异步环境中迭代同步生成器
            try:
                for chunk in sync_gen:
                    audio_data = chunk['tts_speech'].cpu().numpy()
                    yield audio_data.tobytes()
                    
                    # 添加小延迟避免阻塞事件循环
                    await asyncio.sleep(0.001)
            except Exception as e:
                print(f"TTS合成过程中出错: {e}")
                raise
            # async for chunk in self.model.inference_instruct2_fast(
            #     text, 
            #     pre_model_input=pre_input,
            #     stream=True
            # ):
            #     audio_data = chunk['tts_speech'].cpu().numpy()
            #     yield audio_data.tobytes()