# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import os

MODEL_PATH = ''

PROMPTS = {'温柔女':'温柔女生',
           '伤心女':'悲伤女生',
           '阳光女':'开朗女生',
           '合成男':'男生声音'}
WAV_PATH = '/home/lzh/workspace/botcall/data/tts_voice/'

VOICE_PT_PATH = ''

def get_wav_files(directory):
    wav_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            wav_files.append(os.path.splitext(filename)[0])
    return wav_files

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    cosyvoice = CosyVoice2(MODEL_PATH, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm() #删除推理过程中的weight_norm
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend)

    d = {}
    with torch.no_grad():
        for wav_name,prompt in PROMPTS.items():
            wav_file = os.path.join(WAV_PATH,f'{wav_name}.wav')
            model_input = cosyvoice.inference_instruct2_fast_tool(prompt,load_wav(wav_file,16000))
            print(model_input)
            d[wav_name] = model_input

    torch.save(d,VOICE_PT_PATH)