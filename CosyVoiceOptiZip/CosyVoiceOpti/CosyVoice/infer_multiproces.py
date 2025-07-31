import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import copy
import time
from multiprocessing import Process


# 定义请求数据模型（Flask 中直接使用 request.get_json() 解析）
class TTSRequest:
    def __init__(self, text, voice_type):
        self.text = text
        self.voice_type = voice_type

MODEL_PATH = '/home/lzh/workspace/botcall/tts/cosyvoice2/weight/CosyVoice2-0.5B'
WARMUP_TIMES = 2
STREAM = True

# 模型初始化和预热函数（与原代码一致）
def init_model():
    model_input = torch.load("/home/lzh/workspace/botcall/tts/cosyvoice2/weight/CosyVoice2-0.5B/spk2info.pt")
    voice_templates = {
        "温柔女": copy.deepcopy(model_input['温柔女']),
    }
    cosyvoice = CosyVoice2(MODEL_PATH, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    cosyvoice.model.hift.remove_weight_norm()
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(
        cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend
    )

    warmup_txt = '收到好友'

    with torch.no_grad():
        print('Warm up start')
        for _ in range(WARMUP_TIMES):
            next(cosyvoice.inference_instruct2_fast(warmup_txt, pre_model_input=copy.deepcopy(voice_templates['温柔女']), stream=STREAM))
        print('Warm up end')
        start_time = time.time()
        for i, chunk in enumerate(cosyvoice.inference_instruct2_fast(
                "测试"*10,
                pre_model_input=copy.deepcopy(voice_templates['温柔女']),
                stream=STREAM
            )):
                audio_data = chunk['tts_speech'].detach().cpu().numpy()
                t = time.time() - start_time
                if i == 0:
                    print(f"首段时间: {t:.3f} 秒")
                # yield audio_data.tobytes()
                

if __name__ == "__main__":
    torch_npu.npu.set_compile_mode(jit_compile=False)

    processes = []
    for index in range(2):
        p = Process(target=init_model,args=())
        processes.append(p)
        p.start()
        print(f"process {index} start")
    for p in processes:
        p.join()
    print('all')
