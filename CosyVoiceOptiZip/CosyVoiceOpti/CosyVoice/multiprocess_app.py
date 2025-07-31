import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
import copy
import time
from multiprocessing import Process, Manager, current_process
from flask import Flask, request, jsonify, Response, stream_with_context
import uuid
import queue
import os
import logging
from logging.handlers import RotatingFileHandler

WARMUP_TIMES = 1
STREAM = True

NUM_WORKERS = 1  
PORT = 8011
CHUNK_TIMEOUT = 30

MODEL_PATH = '/mnt/sfs_turbo/models/tts/CosyVoice2-0.5B'
VOICES = torch.load(os.path.join(MODEL_PATH,'voice_4.pt'))

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                os.path.join(log_dir, 'tts_service.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )

# init log
setup_logging()
logger = logging.getLogger(__name__)

class TTSWorker(Process):
    def __init__(self, task_queue, manager):
        super().__init__()
        self.task_queue = task_queue
        self.manager = manager
        self.model = None
        self.voice_templates = {}
        self.device = None
        self.logger = logging.getLogger(f"{__name__}.TTSWorker")

    def load_model(self):
        # set NPU compile mode
        torch_npu.npu.set_compile_mode(jit_compile=False)
        
        # load voice
        for key,value in VOICES.items():
            self.voice_templates[key] = copy.deepcopy(value)
        
        # init model
        self.logger.info(f"{current_process().name} Loading model...")
        self.cosyvoice = CosyVoice2(MODEL_PATH, load_om=True, fp16=True)
        self.cosyvoice.model.llm.eval()
        self.cosyvoice.model.llm.llm.model.model.half()
        
        # set complier
        self.cosyvoice.model.hift.remove_weight_norm()
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=config)
        self.cosyvoice.model.hift.decode = torch.compile(
            self.cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend
        )
        self.logger.info(f"{current_process().name} Model loaded")

    def warmup(self):
        warmup_txt = '预热文本'
        self.logger.info(f"{current_process().name} Warming up...")
        
        with torch.no_grad():
            for _ in range(WARMUP_TIMES):
                next(self.cosyvoice.inference_instruct2_fast(
                    warmup_txt,
                    pre_model_input=copy.deepcopy(self.voice_templates['温柔女']),
                    stream=STREAM
                ))
            
            # inference test
            start_time = time.time()
            for i, chunk in enumerate(self.cosyvoice.inference_instruct2_fast(
                warmup_txt*5,
                pre_model_input=copy.deepcopy(self.voice_templates['温柔女']),
                stream=STREAM
            )):
                if i == 0:
                    self.logger.info(f"{current_process().name} Warmup First chunk latency: {time.time()-start_time:.3f}s")
        
        self.logger.info(f"{current_process().name} Warmup completed")

    def run(self):
        # Model loading and warm-up phase upon process initialization
        self.load_model()
        self.warmup()
        
        # stream inference
        while True:
            task_data = self.task_queue.get()
            if task_data is None:  # quit
                break
                
            task_id, text, voice_type, result_queue = task_data
            self.logger.info(f"{current_process().name} Processing task {task_id}: {text[:20]}...")
            
            try:
                start_time = time.time()
                chunk_count = 0
                fisrt_chunk_send_time = 0
                
                with torch.no_grad():
                    for i,chunk in enumerate(self.cosyvoice.inference_instruct2_fast(
                        text,
                        pre_model_input=copy.deepcopy(self.voice_templates[voice_type]),
                        stream=STREAM
                    )):
                        if i == 0:
                            fisrt_chunk_time  = time.time() - start_time
                        audio_data = chunk['tts_speech'].detach().cpu().numpy()
                        result_queue.put((task_id, 'data', audio_data.tobytes()))
                        if i == 0:
                            fisrt_chunk_send_time  = time.time() - start_time
                        chunk_count += 1
                
                # end
                result_queue.put((task_id, 'end', None))
                duration = time.time() - start_time
                self.logger.info(
                    f"{current_process().name} Task {task_id} completed: "
                    f"{len(text)} chars, {chunk_count} chunks, {duration:.2f}s, "
                    f"first_chunk_time {fisrt_chunk_time:.2f}s, "
                    f"first_chunk_send_time {fisrt_chunk_send_time:.2f}s"
                )
                
            except Exception as e:
                self.logger.error(f"{current_process().name} Error in task {task_id}: {str(e)}", exc_info=True)
                result_queue.put((task_id, 'error', str(e)))

class TTSService:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TTSService")
        # Use Manager to create shared objects
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.workers = []
        self.task_queues = {}  # task_id -> result_queue
        
        # create worker
        for i in range(NUM_WORKERS):
            worker = TTSWorker(self.task_queue, self.manager)
            worker.start()
            self.workers.append(worker)
            self.logger.info(f"Worker {i} started")

    def submit_task(self, text, voice_type="温柔女"):
        task_id = str(uuid.uuid4())
        # Use Manager to create result queue
        result_queue = self.manager.Queue()
        
        # Store task queue
        self.task_queues[task_id] = result_queue
        
        # push task
        self.task_queue.put((task_id, text, voice_type, result_queue))
        self.logger.info(f"Task submitted: {task_id}")
        return task_id

    def stream_generator(self, task_id):
        """Generator function for streaming responses"""
        if task_id not in self.task_queues:
            self.logger.warning(f"Task {task_id} not found in task queues")
            yield b''
            return
            
        result_queue = self.task_queues[task_id]
        
        try:
            while True:
                try:
                    # Block until audio chunk arrives or timeout occurs
                    data = result_queue.get(timeout=CHUNK_TIMEOUT)
                    recv_task_id, msg_type, content = data
                    
                    if recv_task_id != task_id:
                        # Return data to queue if it's not for the current task
                        result_queue.put(data)
                        time.sleep(0.01)
                        continue
                    
                    if msg_type == 'data':
                        yield content  # Stream audio payload to client
                    elif msg_type == 'end':
                        self.logger.info(f"Task {task_id} completed successfully")
                        break
                    elif msg_type == 'error':
                        self.logger.error(f"Task {task_id} failed: {content}")
                        yield b''  # Signal error by sending null/empty packet
                        break
                        
                except queue.Empty:
                    self.logger.warning(f"Task {task_id} timed out waiting for audio chunk")
                    break
                    
        finally:
            # Release allocated resources
            if task_id in self.task_queues:
                del self.task_queues[task_id]
                self.logger.debug(f"Cleaned up resources for task {task_id}")

# Bootstrap service components
app = Flask(__name__)
app.logger = logging.getLogger(f"{__name__}.Flask")
tts_service = TTSService()

@app.route('/tts/fast/stream', methods=['POST'])
def tts_endpoint():
    data = request.json
    if not data or 'text' not in data:
        app.logger.warning("Missing text parameter in request")
        return jsonify({"error": "Missing text parameter"}), 400
    
    text = data['text']
    voice_type = data.get('voice_type', '温柔女')
    
    # Dispatch task to worker queue
    task_id = tts_service.submit_task(text, voice_type)
    app.logger.info(f"Task {task_id} created for text: {text[:50]}...")
    
    # Instantiate streaming response object
    response = Response(
        stream_with_context(tts_service.stream_generator(task_id)),
        mimetype="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=tts_{task_id}.wav",
            "X-Task-ID": task_id
        }
    )
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    app.logger.debug("Health check requested")
    return jsonify({
        "status": "ok",
        "workers": NUM_WORKERS,
        "active_tasks": len(tts_service.task_queues)
    })

def shutdown_workers():
    app.logger.info("Shutting down workers...")
    for _ in range(len(tts_service.workers)):
        tts_service.task_queue.put(None)
    for worker in tts_service.workers:
        worker.join()
    # shut down Manager
    tts_service.manager.shutdown()
    app.logger.info("All workers shut down")

if __name__ == '__main__':
    import atexit
    atexit.register(shutdown_workers)
    
    app.logger.info(f"TTS Service started on port {PORT}")
    app.logger.info(f"Streaming mode: {'enabled' if STREAM else 'disabled'}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)