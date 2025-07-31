from openai import AsyncOpenAI  # 异步客户端
from typing import AsyncGenerator, List, Dict, Any
import json
import asyncio
import re

class LLMModule:
    def __init__(self, config: dict):
        self.api_url = config['api_url']
        self.api_key = config['api_key']  # 默认API密钥
        self.model_name = config['model_name']
        self.system_prompt = config['system_prompt']
        self.client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key
        )
        self.sentence_delimiters = ['.', '?', '!', '。', '？', '！', ';', '；']
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """流式调用大模型API，按句子边界返回结果，并过滤纯换行符"""
        # 创建流式聊天请求
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            max_tokens=1024,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        
        # 缓冲区用于累积字符直到形成完整句子
        buffer = ""
        
        # 处理流式响应
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                buffer += token
                
                # 检查缓冲区中是否有完整的句子
                sentence, remaining = self._extract_complete_sentence(buffer)
                if sentence:
                    buffer = remaining
                    # 过滤掉纯换行符或空白的句子
                    if self._is_valid_sentence(sentence):
                        yield sentence.strip()
        
        # 返回剩余内容，同样过滤无效句子
        if buffer and self._is_valid_sentence(buffer):
            yield buffer.strip()
    
    def _extract_complete_sentence(self, text: str) -> (str, str):
        """
        从文本中提取第一个完整的句子
        返回：(完整句子, 剩余文本)
        """
        # 查找第一个句子结束位置
        min_index = len(text)
        for delimiter in self.sentence_delimiters:
            index = text.find(delimiter)
            if 0 <= index < min_index:
                min_index = index + len(delimiter)
        
        # 如果找到句子结束符
        if min_index < len(text):
            return text[:min_index], text[min_index:]
        
        # 如果没有找到句子结束符，检查是否超过最大长度
        if len(text) > 20:  # 最大句子长度阈值
            # 尝试在最后一个空格处分割
            last_space = text.rfind(' ')
            if last_space > 0:
                return text[:last_space], text[last_space + 1:]
            return text, ""
        
        return None, text
    
    def _is_valid_sentence(self, sentence: str) -> bool:
        """检查句子是否有效（不是纯换行符或空白）"""
        # 去除空白字符后检查是否为空
        stripped = re.sub(r'\s+', '', sentence)
        # 检查是否包含至少一个非空白字符
        return bool(stripped)

# 测试代码
if __name__ == "__main__":
    import asyncio
    import yaml
    with open("/mnt/sfs_turbo/botcall/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # 加载配置
    llm_config = config['llm']
    
    # 创建LLM模块实例
    llm = LLMModule(llm_config)
    
    import time
    start = time.time()
    async def test_stream():
        async for sentence in llm.generate_stream("测试文本"):
            end = time.time() - start
            print(f"生成句子: {sentence}, time:{end:.3f}s")
    
    asyncio.run(test_stream())