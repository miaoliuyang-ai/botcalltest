#!/usr/bin/env python3
"""
多进程服务器启动脚本
"""

import os
import sys
import logging
import signal
import time

# 添加必要的路径
sys.path.append("/mnt/sfs_turbo/botcall")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multiprocess_server.log")
    ]
)
logger = logging.getLogger("MultiprocessServer")

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在关闭服务器...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("启动多进程AI外呼助手服务器...")
    
    try:
        # 导入并运行服务器
        from orchestrator_copy import app, config, uvicorn
        
        server_config = config['server']
        logger.info(f"服务器配置: {server_config}")
        logger.info(f"工作进程数量: {config.get('server', {}).get('workers', 2)}")
        
        # 启动服务器
        uvicorn.run(
            app,
            host=server_config['host'],
            port=server_config['port'],
            log_level="info",
            timeout_keep_alive=30,
            ws_ping_interval=10,
            ws_ping_timeout=30
        )
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 