# 1. 使用极小的 Python 镜像作为基础
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装必要的系统工具（最简安装）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 【核心瘦身步】：强制安装 CPU 版本的 Torch
# 这能让镜像体积从 4GB 降到几百 MB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers sqlalchemy

# 6. 复制项目代码
COPY . .

# 7. 启动程序（端口设为 8000）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
