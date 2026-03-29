import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

XIAOAI_API_KEY = os.getenv("XIAOAI_API_KEY")
XIAOAI_BASE_URL = os.getenv("XIAOAI_BASE_URL")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_BASE_URL = os.getenv("BAILIAN_BASE_URL")

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 本地 query-aware 压缩（utils/query_compress）；示例：Qwen/Qwen2.5-1.5B-Instruct 或 Qwen/Qwen3.5-4B
COMPRESS_MODEL_ID = os.getenv("COMPRESS_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ENABLE_QUERY_COMPRESS = os.getenv("ENABLE_QUERY_COMPRESS", "true").lower() in ("1", "true", "yes")

# print(XIAOAI_API_KEY)
