import os

from langchain_core.globals import set_verbose, set_debug, set_llm_cache

from langchain.config_load import load_env_file

load_env_file('./env_config.env')

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

set_verbose(True)
set_debug(True)
set_llm_cache(False)
# 初始化模型(千问plus)
model = ChatTongyi(model_name="qwen-plus", temperature=0.1, api_key=os.getenv("DASHSCOPE_API_KEY"))

# 直接调用
messages = [
    HumanMessage(content="请你介绍一下你自己")
]
result = model.invoke(messages)

print(f"响应：{result.content} \nTokens 用量：{result.response_metadata['token_usage']}")
