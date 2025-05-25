import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.globals import set_verbose, set_debug, set_llm_cache
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig


def load_env():
    """
    加载环境变量
    """
    load_dotenv("prompt_guide/env_config.env")

    # 验证加载是否成功（可选）
    print("已加载环境变量:")
    print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE')}")
    print(f"LLM_NAME: {os.getenv('LLM_NAME')}")


def get_completion(prompt: str, model: BaseChatModel) -> str:
    """
    调用模型生成文本
    :param prompt: 提示词
    :param model: 模型对象
    :return: 模型生成的文本
    """
    messages = [{"role": "user", "content": prompt}]
    # 必须显示传入metadata，否则langsmith看不到model_name和temperature
    response = model.invoke(messages, config=RunnableConfig(metadata={
        "model_name": model.model_name,
        "temperature": model.model_kwargs.get("temperature")  # 从model_kwargs获取
    }))
    return response.content
# 我们使用通义千问Qwen-plus模型来演示
# 加载模型的环境变量
load_env()
# 创建chatModel对象
set_verbose(True)
set_debug(True)
set_llm_cache(None)
chat_model = ChatTongyi(model=os.getenv("LLM_NAME"), model_kwargs={
    "temperature": 0.0,
}, api_key=os.getenv("DASHSCOPE_API_KEY"))
text = f"""
       你应该提供尽可能清晰、具体的指示，以表达你希望模型执行的任务。\
       这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
       不要将写清晰的提示与写简短的提示混淆。\
       在许多情况下，更长的提示可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
       """
prompt = f"""
       把用三个反引号括起来的文本总结成一句话。
       ```{text}```"""
# 指令内容，使用 ``` 来分隔指令和待总结的内容,这样模型能够非常清楚地知道它应该总结的确切文本。因此，分隔符可以是任何清晰的标点符号，将特定文本与提示的其余部分分开。这可以是三个反引号，你可以使用引号，也可以使用XML标签、章节标题，只要让模型清楚地知道这是一个单独的部分
content = get_completion(prompt, chat_model)

prompt = f"""
请生成包括书名、作者和类别的三本虚构书籍清单，\
并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。不要输出json以外的内容。
"""
response = get_completion(prompt, chat_model)
print(response)