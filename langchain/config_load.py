from dotenv import load_dotenv
import os

def load_env_file(file_path):
    """
    加载.env格式的环境变量文件
    参数:
        file_path: .env文件路径
    """
    load_dotenv(file_path)

    # 验证加载是否成功（可选）
    print("已加载环境变量:")
    print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE')}")
    print(f"LLM_NAME: {os.getenv('LLM_NAME')}")

# 使用示例
# load_env_file('./env_config.env')
