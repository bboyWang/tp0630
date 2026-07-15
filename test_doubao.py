"""Test Volcengine/Doubao API connectivity using OpenAI-compatible endpoint."""
import os
import json

# ====== 把你的 API Key 填在这里 ======
API_KEY = os.getenv("ARK_API_KEY", "")  # 或直接写 "your-api-key-here"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-1-pro-260628"
# =====================================

if not API_KEY:
    API_KEY = input("请输入你的火山引擎 API Key: ").strip()

try:
    from openai import OpenAI
except ImportError:
    print("❌ 请先安装 openai: pip install openai")
    exit(1)

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

print(f"🔗 连接 {BASE_URL} ...")
print(f"🤖 模型: {MODEL}")

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
        max_tokens=200,
    )
    print(f"✅ 连接成功！")
    print(f"📝 回复: {response.choices[0].message.content}")
    print(f"📊 用量: {response.usage}")

except Exception as e:
    print(f"❌ 连接失败: {e}")
