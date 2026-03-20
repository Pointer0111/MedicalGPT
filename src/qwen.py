import os
from dashscope import Generation
import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"},
]
response = Generation.call(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-flash",
    messages=messages,
    result_format="message",
    # 开启深度思考
    enable_thinking=False,
)


print("=" * 20 + "完整回复" + "=" * 20)
print(response.output.choices[0].message.content)