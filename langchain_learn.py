import os  
# os.environ["OPENAI_API_KEY"] = "sk-fhiarggsenvokytospnfdujbhoesbeoszesxovjfegzcygfi"

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# model = ChatOllama(model="llama3.1", temperature=0.7)

# model = ChatOpenAI(model="gpt-4")
from langchain_core.messages import HumanMessage,SystemMessage

llm = ChatOpenAI(
    openai_api_base="https://api.siliconflow.cn/",
    openai_api_key="sk-fhiarggsenvokytospnfdujbhoesbeoszesxovjfegzcygfi",    # app_key
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # 模型名称
)

messages = [
    SystemMessage(content="把这段话从中文翻译成意大利语"),
    HumanMessage(content="你好")

]

ans = llm.invoke(messages)

print(ans.pretty_print)