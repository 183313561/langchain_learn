import os  

os.environ["LANGCHIAN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from fastapi import FastAPI
from langserve import add_routes

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


# 使用解析器

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = llm | parser
# print(chain.invoke(messages))

# 接下来使用提示词模板 

from langchain_core.prompts import ChatPromptTemplate

system_template = "把这段翻译成{language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), 
     ("user","{text}")
    ]
)

result = prompt_template.invoke({"language":"japanese","text":"hi"})

chain = prompt_template | llm | parser

# print(chain.invoke({"language":"japanese","text":"hi"}))

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)


add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)