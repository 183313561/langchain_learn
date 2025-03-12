import os

from langchain_openai import ChatOpenAI

os.environ["LANGCHIAN_TRACING_V2"] = "true"




from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(  # 通过prompt的方式，让LLM只提取函数中所有的数据成员
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Only return valid JSON objects, do not add any additional text, interpretation or formatting!

Passage:
{input}

JSON 响应：
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


model = ChatOpenAI(
    openai_api_base="https://api.siliconflow.cn/v1/",
    openai_api_key="sk-fhiarggsenvokytospnfdujbhoesbeoszesxovjfegzcygfi",    # app_key
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # 模型名称
    temperature=0
).with_structured_output(Classification)

inp = "you are such a good person"
prompt = tagging_prompt.invoke({"input": inp})
response = model.invoke(prompt)

print(response)