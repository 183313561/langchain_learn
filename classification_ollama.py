from langchain.chat_models import init_chat_model

llm = init_chat_model("qwen2.5:7b", model_provider="ollama")

print("成功")


from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="这段话的情感")
    aggressiveness: int = Field(
        description="这段话的攻击性，从1到10的等级"
    )
    language: str = Field(description="这段话使用的语言")


# LLM
llm = ChatOllama(temperature=0, model="qwen2.5:7b").with_structured_output(   # 这里其实涉及了工具的调用
    Classification
)

inp = "我日你仙人"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(response)