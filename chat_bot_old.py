import os

from langchain_openai import ChatOpenAI

os.environ["LANGCHIAN_TRACING_V2"] = "true"

model = ChatOpenAI(
    openai_api_base="https://api.siliconflow.cn/",
    openai_api_key="sk-fhiarggsenvokytospnfdujbhoesbeoszesxovjfegzcygfi",    # app_key
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # 模型名称
)

from langchain_core.messages import HumanMessage

# temp = model.invoke([HumanMessage(content="Hi! I'm Bob")])
# print(temp)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:  #注解用法，表明这个函数返回的类型是BaseChatmessgaeHisotry
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

config1 = {"configurable": {"session_id": "abc2"}}     # configurable 是性质：可运行的  session_id是让robot知道这是abc2的这段对话

response1 = with_message_history.invoke(               # 第一次调用LLM
    [HumanMessage(content="Hi! I'm Bob")],
    config=config1,
)

config2 = {"configurable": {"session_id": "abc3"}}     # configurable 是性质：可运行的  session_id是让robot知道这是abc2的这段对话

response2 = with_message_history.invoke(               # 调用LLM
    [HumanMessage(content="你好我是小兰")],
    config=config2,
)

# print(response1.content)   
# print(response2.content)


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
# chain = prompt | model

# with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# config = {"configurable": {"session_id": "abc5"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Jim")],
#     config=config,
# )



# with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# config3 = {"configurable": {"session_id": "abc5"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Jim")],
#     config=config3,
# )

# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config3,
# )

# print(response.content)




# 更加复杂的提示词

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
)

# print(response.content)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm 兰")], "language": "Chinese"},
    config=config,
)

response.content

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Chinese"},
    config=config,
)

print(response.content)