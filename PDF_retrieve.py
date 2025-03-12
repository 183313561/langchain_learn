print("hello world")

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},       # meta data中 source表明了数据的来源
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

from langchain_community.document_loaders import PyPDFLoader

file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

# 分割

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


# embedding

from langchain_openai import OpenAIEmbeddings

# 创建OpenAIEmbeddings实例，传入自定义URL、API密钥和模型名称
# embeddings = OpenAIEmbeddings(
#     openai_api_base="https://api.siliconflow.cn/v1/",
#     openai_api_key="sk-fhiarggsenvokytospnfdujbhoesbeoszesxovjfegzcygfi",    # app_key
#     model="BAAI/bge-m3",   # 模型名称
# )


from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="bge-m3")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])


# 建立向量数据库

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(all_splits, embeddings)


# 查询的方法：
# 同步和异步；
# 按字符串查询

results = vector_store.similarity_search(
    "耐克在美国有多少个配送中心？"
)

print(results[0])


# 和按向量；
# 返回和不返回相似性分数；
# 按相似性和最大边际相关性（平衡相似性和查询，以实现检索结果的多样性）。

