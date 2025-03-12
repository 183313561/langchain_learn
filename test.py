from openai import OpenAI
client = OpenAI(api_key="sk-fhiarggsenvokytospnfdujbhoesbeoszesxovjfegzcygfi", base_url="https://api.siliconflow.cn/v1/")
embedding = client.embeddings.create(input="sample text", model="BAAI/bge-m3")