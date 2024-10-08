from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

client = QianfanChatEndpoint(streaming=True, model="ERNIE-Lite-8K")

embedding = QianfanEmbeddingsEndpoint()
