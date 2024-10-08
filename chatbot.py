from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  # RetrievalQA链
from langchain.retrievers.multi_query import MultiQueryRetriever  # MultiQueryRetriever工具
import logging  # 导入Logging工具
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser, RetryWithErrorOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from chatmodel.qianfan import client, embedding
import gradio as gr
import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 1.Load 导入Document Loaders

# 加载Documents
base_dir = './data'  # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
vectorstore = Qdrant.from_documents(
    documents=chunked_documents,  # 以分块的文档
    embedding=embedding,  # 用OpenAI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",)  # 指定collection_name

# 4. Retrieval 准备模型和Retrieval链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=client)

memory = ConversationSummaryBufferMemory(
    memory_key="chat_history",
    llm=client,
    return_messages=True
)

# 实例化一个RetrievalQA链
qa_chain = ConversationalRetrievalChain.from_llm(
    client, retriever=retriever_from_llm, memory=memory)


def get_response(question):
    result = qa_chain.invoke(question)
    print("result", result)
    return result["answer"]


def respond(message, chat_history):
    bot_message = get_response(message)
    chat_history.append((message, bot_message))
    return "", chat_history


with gr.Blocks(css="footer{display:none !important}") as demo:
    chatbot = gr.Chatbot(height=240)  # 对话框
    msg = gr.Textbox(label="Prompt", autoscroll=True)  # 输入框
    btn = gr.Button("Submit")  # 提交按钮
    # 提交
    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
gr.close_all()
demo.launch()
