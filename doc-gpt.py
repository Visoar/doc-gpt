import os
import openai
import pypdf
import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAIChat
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader

@st.cache_data
def split_pdf(fpath,chunk_chars=4000,overlap=50):
    """
    Pre-process PDF into chunks
    Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
    """
    pdfReader = pypdf.PdfReader(fpath)
    splits = []
    split = ""
    pages = []
    for i, page in enumerate(pdfReader.pages):
        pages.append(str(i + 1))
        split += page.extract_text()
        if len(split) > chunk_chars:
            splits.append(split[:chunk_chars])
            split = split[chunk_chars - overlap:]
        if len(split) <= chunk_chars:
            splits.append(split)
    return splits

@st.cache_resource
def create_ix(splits):
    """ 
    Create vector DB index of PDF
    """
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(splits,embeddings)
    return docsearch

st.sidebar.header("PDF 阅读助手")
# st.sidebar.image("Img/reading.jpg")
# Auth & Setting
api_key = st.sidebar.text_input("OpenAI API Key:", type="password")

os.environ["OPENAI_API_KEY"] = api_key
chunk_chars = st.sidebar.radio("选择要拆分的「块」大小", (3000, 3500, 4000), index=1)
langInfo = st.sidebar.radio("选择期望回复所用语言",("使用中文回复","Reply in English"))
st.sidebar.info("`更大的「块」大小可以产生更好的答案，但可能会超过ChatGPT上下文限制（4096个tokens）。`")

# App 
# st.write("你好！我是 ChatGPT，请上传你要解读的 PDF 文档。")
uploaded_file_pdf = st.file_uploader("上传 PDF 文件: ", type = ['pdf'] , accept_multiple_files=False)
if uploaded_file_pdf and api_key:
    # Split and create index
    with st.spinner("文档读取和拆分中..."):
        d=split_pdf(uploaded_file_pdf,chunk_chars)
    with st.spinner("索引创建中..."):
        ix=create_ix(d)
    # Use ChatGPT with index QA chain
    llm = OpenAIChat(temperature=0)
    chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=ix)
    query = st.text_input("请输入问题后按回车： ","总结这个文件的主要内容。")
    message(query, is_user=True) 
    query = query + "。" + langInfo
    print(query)
    try:
        with st.spinner("思考中..."):
            reply = chain.run(query)
            # st.info(reply)
            message(reply)
    except openai.error.InvalidRequestError:
        # Limitation w/ ChatGPT: 4096 token context length
        # https://github.com/acheong08/ChatGPT/discussions/649
        st.warning('模型请求出错，通常是由于上下文长度问题。尝试减少「块」的大小。', icon="⚠️")

else:
    st.info("请输入 OpenAI Key 并上传 PDF 文件")
