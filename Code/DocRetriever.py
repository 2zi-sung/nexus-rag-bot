import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from TokenKey import api_key

os.environ["OPENAI_API_KEY"] = api_key

curpath = os.getcwd()
data_path = os.path.join(curpath, 'Data')
raw_data_path = os.path.join(data_path, 'RawData')
pdf_data_path = os.path.join(raw_data_path, '원료-혼합물질_MSDS-국문(200028).pdf')

loader = PyPDFLoader(pdf_data_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

db = Chroma.from_documents(docs, embedding_model)

retriever = db.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 검색
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),  # OpenAI GPT-4 사용
    retriever=retriever
)

def chatbot():
    print("PDF 기반 질문-응답 챗봇입니다. '종료'라고 입력하면 프로그램이 종료됩니다.")
    while True:
        query = input("\n질문을 입력하세요: ")
        if query.lower() == "종료":
            print("프로그램을 종료합니다.")
            break
        try:
            answer = qa_chain.run(query)
            print("\n답변:", answer)
        except Exception as e:
            print("\n오류가 발생했습니다:", e)

chatbot()
