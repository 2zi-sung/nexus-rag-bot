import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# PDF 파일 경로 설정
curpath = os.getcwd()
data_path = os.path.join(curpath, 'Data')
raw_data_path = os.path.join(data_path, 'RawData')
pdf_data_path = os.path.join(raw_data_path, '원료-혼합물질_MSDS-국문(200028).pdf')

# 1. PDF 로드 및 텍스트 분할
loader = PyPDFLoader(pdf_data_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

# 2. OpenAI 임베딩 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 3. 벡터스토어 생성
db = Chroma.from_documents(docs, embedding_model)

# 4. 검색기 생성 (RetrievalQA)
retriever = db.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 검색
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),  # OpenAI GPT-4 사용
    retriever=retriever
)

# 5. 챗봇 실행
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

# 실행
chatbot()
