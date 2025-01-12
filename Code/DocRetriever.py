import os
from langchain.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain_text_splitters import CharacterTextSplitter

curpath = os. getcwd()
code_path = os.path.join(curpath, 'Code')
data_path = os.path.join(curpath, 'Data')
raw_data_path = os.path.join(data_path, 'RawData')
result_data_path = os.path.join(data_path, 'ResultData')
pdf_data_path = os.path.join(raw_data_path, '원료-혼합물질_MSDS-국문(200028)-3.pdf')
pdf_data_path = os.path.join(raw_data_path, '원료-혼합물질_MSDS-국문(200028).pdf')

loader = PyPDFium2Loader(pdf_data_path)

pages = loader.load()