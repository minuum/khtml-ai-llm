import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.docstore.document import Document

# 환경 변수 로드
load_dotenv()

class DocumentProcessor:
    def __init__(self, api_key, vector_db_path="./vectordb"):
        self.api_key = api_key
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)

    def load_text_files(self, file_directory):
        documents = []
        print("Start to load txts.")
        text_files = glob(os.path.join(file_directory, '*.txt'))
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as file:
                text = file.read()
                text_document = Document(page_content=text, metadata={"source": text_file})
                documents.append(text_document)
        print(f"Loaded {len(documents)} text documents.")
        return documents

    def load_pdf_files(self, file_directory):
        documents = []
        print("Start to load PDFs.")
        pdf_files = glob(os.path.join(file_directory, '*.pdf'))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pdf_documents = loader.load()
            documents.extend(pdf_documents)
        print(f"Loaded {len(documents)} PDF documents.")
        return documents

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunk_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks

    def save_db(self, chunks):
        if not chunks:
            print("No chunks to save to VectorDB.")
            return None
        print("Save VectorDB")
        vectordb = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=self.vector_db_path)
        vectordb.persist()
        return vectordb.as_retriever()
    ##자동저장
    # def save_db(self, chunks):
    #     if not chunks:
    #         print("No chunks to save to VectorDB.")
    #         return None
    #     print("Saving to VectorDB (automatic persistence).")
    #     vectordb = Chroma.from_documents(
    #         documents=chunks,
    #         embedding=self.embeddings,
    #         persist_directory=self.vector_db_path
    #     )
    #     return vectordb.as_retriever()

    def process_documents(self, file_directory, data_format="txt", save_db=True, chunk_size=1000, chunk_overlap=200):
        if data_format == "txt":
            documents = self.load_text_files(file_directory)
        elif data_format == "pdf":
            documents = self.load_pdf_files(file_directory)
        else:
            print("Wrong data format.")
            return None

        chunks = self.split_documents(documents, chunk_size, chunk_overlap)

        if save_db:
            return self.save_db(chunks)
        else:
            vectordb = Chroma.from_documents(documents=chunks, embedding=self.embeddings)
            return vectordb.as_retriever()

    def load_db(self):
        loaded_vectordb = Chroma(persist_directory=self.vector_db_path, embedding_function=self.embeddings)
        return loaded_vectordb.as_retriever()

if __name__ == "__main__":
    # 환경 변수에서 OpenAI API 키를 가져옴
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # DocumentProcessor 인스턴스 생성
    processor = DocumentProcessor(api_key=OPENAI_API_KEY)
    
    # 사용할 파일 디렉토리와 데이터 형식 설정
    file_directory = "/home/user/khtml-ai-llm/data/academic resources"
    data_format = "pdf"
    
    # 벡터 데이터베이스 생성 및 저장
    retriever = processor.process_documents(file_directory, data_format=data_format)
    if retriever:
        print("Database created and retriever initialized.")
    else:
        print("Database not saved.")
