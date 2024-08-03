# 파일: ai_assistant.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from store_db import DocumentProcessor  # DocumentProcessor를 사용한다고 가정합니다.

# 환경 변수 로드 (API 키 등)
load_dotenv()

# AI Assistant 클래스를 정의합니다.
class AIAssistant:
    def __init__(self, model_name='gpt-4o', temperature=0, api_key=None, vector_db_path="./vectordb"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = ChatOpenAI(model=model_name, temperature=temperature, api_key=self.api_key)
        self.processor = DocumentProcessor(api_key=self.api_key, vector_db_path=vector_db_path)
        
        # 프롬프트 템플릿 정의
        SYS_PROMPT = '''
        너는 강남대학교 AI 로드맵을 위한 참고 자료를 제공하는 시스템이야.
        '''
        template = SYS_PROMPT + '''
        1.강남대학교 AI로드맵을 위한 참고 자료야 :{context}
        2.사용자 입력 메세지에 잘 따라줘야해 : {question}
        '''
        self.prompt = ChatPromptTemplate.from_template(template)

    def format_docs(self, docs):
        """문서의 내용을 포맷팅하여 하나의 문자열로 반환합니다."""
        return '\n\n'.join(doc.page_content for doc in docs)

    def get_rag_response(self, question):
        """질문에 대한 RAG 체인의 응답을 생성합니다."""
        retriever = self.processor.load_db()
        
        rag_chain = (
            {'context': retriever | self.format_docs, 'question': RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

if __name__ == "__main__":
    # AI Assistant 인스턴스 생성
    assistant = AIAssistant()

    # 사용자 질문 설정 (예시)
    user_question = "AI 로드맵에 대해 알려주세요."

    # 질문에 대한 RAG 응답 얻기
    response = assistant.get_rag_response(user_question)
    print(response)
